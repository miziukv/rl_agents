import os
import time
import math
import argparse
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.exception import UnityWorkerInUseException
from torch.utils.tensorboard import SummaryWriter


@dataclass
class SACCfg:
    total_env_steps: int = 500_000
    init_random_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 10
    gradient_steps: int = 1
    batch_size: int = 128
    gamma: float = 0.995
    tau: float = 0.005
    lr: float = 3e-4
    hidden: int = 256
    target_entropy_scale: float = 0.5
    replay_size: int = 1_000_000
    device: str = "cuda"
    save_every: int = 100_000
    out_dir: str = "./results/sac_course"
    summary_freq: int = 20000
    worker_id: int = 0
    base_port: int = 5004

cfg = SACCfg()


class Replay:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.size = size
        self.ptr = 0
        self.count = 0
        self.device = device

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.count, size=batch)
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.act[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.rew[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.done[idx], dtype=torch.float32, device=self.device),
        )


def mlp(in_dim, out_dim, hidden, act=nn.ReLU):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), act(),
        nn.Linear(hidden, hidden), act(),
        nn.Linear(hidden, out_dim),
    )


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 2*act_dim, hidden)
        self.act_dim = act_dim
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, obs):
        out = self.net(obs)
        mu, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)
        logp = (-0.5 * (((pre_tanh - mu) / (std + 1e-6))**2 + 2*log_std + math.log(2*math.pi))).sum(-1)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp


class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACAgent:
    def __init__(self, obs_dim, act_dim, cfg: SACCfg):
        self.device = torch.device(cfg.device)
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.actor = GaussianPolicy(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic = QCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_tgt = QCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.log_alpha = None
        self.alpha = torch.tensor(0.2, device=self.device)
        self.opt_alpha = None
        self.target_entropy = None

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.cfg = cfg

    def act(self, obs_t, deterministic=False):
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(obs_t)
                return torch.tanh(mu)
            a, _ = self.actor.sample(obs_t)
            return a

    def update(self, replay: Replay):
        o, a, r, no, d = replay.sample(self.cfg.batch_size)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(no)
            q1_t, q2_t = self.critic_tgt(no, a2)
            q_tgt = torch.min(q1_t, q2_t) - self.alpha * logp2
            y = r + (1.0 - d) * self.gamma * q_tgt

        q1, q2 = self.critic(o, a)
        critic_loss = ((q1 - y).pow(2).mean() + (q2 - y).pow(2).mean())

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        a_pi, logp = self.actor.sample(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)
            q_mean = (q1 + q2).mean() / 2.0
        
        return {
            "critic": critic_loss.item(),
            "actor": actor_loss.item(),
            "alpha": self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
            "alpha_loss": 0.0,
            "q_value": q_mean.item()
        }


def connect_unity(worker_id=None, base_port=None):
    if worker_id is None:
        worker_id = cfg.worker_id
    if base_port is None:
        base_port = cfg.base_port
    
    if worker_id != 0:
        print("Warning: Unity Editor requires worker_id=0, using 0")
        worker_id = 0
    
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=20, quality_level=0, width=640, height=480)
    
    try:
        env = UnityEnvironment(file_name=None, side_channels=[engine], worker_id=0, base_port=base_port)
    except UnityWorkerInUseException:
        print(f"\nError: Port {base_port} is already in use!")
        print("Close other ML-Agents processes or kill the process using this port.")
        raise RuntimeError(f"Port {base_port} is in use")
    
    env.reset()
    behavior = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior]
    
    if spec.action_spec.continuous_size == 0:
        raise ValueError(f"Expected continuous actions, got discrete_size={spec.action_spec.discrete_size}")
    
    print("Waiting for agents...")
    dec, term = env.get_steps(behavior)
    
    if len(dec) == 0 and len(term) == 0:
        dummy_continuous = np.zeros((1, spec.action_spec.continuous_size), dtype=np.float32)
        if spec.action_spec.discrete_size > 0:
            dummy_discrete = np.zeros((1, spec.action_spec.discrete_size), dtype=np.int32)
            dummy_actions = ActionTuple(continuous=dummy_continuous, discrete=dummy_discrete)
        else:
            dummy_actions = ActionTuple(continuous=dummy_continuous)
        
        env.set_actions(behavior, dummy_actions)
        env.step()
        dec, term = env.get_steps(behavior)
    
    if len(dec) == 0 and len(term) == 0:
        raise RuntimeError("Unity environment did not produce any agents")
    
    print(f"Connected: {len(dec)} decision agents, {len(term)} terminal agents")
    
    obs_dim = spec.observation_specs[0].shape[0]
    act_dim = spec.action_spec.continuous_size
    discrete_size = spec.action_spec.discrete_size
    return env, behavior, spec, obs_dim, act_dim, discrete_size


def unity_step(env, behavior, actions_np: np.ndarray, spec=None, discrete_actions=None):
    num_agents = actions_np.shape[0] if actions_np.ndim == 2 else 1
    
    if discrete_actions is None and spec is not None and spec.action_spec.discrete_size > 0:
        discrete_actions = np.zeros((num_agents, spec.action_spec.discrete_size), dtype=np.int32)
    
    if discrete_actions is not None:
        actions = ActionTuple(continuous=actions_np, discrete=discrete_actions)
    else:
        actions = ActionTuple(continuous=actions_np)
    
    env.set_actions(behavior, actions)
    
    try:
        env.step()
    except Exception as e:
        if "timeout" in str(e).lower() or "took too long" in str(e).lower():
            print("\nError: Unity Editor is not responding!")
            print("Make sure Unity is running in Play mode with agents in the scene.")
        raise
    
    return env.get_steps(behavior)


def process_episode(dec, term, rewards, episode_rewards, episode_lengths, 
                   episode_reward_history, episode_length_history, episode_count, writer):
    """Process episode completion and log metrics."""
    for i, aid in enumerate(dec.agent_id):
        if aid not in episode_rewards:
            episode_rewards[aid] = 0.0
            episode_lengths[aid] = 0
        
        episode_rewards[aid] += rewards[i]
        episode_lengths[aid] += 1
        
        if aid in term:
            episode_reward_history.append(episode_rewards[aid])
            episode_length_history.append(episode_lengths[aid])
            episode_count += 1
            
            if episode_count % 10 == 0:
                writer.add_scalar('Environment/Cumulative Reward', episode_rewards[aid], episode_count)
                writer.add_scalar('Environment/Episode Length', episode_lengths[aid], episode_count)
                if len(episode_reward_history) >= 10:
                    writer.add_scalar('Environment/Mean Cumulative Reward', 
                                    np.mean(episode_reward_history), episode_count)
                    writer.add_scalar('Environment/Mean Episode Length', 
                                    np.mean(episode_length_history), episode_count)
            
            del episode_rewards[aid]
            del episode_lengths[aid]
    
    return episode_count


def collect_transitions(dec, term, dec2, obs):
    """Collect transitions from Unity environment."""
    rewards = np.zeros(len(dec), dtype=np.float32)
    dones = np.zeros(len(dec), dtype=np.float32)
    next_obs = np.zeros_like(obs, dtype=np.float32)

    for i, aid in enumerate(dec.agent_id):
        if aid in term:
            rewards[i] = term[aid].reward
            dones[i] = 1.0
            next_obs[i] = term[aid].obs[0]
        else:
            rewards[i] = dec2[aid].reward
            next_obs[i] = dec2[aid].obs[0]

    return rewards, dones, next_obs


def train(run_id=None):
    if run_id is not None:
        cfg.out_dir = f"./results/{run_id}"
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    env, behavior, spec, obs_dim, act_dim, discrete_size = connect_unity()
    print(f"behavior={behavior} obs_dim={obs_dim} act_dim={act_dim} discrete_size={discrete_size}")

    agent = SACAgent(obs_dim, act_dim, cfg)
    replay = Replay(obs_dim, act_dim, cfg.replay_size, agent.device)

    writer = SummaryWriter(log_dir=cfg.out_dir)
    print(f"Logging to {cfg.out_dir}")

    env_steps = 0
    last_save = 0
    stats = deque(maxlen=100)
    
    episode_rewards = {}
    episode_lengths = {}
    episode_count = 0
    episode_reward_history = deque(maxlen=100)
    episode_length_history = deque(maxlen=100)

    # Random exploration phase
    while env_steps < cfg.init_random_steps:
        dec, term = env.get_steps(behavior)
        
        if len(dec) == 0:
            if len(term) > 0:
                env.step()
            else:
                time.sleep(0.01)
            continue
        
        obs = dec.obs[0]
        a = np.random.uniform(-1.0, 1.0, size=(len(dec), act_dim)).astype(np.float32)
        dec2, term = unity_step(env, behavior, a, spec=spec)

        rewards, dones, next_obs = collect_transitions(dec, term, dec2, obs)
        episode_count = process_episode(dec, term, rewards, episode_rewards, episode_lengths,
                                       episode_reward_history, episode_length_history, episode_count, writer)

        for i in range(len(rewards)):
            replay.add(obs[i], a[i], rewards[i], next_obs[i], dones[i])
        env_steps += len(rewards)

    # Main training loop
    while env_steps < cfg.total_env_steps:
        dec, term = env.get_steps(behavior)
        
        if len(dec) == 0:
            if len(term) > 0:
                env.step()
            continue
        
        obs = dec.obs[0]
        obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            a_t = agent.act(obs_t, deterministic=False)
        a = a_t.cpu().numpy().astype(np.float32)
        a = np.clip(a, -1.0, 1.0)
        
        if a.ndim == 1:
            a = a.reshape(1, -1)

        dec2, term = unity_step(env, behavior, a, spec=spec)

        rewards, dones, next_obs = collect_transitions(dec, term, dec2, obs)
        episode_count = process_episode(dec, term, rewards, episode_rewards, episode_lengths,
                                       episode_reward_history, episode_length_history, episode_count, writer)

        for i in range(len(rewards)):
            replay.add(obs[i], a[i], rewards[i], next_obs[i], dones[i])
        env_steps += len(rewards)

        if env_steps >= cfg.update_after and env_steps % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                metrics = agent.update(replay)
            
            stats.append(metrics["critic"])
            
            if env_steps % cfg.summary_freq == 0:
                writer.add_scalar('Policy/Critic Loss', metrics["critic"], env_steps)
                writer.add_scalar('Policy/Actor Loss', metrics["actor"], env_steps)
                writer.add_scalar('Policy/Alpha', metrics["alpha"], env_steps)
                writer.add_scalar('Policy/Alpha Loss', metrics["alpha_loss"], env_steps)
                writer.add_scalar('Policy/Value Estimate', metrics["q_value"], env_steps)
                writer.add_scalar('Policy/Learning Rate', cfg.lr, env_steps)
                
                with torch.no_grad():
                    entropy_batch_size = min(200, replay.count)
                    if entropy_batch_size > 0:
                        o_sample, _, _, _, _ = replay.sample(entropy_batch_size)
                        _, logp_sample = agent.actor.sample(o_sample)
                        entropy_estimate = -logp_sample.mean().item()
                        writer.add_scalar('Policy/Entropy', entropy_estimate, env_steps)
            
            if env_steps % 10_000 == 0:
                avg_qloss = sum(stats)/len(stats) if len(stats) > 0 else metrics["critic"]
                print(f"Steps {env_steps:>8} | Qloss {avg_qloss:.4f} | alpha {metrics['alpha']:.3f}")

        if env_steps - last_save >= cfg.save_every:
            alpha_value = agent.alpha.detach().cpu() if isinstance(agent.alpha, torch.Tensor) else agent.alpha
            
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "alpha": alpha_value,
                "cfg": cfg.__dict__,
                "obs_dim": obs_dim,
                "act_dim": act_dim
            }, os.path.join(cfg.out_dir, f"sac_{env_steps}.pt"))
            last_save = env_steps
            print(f"Saved checkpoint at {env_steps} steps.")

    writer.close()
    env.close()
    torch.save(agent.actor.state_dict(), os.path.join(cfg.out_dir, "sac_actor_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAC agent on Unity ML-Agents environment')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Run identifier for output directory')
    
    args = parser.parse_args()
    train(run_id=args.run_id)
