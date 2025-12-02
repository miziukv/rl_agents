import os, time, random, math
from dataclasses import dataclass
from typing import Tuple, Deque
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import DecisionSteps, TerminalSteps, ActionTuple

# -----------------------------
# Config
# -----------------------------
@dataclass
class SACCfg:
    total_env_steps: int = 500_000
    init_random_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 50
    gradient_steps: int = 50      # updates per update_every
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005            # target smoothing
    lr: float = 3e-4
    hidden: int = 256
    # entropy
    target_entropy_scale: float = 0.5  # target = -act_dim * scale (0.5~1.0 typical)
    # replay
    replay_size: int = 1_000_000
    # device
    device: str = "cpu"
    # logging/checkpoints
    save_every: int = 100_000
    out_dir: str = "./sac_ckpts"  # SAC checkpoints go here (PPO uses ./results)
    # Unity connection
    worker_id: int = 0
    base_port: int = 5004

cfg = SACCfg()

# -----------------------------
# Replay Buffer
# -----------------------------
class Replay:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, ), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, ), dtype=np.float32)
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
        o = torch.tensor(self.obs[idx], dtype=torch.float32, device=self.device)
        a = torch.tensor(self.act[idx], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.rew[idx], dtype=torch.float32, device=self.device)
        no = torch.tensor(self.next_obs[idx], dtype=torch.float32, device=self.device)
        d = torch.tensor(self.done[idx], dtype=torch.float32, device=self.device)
        return o, a, r, no, d

# -----------------------------
# Networks
# -----------------------------
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
        # reparameterization trick
        eps = torch.randn_like(mu)
        pre_tanh = mu + std * eps
        a = torch.tanh(pre_tanh)  # action in [-1,1]
        # log prob with tanh correction
        logp = (-0.5 * (((pre_tanh - mu) / (std + 1e-6))**2 + 2*log_std + math.log(2*math.pi))).sum(-1)
        # tanh correction: sum log(1 - tanh(x)^2)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

# -----------------------------
# SAC Agent
# -----------------------------
class SACAgent:
    def __init__(self, obs_dim, act_dim, cfg: SACCfg):
        self.device = torch.device(cfg.device)
        self.actor = GaussianPolicy(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic = QCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_tgt = QCritic(obs_dim, act_dim, cfg.hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Entropy coef (alpha) with auto-tuning
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg.lr)
        self.target_entropy = -act_dim * cfg.target_entropy_scale

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.cfg = cfg

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs_t, deterministic=False):
        with torch.no_grad():
            if deterministic:
                mu, log_std = self.actor(obs_t)
                a = torch.tanh(mu)
                return a
            a, _ = self.actor.sample(obs_t)
            return a

    def update(self, replay: Replay):
        o, a, r, no, d = replay.sample(self.cfg.batch_size)

        # Critic update
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

        # Actor update
        a_pi, logp = self.actor.sample(o)
        q1_pi, q2_pi = self.critic(o, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        # Alpha (entropy) update
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # Target update
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                p_t.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "critic": critic_loss.item(),
            "actor": actor_loss.item(),
            "alpha": self.alpha.item(),
            "alpha_loss": alpha_loss.item()
        }

# -----------------------------
# Unity env helpers
# -----------------------------
def connect_unity(worker_id=None, base_port=None):
    from mlagents_envs.exception import UnityWorkerInUseException
    
    if worker_id is None:
        worker_id = cfg.worker_id
    if base_port is None:
        base_port = cfg.base_port
    
    # Unity Editor requires worker_id=0 when file_name=None
    if worker_id != 0:
        print(f"[Warning] Unity Editor requires worker_id=0, but got {worker_id}. Using worker_id=0.")
        worker_id = 0
    
    engine = EngineConfigurationChannel()
    engine.set_configuration_parameters(time_scale=20, quality_level=0, width=640, height=480)
    
    # Try to connect to Unity Editor (file_name=None requires worker_id=0)
    try:
        env = UnityEnvironment(file_name=None, side_channels=[engine], worker_id=0, base_port=base_port)
    except UnityWorkerInUseException as e:
        print(f"\n[ERROR] Port {base_port} is already in use!")
        print("This usually means:")
        print("  1. Another Python script is using port 5004")
        print("  2. A previous Unity environment connection wasn't closed properly")
        print("\nTo fix this:")
        print("  - Close any other Python scripts using ML-Agents")
        print("  - Or kill the process using port 5004:")
        print(f"    Windows: netstat -ano | findstr :{base_port}")
        print(f"    Then: taskkill /PID <PID> /F")
        raise RuntimeError(f"Port {base_port} is in use. Please close other ML-Agents processes or kill the process using port {base_port}.")
    
    env.reset()
    behavior = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior]
    # Check if continuous actions are available
    print(f"[Debug] Action spec - continuous_size: {spec.action_spec.continuous_size}, discrete_size: {spec.action_spec.discrete_size}")
    if hasattr(spec.action_spec, 'discrete_branches'):
        print(f"[Debug] Discrete branches: {spec.action_spec.discrete_branches}")
    if spec.action_spec.continuous_size == 0:
        raise ValueError(f"Expected continuous action space, but got discrete_size={spec.action_spec.discrete_size}, continuous_size={spec.action_spec.continuous_size}. "
                         f"Please check Unity Behavior Parameters: Behavior Type should be 'Default' and Actions should be Continuous (Size=4).")
    
    # Wait for Unity to be ready and have agents
    # After reset(), we need to get_steps() to see if agents are available
    # If not, we may need to send dummy actions and step once
    print("[Unity] Waiting for agents to be ready...")
    dec, term = env.get_steps(behavior)
    
    # If no agents yet, send dummy actions and step once to initialize
    if len(dec) == 0 and len(term) == 0:
        # Create dummy actions for initialization
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
        raise RuntimeError("Unity environment did not produce any agents. Make sure Unity Editor is in Play mode and the scene has agents.")
    
    print(f"[Unity] Connected! Found {len(dec)} decision agents, {len(term)} terminal agents")
    
    obs_dim = spec.observation_specs[0].shape[0]
    act_dim = spec.action_spec.continuous_size
    discrete_size = spec.action_spec.discrete_size
    return env, behavior, spec, obs_dim, act_dim, discrete_size

def unity_step(env, behavior, actions_np: np.ndarray, spec=None, discrete_actions=None):
    # Unity may require discrete actions even if we only use continuous
    num_agents = actions_np.shape[0] if actions_np.ndim == 2 else 1
    
    if discrete_actions is None:
        if spec is not None and spec.action_spec.discrete_size > 0:
            # Create discrete actions with correct shape: (num_agents, discrete_branches)
            # discrete_size is the number of branches, each needs a value per agent
            # For now, just use zeros (we're not using discrete actions)
            discrete_actions = np.zeros((num_agents, spec.action_spec.discrete_size), dtype=np.int32)
        else:
            discrete_actions = None
    
    # Always provide discrete actions if spec requires them
    if discrete_actions is not None:
        actions = ActionTuple(continuous=actions_np, discrete=discrete_actions)
    else:
        actions = ActionTuple(continuous=actions_np)
    
    env.set_actions(behavior, actions)
    
    # Step the environment - Unity must be running and responsive
    try:
        env.step()
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "took too long" in error_msg.lower():
            print("\n[ERROR] Unity Editor is not responding!")
            print("Make sure:")
            print("  1. Unity Editor is running and in Play mode")
            print("  2. The scene has agents with Behavior Type = 'Default'")
            print("  3. Unity Editor is not frozen or paused")
            print("  4. ML-Agents versions match between Unity and Python")
        raise
    
    dec, term = env.get_steps(behavior)
    return dec, term

# -----------------------------
# Rollout collection
# -----------------------------
def collect_step(env, behavior, spec, agent: SACAgent, replay: Replay, deterministic=False):
    dec, term = env.get_steps(behavior)
    
    # Skip if no agents need decisions
    if len(dec) == 0:
        # If we have terminal agents, step to get new decision agents
        if len(term) > 0:
            env.step()
        return 0
    
    obs = dec.obs[0]  # [N, obs_dim]
    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        a_t = agent.act(obs_t, deterministic=deterministic)
    a = a_t.cpu().numpy().astype(np.float32)
    a = np.clip(a, -1.0, 1.0)
    
    # Ensure correct shape: (num_agents, act_dim)
    if a.ndim == 1:
        a = a.reshape(1, -1)

    dec2, term = unity_step(env, behavior, a, spec=spec)

    # map rewards/dones per agent id
    rewards = np.zeros(len(dec), dtype=np.float32)
    dones = np.zeros(len(dec), dtype=np.float32)
    next_obs = np.zeros_like(obs, dtype=np.float32)

    for i, aid in enumerate(dec.agent_id):
        if aid in term:  # episode ended
            rewards[i] = term[aid].reward
            dones[i] = 1.0
            # for terminal transitions, Unity still gives next obs in term
            next_obs[i] = term[aid].obs[0]
        else:
            rewards[i] = dec2[aid].reward
            next_obs[i] = dec2[aid].obs[0]

    for i in range(len(rewards)):
        replay.add(obs[i], a[i], rewards[i], next_obs[i], dones[i])

    return len(rewards)

# -----------------------------
# Training loop
# -----------------------------
def train():
    os.makedirs(cfg.out_dir, exist_ok=True)
    env, behavior, spec, obs_dim, act_dim, discrete_size = connect_unity()
    print(f"[Unity] behavior={behavior} obs_dim={obs_dim} act_dim={act_dim} discrete_size={discrete_size}")

    agent = SACAgent(obs_dim, act_dim, cfg)
    replay = Replay(obs_dim, act_dim, cfg.replay_size, agent.device)

    env_steps = 0
    last_save = 0
    stats = deque(maxlen=100)

    # Initial random exploration
    while env_steps < cfg.init_random_steps:
        dec, term = env.get_steps(behavior)
        
        # Skip if no agents need decisions
        if len(dec) == 0:
            # If we have terminal agents, step to get new decision agents
            if len(term) > 0:
                # Terminal agents don't need actions, just step
                env.step()
            else:
                # No agents at all, wait a bit
                time.sleep(0.01)
            continue
        
        obs = dec.obs[0]
        a = np.random.uniform(-1.0, 1.0, size=(len(dec), act_dim)).astype(np.float32)
        dec2, term = unity_step(env, behavior, a, spec=spec)

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

        for i in range(len(rewards)):
            replay.add(obs[i], a[i], rewards[i], next_obs[i], dones[i])
        env_steps += len(rewards)

    # Main loop
    while env_steps < cfg.total_env_steps:
        env_steps += collect_step(env, behavior, spec, agent, replay)

        if env_steps >= cfg.update_after and env_steps % cfg.update_every == 0:
            for _ in range(cfg.gradient_steps):
                metrics = agent.update(replay)
            stats.append(metrics["critic"])
            if len(stats) == stats.maxlen:
                print(f"Steps {env_steps:>8} | Qloss {sum(stats)/len(stats):.4f} | alpha {metrics['alpha']:.3f}")

        if env_steps - last_save >= cfg.save_every:
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "alpha": agent.log_alpha.detach().cpu(),
                "cfg": cfg.__dict__,
                "obs_dim": obs_dim,
                "act_dim": act_dim
            }, os.path.join(cfg.out_dir, f"sac_{env_steps}.pt"))
            last_save = env_steps
            print(f"Saved checkpoint at {env_steps} steps.")

    env.close()
    torch.save(agent.actor.state_dict(), os.path.join(cfg.out_dir, "sac_actor_final.pt"))
    print("Training complete.")

if __name__ == "__main__":
    train()
