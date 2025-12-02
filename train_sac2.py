"""
Soft Actor-Critic (SAC) implementation from scratch for Unity ML-Agents.
Compares against ML-Agents' built-in PPO trainer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import argparse
import os
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network that outputs mean and log_std for a Gaussian policy."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass through actor network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp for numerical stability
        
        return mean, log_std
    
    def sample(self, state, epsilon=1e-6):
        """Sample action from policy distribution."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x_t) * self.max_action
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Tanh correction for log probability
        log_prob -= torch.log(self.max_action * (1 - action.pow(2) / (self.max_action ** 2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean
    
    def deterministic_action(self, state):
        """Get deterministic action (mean of policy)."""
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action


class Critic(nn.Module):
    """Critic network (Q-function)."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """Forward pass through critic network."""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


class SAC:
    """Soft Actor-Critic algorithm."""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.995,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        hidden_dim=256,
        device='cpu'
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Target networks
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Entropy coefficient
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
    
    def select_action(self, state, evaluate=False):
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            action = self.actor.deterministic_action(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, batch_size=256):
        """Update networks using a batch from replay buffer."""
        if len(replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        self._soft_update(self.critic1_target, self.critic1, self.tau)
        self._soft_update(self.critic2_target, self.critic2, self.tau)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        }
    
    def _soft_update(self, target, source, tau):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save(self, filepath):
        """Save model checkpoints."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])


def train_sac(
    run_id='sac_course',
    max_steps=3000000,
    batch_size=256,
    buffer_size=1000000,
    learning_starts=10000,
    train_freq=1,
    update_after=1000,
    gamma=0.995,
    tau=0.005,
    alpha=0.2,
    lr=3e-4,
    hidden_dim=256,
    summary_freq=5000,
    checkpoint_freq=200000,
    resume=False
):
    """Main training loop for SAC."""
    
    # Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        print("Note: CUDA not available. Install PyTorch with CUDA support for GPU training.")
        print("  CPU-only PyTorch detected. For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Create results directory
    results_dir = f"results/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=results_dir)
    
    # Unity environment
    engine_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=None, side_channels=[engine_channel], seed=42)
    env.reset()
    
    # Get environment info
    behavior_name = list(env.behavior_specs.keys())[0]
    behavior_spec = env.behavior_specs[behavior_name]
    
    # Get observation and action dimensions
    # Vector observations
    vector_obs_size = behavior_spec.observation_specs[0].shape[0]
    action_spec = behavior_spec.action_spec
    action_dim = action_spec.continuous_size
    discrete_branches = action_spec.discrete_branches  # List of branch sizes
    
    print(f"Observation size: {vector_obs_size}")
    print(f"Continuous action size: {action_dim}")
    print(f"Discrete action branches: {discrete_branches}")
    print(f"Action spec: {action_spec}")
    
    # Initialize SAC agent
    agent = SAC(
        state_dim=vector_obs_size,
        action_dim=action_dim,
        lr_actor=lr,
        lr_critic=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        hidden_dim=hidden_dim,
        device=device
    )
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    step_count = 0
    episode_count = 0
    
    # Resume from checkpoint if needed
    if resume:
        checkpoint_path = os.path.join(results_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"Resumed from checkpoint at {checkpoint_path}")
            # Load step count from a separate file or infer from checkpoint name
    
    # Initial state
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) > 0:
        state = decision_steps.obs[0][0]
    else:
        state = terminal_steps.obs[0][0]
    
    print("Starting training...")
    
    try:
        while step_count < max_steps:
            # Select action
            if step_count < learning_starts:
                # Random action for exploration
                rng = np.random.default_rng()
                action = rng.uniform(-1, 1, size=action_dim)
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Step environment - ML-Agents requires ActionTuple format
            # Always provide discrete actions if branches exist (even if not used in C# code)
            continuous_actions = action.reshape(1, -1)
            if len(discrete_branches) > 0:
                # Provide zero discrete actions for each branch (not used by CourseAgent, but required by spec)
                # Shape should be (num_agents, num_branches)
                discrete_actions = np.zeros((1, len(discrete_branches)), dtype=np.int32)
                action_tuple = ActionTuple(continuous=continuous_actions, discrete=discrete_actions)
            else:
                # No discrete actions in spec
                action_tuple = ActionTuple(continuous=continuous_actions)
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            # Get next state and reward
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(decision_steps) > 0:
                next_state = decision_steps.obs[0][0]
                reward = decision_steps.reward[0]
                done = False
            else:
                next_state = terminal_steps.obs[0][0]
                reward = terminal_steps.reward[0]
                done = True
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Update stats
            current_episode_reward += reward
            current_episode_length += 1
            step_count += 1
            
            # Update agent
            if step_count >= update_after and step_count % train_freq == 0:
                update_info = agent.update(replay_buffer, batch_size)
                
                if update_info and step_count % summary_freq == 0:
                    # Log training metrics
                    for key, value in update_info.items():
                        writer.add_scalar(f'train/{key}', value, step_count)
            
            # Handle episode end
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_count += 1
                
                # Log episode stats
                if episode_count % 10 == 0:
                    mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    mean_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                    
                    print(f"Step: {step_count}, Episode: {episode_count}, "
                          f"Mean Reward: {mean_reward:.3f}, Mean Length: {mean_length:.1f}")
                    
                    writer.add_scalar('episode/reward', current_episode_reward, episode_count)
                    writer.add_scalar('episode/length', current_episode_length, episode_count)
                    writer.add_scalar('episode/mean_reward', mean_reward, episode_count)
                
                # Reset episode stats
                current_episode_reward = 0
                current_episode_length = 0
                
                # Reset environment
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if len(decision_steps) > 0:
                    state = decision_steps.obs[0][0]
                else:
                    state = terminal_steps.obs[0][0]
            else:
                state = next_state
            
            # Save checkpoint
            if step_count % checkpoint_freq == 0:
                checkpoint_path = os.path.join(results_dir, 'checkpoint.pt')
                agent.save(checkpoint_path)
                print(f"Saved checkpoint at step {step_count}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Final save
        agent.save(os.path.join(results_dir, 'final_model.pt'))
        env.close()
        writer.close()
        print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAC agent on Unity ML-Agents environment')
    parser.add_argument('--run-id', type=str, default='sac_course', help='Run identifier')
    parser.add_argument('--max-steps', type=int, default=3000000, help='Maximum training steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--learning-starts', type=int, default=10000, help='Steps before learning starts')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy coefficient')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_sac(
        run_id=args.run_id,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        resume=args.resume
    )

