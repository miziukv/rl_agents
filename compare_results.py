"""Compare training results between SAC and PPO implementations."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(logdir, tag, default=None):
    """Load scalar values from TensorBoard log directory."""
    if not os.path.exists(logdir):
        return None, None
    
    try:
        ea = EventAccumulator(logdir)
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            if default is not None:
                return np.array([0]), np.array([default])
            return None, None
        
        events = ea.Scalars(tag)
        return np.array([s.step for s in events]), np.array([s.value for s in events])
    except Exception as e:
        print(f"Error loading {logdir}: {e}")
        return None, None


def smooth(values, window=100):
    """Apply moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    return np.concatenate([values[:window-1], smoothed])


def find_mlagents_logdir(base_dir):
    """Find ML-Agents TensorBoard log directory (may be in behavior subdirectory)."""
    if not os.path.exists(base_dir):
        return base_dir
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            try:
                if any(f.startswith('events.out.tfevents') for f in os.listdir(item_path)):
                    return item_path
            except (OSError, PermissionError):
                continue
    
    return base_dir


def try_load(logdir, tags):
    """Try loading a tag using multiple possible names."""
    for tag in tags:
        steps, values = load_scalar(logdir, tag)
        if values is not None:
            return steps, values
    return None, None


def plot_metric(ax, ppo_data, sac_data, xlabel, ylabel, title, log_scale=False, smooth_window=100):
    """Helper to plot a metric comparison."""
    ppo_steps, ppo_values = ppo_data
    sac_steps, sac_values = sac_data
    
    if ppo_values is not None:
        if smooth_window > 1:
            ppo_values = smooth(ppo_values, smooth_window)
        ax.plot(ppo_steps, ppo_values, label='PPO', color='blue', alpha=0.7 if log_scale else 1, linewidth=2)
    
    if sac_values is not None:
        if smooth_window > 1:
            sac_values = smooth(sac_values, smooth_window)
        ax.plot(sac_steps, sac_values, label='SAC', color='red', alpha=0.7 if log_scale else 1, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')


def compare_results(ppo_dir, sac_dir, output_file='comparison.png', smooth_window=100):
    """Compare PPO and SAC training results."""
    ppo_logdir = find_mlagents_logdir(ppo_dir)
    print(f"Loading PPO from: {ppo_logdir}")
    print(f"Loading SAC from: {sac_dir}")
    
    # Load rewards
    ppo_steps, ppo_rewards = try_load(ppo_logdir, [
        'Environment/Cumulative Reward',
        'CourseAgent/Cumulative Reward',
        'Cumulative Reward'
    ])
    
    ppo_mean_steps, ppo_mean_rewards = try_load(ppo_logdir, [
        'Environment/Cumulative Reward',
        'CourseAgent/Mean Reward',
        'CourseAgent/Mean Cumulative Reward',
        'Environment/Mean Cumulative Reward',
        'Mean Reward',
        'Mean Cumulative Reward'
    ])
    
    if ppo_mean_rewards is None and ppo_rewards is not None:
        ppo_mean_steps, ppo_mean_rewards = ppo_steps, ppo_rewards
    
    sac_steps, sac_rewards = load_scalar(sac_dir, 'Environment/Cumulative Reward')
    sac_mean_steps, sac_mean_rewards = load_scalar(sac_dir, 'Environment/Mean Cumulative Reward')
    
    # Load other metrics
    ppo_length_steps, ppo_lengths = try_load(ppo_logdir, [
        'Environment/Episode Length',
        'CourseAgent/Episode Length',
        'Episode Length'
    ])
    sac_length_steps, sac_lengths = load_scalar(sac_dir, 'Environment/Episode Length')
    
    ppo_policy_steps, ppo_policy_loss = try_load(ppo_logdir, [
        'Losses/Policy Loss',
        'CourseAgent/Policy Loss',
        'Policy/Policy Loss',
        'Policy Loss'
    ])
    sac_policy_steps, sac_policy_loss = load_scalar(sac_dir, 'Policy/Actor Loss')
    
    ppo_value_steps, ppo_value_loss = try_load(ppo_logdir, [
        'Losses/Value Loss',
        'CourseAgent/Value Loss',
        'Policy/Value Loss',
        'Value Loss'
    ])
    sac_value_steps, sac_value_loss = load_scalar(sac_dir, 'Policy/Critic Loss')
    
    ppo_entropy_steps, ppo_entropy = try_load(ppo_logdir, [
        'Policy/Entropy',
        'CourseAgent/Entropy',
        'Entropy'
    ])
    sac_entropy_steps, sac_entropy = load_scalar(sac_dir, 'Policy/Entropy')
    
    # Create plots
    plt.figure(figsize=(16, 10))
    
    # Raw rewards
    ax = plt.subplot(3, 3, 1)
    if ppo_rewards is not None:
        ax.plot(ppo_steps, ppo_rewards, alpha=0.2, label='PPO', color='blue', linewidth=0.5)
    if sac_rewards is not None:
        ax.plot(sac_steps, sac_rewards, alpha=0.2, label='SAC', color='red', linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Raw Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean rewards
    plot_metric(plt.subplot(3, 3, 2), 
                (ppo_mean_steps, ppo_mean_rewards),
                (sac_mean_steps, sac_mean_rewards),
                'Episode', 'Mean Reward', 'Mean Episode Rewards', smooth_window=smooth_window)
    
    # Final performance
    ax = plt.subplot(3, 3, 3)
    bars, labels, colors = [], [], []
    if ppo_mean_rewards is not None:
        final_ppo = np.mean(ppo_mean_rewards[-100:]) if len(ppo_mean_rewards) >= 100 else np.mean(ppo_mean_rewards)
        bars.append(final_ppo)
        labels.append('PPO')
        colors.append('blue')
    if sac_mean_rewards is not None:
        final_sac = np.mean(sac_mean_rewards[-100:]) if len(sac_mean_rewards) >= 100 else np.mean(sac_mean_rewards)
        bars.append(final_sac)
        labels.append('SAC')
        colors.append('red')
    
    if bars:
        ax.bar(labels, bars, color=colors, alpha=0.7)
        ax.set_ylabel('Mean Reward (last 100)')
        ax.set_title('Final Performance')
        ax.grid(True, axis='y', alpha=0.3)
    
    # Episode lengths
    plot_metric(plt.subplot(3, 3, 4),
                (ppo_length_steps, ppo_lengths),
                (sac_length_steps, sac_lengths),
                'Episode', 'Length', 'Episode Lengths', smooth_window=smooth_window)
    
    # Policy loss
    plot_metric(plt.subplot(3, 3, 5),
                (ppo_policy_steps, ppo_policy_loss),
                (sac_policy_steps, sac_policy_loss),
                'Steps', 'Policy Loss', 'Policy Loss', log_scale=True)
    
    # Value loss
    plot_metric(plt.subplot(3, 3, 6),
                (ppo_value_steps, ppo_value_loss),
                (sac_value_steps, sac_value_loss),
                'Steps', 'Value Loss', 'Value Loss', log_scale=True)
    
    # Entropy
    plot_metric(plt.subplot(3, 3, 7),
                (ppo_entropy_steps, ppo_entropy),
                (sac_entropy_steps, sac_entropy),
                'Steps', 'Entropy', 'Policy Entropy')
    
    # Learning curves (duplicate of mean rewards but kept for consistency)
    plot_metric(plt.subplot(3, 3, 8),
                (ppo_mean_steps, ppo_mean_rewards),
                (sac_mean_steps, sac_mean_rewards),
                'Episode', 'Mean Reward', 'Learning Curves', smooth_window=smooth_window)
    
    # Stats table
    ax = plt.subplot(3, 3, 9)
    ax.axis('off')
    
    stats = ["Summary Statistics", "="*30, ""]
    
    if ppo_mean_rewards is not None:
        stats.extend(["PPO:", f"  Episodes: {len(ppo_mean_rewards)}"])
        if len(ppo_mean_rewards) > 0:
            stats.append(f"  Final: {ppo_mean_rewards[-1]:.2f}")
            if len(ppo_mean_rewards) >= 100:
                stats.append(f"  Last 100: {np.mean(ppo_mean_rewards[-100:]):.2f}")
            stats.append(f"  Max: {np.max(ppo_mean_rewards):.2f}")
        stats.append("")
    
    if sac_mean_rewards is not None:
        stats.extend(["SAC:", f"  Episodes: {len(sac_mean_rewards)}"])
        if len(sac_mean_rewards) > 0:
            stats.append(f"  Final: {sac_mean_rewards[-1]:.2f}")
            if len(sac_mean_rewards) >= 100:
                stats.append(f"  Last 100: {np.mean(sac_mean_rewards[-100:]):.2f}")
            stats.append(f"  Max: {np.max(sac_mean_rewards):.2f}")
    
    ax.text(0.1, 0.9, '\n'.join(stats), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if ppo_mean_rewards is not None:
        print("\nPPO:")
        print(f"  Episodes: {len(ppo_mean_rewards)}")
        if len(ppo_mean_rewards) > 0:
            print(f"  Final: {ppo_mean_rewards[-1]:.2f}")
            if len(ppo_mean_rewards) >= 100:
                mean = np.mean(ppo_mean_rewards[-100:])
                std = np.std(ppo_mean_rewards[-100:])
                print(f"  Last 100: {mean:.2f} ± {std:.2f}")
            print(f"  Max: {np.max(ppo_mean_rewards):.2f}")
    
    if sac_mean_rewards is not None:
        print("\nSAC:")
        print(f"  Episodes: {len(sac_mean_rewards)}")
        if len(sac_mean_rewards) > 0:
            print(f"  Final: {sac_mean_rewards[-1]:.2f}")
            if len(sac_mean_rewards) >= 100:
                mean = np.mean(sac_mean_rewards[-100:])
                std = np.std(sac_mean_rewards[-100:])
                print(f"  Last 100: {mean:.2f} ± {std:.2f}")
            print(f"  Max: {np.max(sac_mean_rewards):.2f}")
    
    if ppo_mean_rewards is not None and sac_mean_rewards is not None:
        ppo_final = np.mean(ppo_mean_rewards[-100:]) if len(ppo_mean_rewards) >= 100 else np.mean(ppo_mean_rewards)
        sac_final = np.mean(sac_mean_rewards[-100:]) if len(sac_mean_rewards) >= 100 else np.mean(sac_mean_rewards)
        diff = sac_final - ppo_final
        pct = (diff / abs(ppo_final)) * 100 if ppo_final != 0 else 0
        print(f"\nComparison: SAC vs PPO = {diff:+.2f} ({pct:+.1f}%)")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare SAC and PPO training results')
    parser.add_argument('--ppo-dir', type=str, default='results/ppo_course_v1',
                       help='PPO TensorBoard log directory')
    parser.add_argument('--sac-dir', type=str, default='results/sac_course',
                       help='SAC TensorBoard log directory')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output filename')
    parser.add_argument('--smooth', type=int, default=100,
                       help='Smoothing window size')
    
    args = parser.parse_args()
    compare_results(args.ppo_dir, args.sac_dir, args.output, args.smooth)
