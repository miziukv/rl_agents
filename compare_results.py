"""
Compare training results between SAC and PPO implementations.
Loads TensorBoard logs and creates comparison visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalars(logdir, tag, default_value=None):
    """Load scalar values from TensorBoard log directory.
    
    Args:
        logdir: Path to TensorBoard log directory
        tag: Tag name to load (e.g., 'Environment/Cumulative Reward')
        default_value: Value to return if tag not found
    
    Returns:
        steps: numpy array of step values
        values: numpy array of scalar values
    """
    if not os.path.exists(logdir):
        print(f"Warning: Directory {logdir} does not exist")
        return None, None
    
    try:
        ea = EventAccumulator(logdir)
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            if default_value is not None:
                print(f"Warning: Tag '{tag}' not found in {logdir}, using default value")
                return np.array([0]), np.array([default_value])
            print(f"Warning: Tag '{tag}' not found in {logdir}")
            return None, None
        
        scalar_events = ea.Scalars(tag)
        steps = np.array([s.step for s in scalar_events])
        values = np.array([s.value for s in scalar_events])
        
        return steps, values
    except Exception as e:
        print(f"Error loading {logdir}: {e}")
        return None, None


def smooth_curve(values, window_size=100):
    """Apply moving average smoothing to a curve."""
    if len(values) < window_size:
        return values
    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    # Pad beginning to match original length
    padded = np.concatenate([values[:window_size-1], smoothed])
    return padded


def find_mlagents_logdir(base_dir):
    """Find ML-Agents TensorBoard log directory.
    ML-Agents stores logs in a behavior subdirectory (e.g., CourseAgent/).
    """
    if not os.path.exists(base_dir):
        return base_dir
    
    # Check if there's a behavior subdirectory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains TensorBoard event files
            try:
                for file in os.listdir(item_path):
                    if file.startswith('events.out.tfevents'):
                        return item_path
            except (OSError, PermissionError):
                continue
    
    # If no subdirectory found, return base directory
    return base_dir


def list_available_tags(logdir):
    """List all available TensorBoard tags in a log directory."""
    try:
        ea = EventAccumulator(logdir)
        ea.Reload()
        if 'scalars' in ea.Tags():
            return ea.Tags()['scalars']
        return []
    except Exception as e:
        print(f"Error listing tags in {logdir}: {e}")
        return []


def try_load_tag(logdir, tag_variants):
    """Try to load a tag using multiple possible names.
    
    Args:
        logdir: Path to TensorBoard log directory
        tag_variants: List of tag names to try
    
    Returns:
        steps, values: numpy arrays or (None, None) if not found
    """
    for tag in tag_variants:
        steps, values = load_tensorboard_scalars(logdir, tag)
        if values is not None:
            return steps, values
    return None, None


def compare_results(ppo_dir, sac_dir, output_file='comparison.png', smooth_window=100):
    """Compare PPO and SAC training results.
    
    Args:
        ppo_dir: Path to PPO TensorBoard log directory (ML-Agents format)
        sac_dir: Path to SAC TensorBoard log directory
        output_file: Output filename for comparison plot
        smooth_window: Window size for smoothing curves
    """
    
    # Find ML-Agents log directory (may be in behavior subdirectory)
    ppo_logdir = find_mlagents_logdir(ppo_dir)
    print(f"Loading PPO results from: {ppo_logdir}")
    print(f"Loading SAC results from: {sac_dir}")
    
    # Debug: List available tags
    print("\nAvailable PPO tags:")
    ppo_tags = list_available_tags(ppo_logdir)
    for tag in sorted(ppo_tags)[:15]:  # Show first 15
        print(f"  - {tag}")
    if len(ppo_tags) > 15:
        print(f"  ... and {len(ppo_tags) - 15} more")
    
    # Load episode rewards - ML-Agents uses 'Environment/Cumulative Reward'
    ppo_steps, ppo_rewards = try_load_tag(ppo_logdir, [
        'Environment/Cumulative Reward',
        'CourseAgent/Cumulative Reward',
        'Cumulative Reward'
    ])
    
    # ML-Agents doesn't have a separate "Mean Reward" tag, so we'll use cumulative reward
    # and calculate mean ourselves, or use cumulative reward for both
    ppo_mean_steps, ppo_mean_rewards = try_load_tag(ppo_logdir, [
        'Environment/Cumulative Reward',  # Use cumulative reward as mean (ML-Agents logs it per episode)
        'CourseAgent/Mean Reward',
        'CourseAgent/Mean Cumulative Reward',
        'Environment/Mean Cumulative Reward',
        'Mean Reward',
        'Mean Cumulative Reward'
    ])
    
    # If we loaded cumulative reward as mean, we can use it directly
    if ppo_mean_rewards is None and ppo_rewards is not None:
        ppo_mean_steps = ppo_steps
        ppo_mean_rewards = ppo_rewards
    
    sac_steps, sac_rewards = load_tensorboard_scalars(sac_dir, 'Environment/Cumulative Reward')
    sac_mean_steps, sac_mean_rewards = load_tensorboard_scalars(sac_dir, 'Environment/Mean Cumulative Reward')
    
    # Load episode lengths
    ppo_length_steps, ppo_lengths = try_load_tag(ppo_logdir, [
        'Environment/Episode Length',
        'CourseAgent/Episode Length',
        'Episode Length'
    ])
    sac_length_steps, sac_lengths = load_tensorboard_scalars(sac_dir, 'Environment/Episode Length')
    
    # Load policy metrics - ML-Agents uses 'Losses/Policy Loss' and 'Losses/Value Loss'
    ppo_policy_loss_steps, ppo_policy_loss = try_load_tag(ppo_logdir, [
        'Losses/Policy Loss',
        'CourseAgent/Policy Loss',
        'Policy/Policy Loss',
        'Policy Loss'
    ])
    sac_policy_loss_steps, sac_policy_loss = load_tensorboard_scalars(sac_dir, 'Policy/Actor Loss')
    
    ppo_value_loss_steps, ppo_value_loss = try_load_tag(ppo_logdir, [
        'Losses/Value Loss',
        'CourseAgent/Value Loss',
        'Policy/Value Loss',
        'Value Loss'
    ])
    sac_value_loss_steps, sac_value_loss = load_tensorboard_scalars(sac_dir, 'Policy/Critic Loss')
    
    ppo_entropy_steps, ppo_entropy = try_load_tag(ppo_logdir, [
        'Policy/Entropy',
        'CourseAgent/Entropy',
        'Entropy'
    ])
    sac_entropy_steps, sac_entropy = load_tensorboard_scalars(sac_dir, 'Policy/Entropy')
    
    # Create comparison plots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Episode Rewards (Raw)
    ax1 = plt.subplot(3, 3, 1)
    if ppo_rewards is not None:
        plt.plot(ppo_steps, ppo_rewards, alpha=0.2, label='PPO (raw)', color='blue', linewidth=0.5)
    if sac_rewards is not None:
        plt.plot(sac_steps, sac_rewards, alpha=0.2, label='SAC (raw)', color='red', linewidth=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards (Raw)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Mean Episode Rewards
    ax2 = plt.subplot(3, 3, 2)
    if ppo_mean_rewards is not None:
        ppo_smooth = smooth_curve(ppo_mean_rewards, smooth_window)
        plt.plot(ppo_mean_steps, ppo_smooth, label='PPO', color='blue', linewidth=2)
    if sac_mean_rewards is not None:
        sac_smooth = smooth_curve(sac_mean_rewards, smooth_window)
        plt.plot(sac_mean_steps, sac_smooth, label='SAC', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (smoothed)')
    plt.title('Mean Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Final Performance Comparison
    ax3 = plt.subplot(3, 3, 3)
    final_ppo = None
    final_sac = None
    if ppo_mean_rewards is not None and len(ppo_mean_rewards) >= 100:
        final_ppo = np.mean(ppo_mean_rewards[-100:])
    elif ppo_mean_rewards is not None:
        final_ppo = np.mean(ppo_mean_rewards)
    
    if sac_mean_rewards is not None and len(sac_mean_rewards) >= 100:
        final_sac = np.mean(sac_mean_rewards[-100:])
    elif sac_mean_rewards is not None:
        final_sac = np.mean(sac_mean_rewards)
    
    if final_ppo is not None or final_sac is not None:
        bars = []
        labels = []
        colors = []
        if final_ppo is not None:
            bars.append(final_ppo)
            labels.append('PPO')
            colors.append('blue')
        if final_sac is not None:
            bars.append(final_sac)
            labels.append('SAC')
            colors.append('red')
        
        plt.bar(labels, bars, color=colors, alpha=0.7)
        plt.ylabel('Mean Reward (last 100 episodes)')
        plt.title('Final Performance Comparison')
        plt.grid(True, axis='y', alpha=0.3)
    
    # 4. Episode Lengths
    ax4 = plt.subplot(3, 3, 4)
    if ppo_lengths is not None:
        ppo_length_smooth = smooth_curve(ppo_lengths, smooth_window)
        plt.plot(ppo_length_steps, ppo_length_smooth, label='PPO', color='blue', linewidth=2)
    if sac_lengths is not None:
        sac_length_smooth = smooth_curve(sac_lengths, smooth_window)
        plt.plot(sac_length_steps, sac_length_smooth, label='SAC', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Policy Loss
    ax5 = plt.subplot(3, 3, 5)
    if ppo_policy_loss is not None:
        plt.plot(ppo_policy_loss_steps, ppo_policy_loss, label='PPO', color='blue', alpha=0.7)
    if sac_policy_loss is not None:
        plt.plot(sac_policy_loss_steps, sac_policy_loss, label='SAC', color='red', alpha=0.7)
    plt.xlabel('Environment Steps')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 6. Value Loss
    ax6 = plt.subplot(3, 3, 6)
    if ppo_value_loss is not None:
        plt.plot(ppo_value_loss_steps, ppo_value_loss, label='PPO', color='blue', alpha=0.7)
    if sac_value_loss is not None:
        plt.plot(sac_value_loss_steps, sac_value_loss, label='SAC', color='red', alpha=0.7)
    plt.xlabel('Environment Steps')
    plt.ylabel('Value Loss')
    plt.title('Value Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 7. Entropy
    ax7 = plt.subplot(3, 3, 7)
    if ppo_entropy is not None:
        plt.plot(ppo_entropy_steps, ppo_entropy, label='PPO', color='blue', alpha=0.7)
    if sac_entropy is not None:
        plt.plot(sac_entropy_steps, sac_entropy, label='SAC', color='red', alpha=0.7)
    plt.xlabel('Environment Steps')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Sample Efficiency (Reward vs Steps)
    ax8 = plt.subplot(3, 3, 8)
    if ppo_mean_rewards is not None and ppo_mean_steps is not None:
        # Convert episode count to approximate steps (if needed)
        # For now, use episode as x-axis
        ppo_smooth = smooth_curve(ppo_mean_rewards, smooth_window)
        plt.plot(ppo_mean_steps, ppo_smooth, label='PPO', color='blue', linewidth=2)
    if sac_mean_rewards is not None and sac_mean_steps is not None:
        sac_smooth = smooth_curve(sac_mean_rewards, smooth_window)
        plt.plot(sac_mean_steps, sac_smooth, label='SAC', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = "Summary Statistics\n" + "="*30 + "\n\n"
    
    if ppo_mean_rewards is not None:
        stats_text += "PPO:\n"
        stats_text += f"  Episodes: {len(ppo_mean_rewards)}\n"
        if len(ppo_mean_rewards) > 0:
            stats_text += f"  Final Mean: {ppo_mean_rewards[-1]:.2f}\n"
            if len(ppo_mean_rewards) >= 100:
                stats_text += f"  Last 100 Mean: {np.mean(ppo_mean_rewards[-100:]):.2f}\n"
            stats_text += f"  Max: {np.max(ppo_mean_rewards):.2f}\n"
        stats_text += "\n"
    
    if sac_mean_rewards is not None:
        stats_text += "SAC:\n"
        stats_text += f"  Episodes: {len(sac_mean_rewards)}\n"
        if len(sac_mean_rewards) > 0:
            stats_text += f"  Final Mean: {sac_mean_rewards[-1]:.2f}\n"
            if len(sac_mean_rewards) >= 100:
                stats_text += f"  Last 100 Mean: {np.mean(sac_mean_rewards[-100:]):.2f}\n"
            stats_text += f"  Max: {np.max(sac_mean_rewards):.2f}\n"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if ppo_mean_rewards is not None:
        print("\nPPO:")
        print(f"  Total Episodes: {len(ppo_mean_rewards)}")
        if len(ppo_mean_rewards) > 0:
            print(f"  Final Mean Reward: {ppo_mean_rewards[-1]:.2f}")
            if len(ppo_mean_rewards) >= 100:
                print(f"  Last 100 Episodes Mean: {np.mean(ppo_mean_rewards[-100:]):.2f} ± {np.std(ppo_mean_rewards[-100:]):.2f}")
            print(f"  Maximum Mean Reward: {np.max(ppo_mean_rewards):.2f}")
    
    if sac_mean_rewards is not None:
        print("\nSAC:")
        print(f"  Total Episodes: {len(sac_mean_rewards)}")
        if len(sac_mean_rewards) > 0:
            print(f"  Final Mean Reward: {sac_mean_rewards[-1]:.2f}")
            if len(sac_mean_rewards) >= 100:
                print(f"  Last 100 Episodes Mean: {np.mean(sac_mean_rewards[-100:]):.2f} ± {np.std(sac_mean_rewards[-100:]):.2f}")
            print(f"  Maximum Mean Reward: {np.max(sac_mean_rewards):.2f}")
    
    if ppo_mean_rewards is not None and sac_mean_rewards is not None:
        print("\nComparison:")
        ppo_final = np.mean(ppo_mean_rewards[-100:]) if len(ppo_mean_rewards) >= 100 else np.mean(ppo_mean_rewards)
        sac_final = np.mean(sac_mean_rewards[-100:]) if len(sac_mean_rewards) >= 100 else np.mean(sac_mean_rewards)
        diff = sac_final - ppo_final
        pct_diff = (diff / abs(ppo_final)) * 100 if ppo_final != 0 else 0
        print(f"  SAC vs PPO: {diff:+.2f} ({pct_diff:+.1f}%)")
    
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare SAC and PPO training results')
    parser.add_argument('--ppo-dir', type=str, 
                       default='results/ppo_course_v1',
                       help='Path to PPO TensorBoard log directory')
    parser.add_argument('--sac-dir', type=str,
                       default='results/sac_course',
                       help='Path to SAC TensorBoard log directory')
    parser.add_argument('--output', type=str,
                       default='comparison.png',
                       help='Output filename for comparison plot')
    parser.add_argument('--smooth', type=int,
                       default=100,
                       help='Smoothing window size for curves')
    
    args = parser.parse_args()
    
    compare_results(args.ppo_dir, args.sac_dir, args.output, args.smooth)

