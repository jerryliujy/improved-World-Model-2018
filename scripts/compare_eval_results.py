"""
Compare evaluation results across different model architectures.

Compares four architectures (vae-rnn, vqvae-rnn, vae-transformer, vqvae-transformer)
across three generations (80, 90, 100) and plots the averaged results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def parse_evaluation_report(file_path):
    """
    Parse evaluation report to extract episode rewards.
    
    Args:
        file_path: Path to evaluation_report.txt
        
    Returns:
        List of 10 rewards for each episode, or None if file not found
    """
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the EPISODE DETAILS section
        episode_section = content.split('EPISODE DETAILS')[-1]
        
        # Extract reward values using regex
        # Pattern: episode number, reward (float), steps
        pattern = r'^\s*\d+\s+([\d.-]+)\s+\d+\s*$'
        
        rewards = []
        for line in episode_section.split('\n'):
            match = re.match(pattern, line)
            if match:
                reward = float(match.group(1))
                rewards.append(reward)
        
        if len(rewards) != 10:
            print(f"  ⚠️  Expected 10 episodes, got {len(rewards)} in {file_path}")
            return None
        
        return rewards
    
    except Exception as e:
        print(f"  ❌ Error parsing {file_path}: {e}")
        return None


def load_all_results(base_path):
    """
    Load evaluation results for all architectures and generations.
    
    Args:
        base_path: Base directory containing vae-rnn, vq-rnn, etc.
        
    Returns:
        Dictionary with structure:
        {
            'vae-rnn': {'gen80': [...], 'gen90': [...], 'gen100': [...]},
            'vqvae-rnn': {...},
            'vae-transformer': {...},
            'vqvae-transformer': {...}
        }
    """
    architectures = ['vae-rnn', 'vqvae-rnn', 'vae-transformer', 'vqvae-transformer']
    generations = ['gen80', 'gen90', 'gen100']
    
    results = {}
    
    for arch in architectures:
        results[arch] = {}
        arch_path = os.path.join(base_path, arch)
        
        print(f"\n📂 Loading {arch}:")
        
        for gen in generations:
            gen_path = os.path.join(arch_path, gen, 'evaluation_report.txt')
            print(f"  Loading {gen}...", end=' ')
            
            rewards = parse_evaluation_report(gen_path)
            
            if rewards is not None:
                results[arch][gen] = rewards
                print(f"✓ ({len(rewards)} episodes)")
            else:
                results[arch][gen] = None
                print("❌")
    
    return results


def calculate_averaged_results(results):
    """
    Calculate averaged results across three generations.
    
    Args:
        results: Dictionary from load_all_results
        
    Returns:
        Dictionary with averaged results:
        {
            'vae-rnn': [avg_ep1, avg_ep2, ..., avg_ep10],
            ...
        }
    """
    averaged = {}
    
    for arch, gens in results.items():
        print(f"\n📊 Processing {arch}:")
        
        # Collect rewards for all generations
        gen80 = gens.get('gen80')
        gen90 = gens.get('gen90')
        gen100 = gens.get('gen100')
        
        # Check if all generations are available
        if gen80 is None or gen90 is None or gen100 is None:
            print(f"  ⚠️  Missing data for some generations")
            continue
        
        # Stack and average across generations
        stacked = np.array([gen80, gen90, gen100])  # [3, 10]
        averaged_rewards = np.mean(stacked, axis=0)  # [10]
        
        averaged[arch] = averaged_rewards
        
        print(f"  ✓ Averaged across gen80, gen90, gen100")
        print(f"    Mean reward: {np.mean(averaged_rewards):.2f}")
        print(f"    Std reward: {np.std(averaged_rewards):.2f}")
        print(f"    Min reward: {np.min(averaged_rewards):.2f}")
        print(f"    Max reward: {np.max(averaged_rewards):.2f}")
    
    return averaged


def plot_comparison(averaged_results, save_path='outputs/plots/eval_comparison.png'):
    """
    Plot comparison of four architectures.
    
    Args:
        averaged_results: Dictionary from calculate_averaged_results
        save_path: Where to save the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Define colors and markers for each architecture
    colors = {
        'vae-rnn': '#1f77b4',           # blue
        'vqvae-rnn': '#ff7f0e',            # orange
        'vae-transformer': '#2ca02c',   # green
        'vqvae-transformer': '#d62728'     # red
    }
    
    markers = {
        'vae-rnn': 'o',
        'vqvae-rnn': 's',
        'vae-transformer': '^',
        'vqvae-transformer': 'D'
    }
    
    # Plot each architecture
    for arch, rewards in averaged_results.items():
        episodes = np.arange(1, 11)
        plt.plot(
            episodes,
            rewards,
            color=colors[arch],
            marker=markers[arch],
            linewidth=2.5,
            markersize=8,
            label=arch,
            alpha=0.8
        )
    
    # Customize plot
    plt.xlabel('Episode Number', fontsize=13, fontweight='bold')
    plt.ylabel('Reward', fontsize=13, fontweight='bold')
    plt.title('Evaluation Comparison: Four Model Architectures\n(Averaged across Generations 80, 90, 100)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(range(1, 11), fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    
    plt.show()


def generate_summary_table(averaged_results):
    """
    Generate and print a summary table of results.
    
    Args:
        averaged_results: Dictionary from calculate_averaged_results
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY TABLE")
    print("="*80)
    print(f"{'Architecture':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*80)
    
    for arch, rewards in averaged_results.items():
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        print(f"{arch:<20} {mean_reward:<10.2f} {std_reward:<10.2f} {min_reward:<10.2f} {max_reward:<10.2f}")
    
    print("="*80 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("🚀 EVALUATION RESULTS COMPARISON")
    print("="*80)
    
    # Base path - adjust this to your actual directory structure
    base_path = 'outputs/eval'
    
    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        print("Please check your directory structure and update the base_path.")
        return
    
    # Load all results
    print("\n📂 Loading evaluation reports...")
    results = load_all_results(base_path)
    
    # Calculate averaged results
    print("\n📊 Calculating averaged results...")
    averaged_results = calculate_averaged_results(results)
    
    if not averaged_results:
        print("❌ No valid results found!")
        return
    
    # Generate summary table
    generate_summary_table(averaged_results)
    
    # Plot comparison
    print("\n📈 Generating comparison plot...")
    plot_comparison(averaged_results)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
