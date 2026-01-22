#!/usr/bin/env python3
"""
Google Dreamer V1 Training Log Processor
Generates visualizations for training metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
INPUT_FILE = 'training_log.csv'
OUTPUT_DIR = './outputs'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} training records")
print(f"Steps range: {df['step'].min()} - {df['step'].max()}")
print(f"Episodes: {df['episode'].max()}")

# Calculate rolling averages for smoother curves
window_size = 5
df['reward_smooth'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
df['model_loss_smooth'] = df['model_loss'].rolling(window=window_size, min_periods=1).mean()
df['actor_loss_smooth'] = df['actor_loss'].rolling(window=window_size, min_periods=1).mean()
df['critic_loss_smooth'] = df['critic_loss'].rolling(window=window_size, min_periods=1).mean()

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Color palette
COLORS = {
    'reward': '#2ecc71',      # Green
    'model': '#3498db',       # Blue
    'actor': '#e74c3c',       # Red
    'critic': '#9b59b6',      # Purple
    'smooth': '#2c3e50'       # Dark gray for smoothed lines
}


# =============================================================================
# Plot 1: Episode Reward over Training
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df['step'], df['reward'], color=COLORS['reward'], alpha=0.3, linewidth=1, label='Raw Reward')
ax.plot(df['step'], df['reward_smooth'], color=COLORS['reward'], linewidth=2, label=f'Smoothed (window={window_size})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Training Steps')
ax.set_ylabel('Episode Reward')
ax.set_title('Google Dreamer V1 - Episode Reward over Training')
ax.legend(loc='lower right')

# Add annotation for best episode
best_idx = df['reward'].idxmax()
best_step = df.loc[best_idx, 'step']
best_reward = df.loc[best_idx, 'reward']
ax.annotate(f'Best: {best_reward:.1f}', xy=(best_step, best_reward), 
            xytext=(best_step + 1000, best_reward + 10),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
            fontsize=9, color=COLORS['smooth'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reward_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reward_curve.png")


# =============================================================================
# Plot 2: All Losses Combined
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Model Loss
axes[0].plot(df['step'], df['model_loss'], color=COLORS['model'], alpha=0.3, linewidth=1)
axes[0].plot(df['step'], df['model_loss_smooth'], color=COLORS['model'], linewidth=2)
axes[0].set_ylabel('Model Loss')
axes[0].set_title('World Model Loss (Reconstruction + Reward + β×KL)')

# Actor Loss
axes[1].plot(df['step'], df['actor_loss'], color=COLORS['actor'], alpha=0.3, linewidth=1)
axes[1].plot(df['step'], df['actor_loss_smooth'], color=COLORS['actor'], linewidth=2)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[1].set_ylabel('Actor Loss')
axes[1].set_title('Actor Loss (-Vλ mean)')

# Critic Loss
axes[2].plot(df['step'], df['critic_loss'], color=COLORS['critic'], alpha=0.3, linewidth=1)
axes[2].plot(df['step'], df['critic_loss_smooth'], color=COLORS['critic'], linewidth=2)
axes[2].set_ylabel('Critic Loss')
axes[2].set_title('Critic Loss (MSE)')
axes[2].set_xlabel('Training Steps')

plt.suptitle('Google Dreamer V1 - Training Losses', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_losses.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_losses.png")


# =============================================================================
# Plot 3: Combined Overview (2x2 grid)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Reward
axes[0, 0].plot(df['step'], df['reward'], color=COLORS['reward'], alpha=0.3, linewidth=1)
axes[0, 0].plot(df['step'], df['reward_smooth'], color=COLORS['reward'], linewidth=2)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Episode Reward')

# Model Loss
axes[0, 1].plot(df['step'], df['model_loss'], color=COLORS['model'], alpha=0.3, linewidth=1)
axes[0, 1].plot(df['step'], df['model_loss_smooth'], color=COLORS['model'], linewidth=2)
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('World Model Loss')

# Actor Loss
axes[1, 0].plot(df['step'], df['actor_loss'], color=COLORS['actor'], alpha=0.3, linewidth=1)
axes[1, 0].plot(df['step'], df['actor_loss_smooth'], color=COLORS['actor'], linewidth=2)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Training Steps')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Actor Loss')

# Critic Loss
axes[1, 1].plot(df['step'], df['critic_loss'], color=COLORS['critic'], alpha=0.3, linewidth=1)
axes[1, 1].plot(df['step'], df['critic_loss_smooth'], color=COLORS['critic'], linewidth=2)
axes[1, 1].set_xlabel('Training Steps')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Critic Loss')

plt.suptitle('Google Dreamer V1 - Training Overview (CarRacing-V3)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_overview.png")


# =============================================================================
# Print Summary Statistics
# =============================================================================
print("\n" + "="*60)
print("TRAINING SUMMARY STATISTICS")
print("="*60)

print(f"\nReward:")
print(f"  Min:  {df['reward'].min():.2f}")
print(f"  Max:  {df['reward'].max():.2f}")
print(f"  Mean: {df['reward'].mean():.2f}")
print(f"  Last 10 avg: {df['reward'].tail(10).mean():.2f}")

print(f"\nModel Loss:")
print(f"  Initial: {df['model_loss'].iloc[0]:.4f}")
print(f"  Final:   {df['model_loss'].iloc[-1]:.4f}")
print(f"  Reduction: {((df['model_loss'].iloc[0] - df['model_loss'].iloc[-1]) / df['model_loss'].iloc[0] * 100):.1f}%")

print(f"\nActor Loss:")
print(f"  Initial: {df['actor_loss'].iloc[0]:.4f}")
print(f"  Final:   {df['actor_loss'].iloc[-1]:.4f}")

print(f"\nCritic Loss:")
print(f"  Initial: {df['critic_loss'].iloc[0]:.4f}")
print(f"  Final:   {df['critic_loss'].iloc[-1]:.4f}")

print("\n" + "="*60)
print(f"Generated plots saved to: {OUTPUT_DIR}")
print("="*60)
