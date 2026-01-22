"""
Dreamer Training Script for CarRacing-v3

Based on Hafner et al. (2020) "Dream to Control: Learning Behaviors by Latent Imagination"

Training consists of three interleaved processes:
1. Environment Interaction: Collect experience using current policy
2. Dynamics Learning: Train world model on collected experience  
3. Behavior Learning: Train actor-critic by imagining in latent space

Features:
- Automatic checkpointing with crash recovery
- Signal handling for graceful termination
- Resume capability from interrupted training
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import cv2
import pandas as pd
import os
import signal
import sys
import atexit
from datetime import datetime
from replay import ReplayBuffer
from dreamer import Dreamer


# ==============================================================================
# HYPERPARAMETERS (from Appendix A of the paper)
# ==============================================================================

# Environment
ENV_NAME = "CarRacing-v3"
ACTION_REPEAT = 2  # Execute same action multiple times (important for CarRacing)

# Model Architecture
STOCH_DIM = 30          # z dimension (stochastic latent)
DET_DIM = 200           # h dimension (RNN hidden state)
HIDDEN_DIM = 300        # Hidden layer size for actor/critic/reward
EMBED_DIM = 1024        # Image embedding dimension

# Training
TOTAL_ENV_STEPS = 1000000   # Total environment steps (paper uses 5M for control suite)
SEED_EPISODES = 5           # Random episodes to seed replay buffer
COLLECT_INTERVAL = 100      # Collect this many steps, then train
TRAIN_STEPS = 100           # Gradient steps per collection phase
BATCH_SIZE = 50             # Batch size for training
SEQ_LEN = 50                # Sequence length for training

# Imagination
HORIZON = 15                # Dream horizon H
GAMMA = 0.99                # Discount factor
LAMBDA = 0.95               # GAE lambda for V_λ targets

# Losses
FREE_NATS = 3.0             # KL divergence free threshold
KL_SCALE = 1.0              # KL loss coefficient (β in the paper)
GRAD_CLIP = 100             # Gradient norm clipping

# Learning rates (from paper) -> hardcoded in dreamer.py optimizers
WORLD_MODEL_LR = 6e-4
ACTOR_LR = 8e-5
CRITIC_LR = 8e-5

# Logging and Checkpointing
LOG_INTERVAL = 10           # Log every N episodes
SAVE_INTERVAL = 50000       # Save checkpoint every N steps
AUTOSAVE_INTERVAL = 5000    # Autosave every N steps (for crash recovery)

# Resume settings - set to checkpoint path to resume, or None to start fresh
RESUME_PATH = None  # e.g., "dreamer_autosave.pth" or "dreamer_step50000.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# GLOBAL STATE FOR SIGNAL HANDLERS
# ==============================================================================

# These will be set in main() and used by signal handlers
_global_agent = None
_global_metrics_log = None
_global_step = 0
_global_episode = 0
_save_dir = "checkpoints"


# ==============================================================================
# CHECKPOINT AND RECOVERY FUNCTIONS
# ==============================================================================

def ensure_save_dir():
    """Create checkpoint directory if it doesn't exist."""
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)
        print(f"Created checkpoint directory: {_save_dir}")


def save_checkpoint(agent, metrics_log, global_step, episode, filename_prefix="dreamer", 
                    reason="checkpoint"):
    """
    Save model checkpoint and training log.
    
    Args:
        agent: The Dreamer agent
        metrics_log: Dictionary containing training metrics
        global_step: Current training step
        episode: Current episode number
        filename_prefix: Prefix for saved files
        reason: Why we're saving (for logging)
    """
    ensure_save_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model state
    model_path = os.path.join(_save_dir, f"{filename_prefix}.pth")
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'global_step': global_step,
        'episode': episode,
        'timestamp': timestamp,
        'reason': reason
    }
    torch.save(checkpoint, model_path)
    
    # Save training log
    log_path = os.path.join(_save_dir, f"{filename_prefix}_log.csv")
    df = pd.DataFrame(metrics_log)
    df.to_csv(log_path, index=False)
    
    print(f"[{reason.upper()}] Saved checkpoint at step {global_step}, episode {episode}")
    print(f"  Model: {model_path}")
    print(f"  Log: {log_path}")
    
    return model_path, log_path


def load_checkpoint(agent, checkpoint_path):
    """
    Load model checkpoint and return training state.
    
    Args:
        agent: The Dreamer agent to load weights into
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        global_step, episode, metrics_log (or defaults if not found)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, 0, None
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle both old format (just state_dict) and new format (full checkpoint)
    if 'model_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['model_state_dict'])
        global_step = checkpoint.get('global_step', 0)
        episode = checkpoint.get('episode', 0)
        print(f"  Loaded from step {global_step}, episode {episode}")
        print(f"  Saved at: {checkpoint.get('timestamp', 'unknown')}")
        print(f"  Reason: {checkpoint.get('reason', 'unknown')}")
    else:
        # Old format - just the state dict
        agent.load_state_dict(checkpoint)
        global_step = 0
        episode = 0
        print("  Loaded model weights (old checkpoint format)")
    
    # Try to load associated log file
    log_path = checkpoint_path.replace('.pth', '_log.csv')
    if not os.path.exists(log_path):
        # Try alternative naming
        log_path = checkpoint_path.replace('dreamer_', 'training_log_').replace('.pth', '.csv')
    
    metrics_log = None
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        metrics_log = df.to_dict(orient='list')
        print(f"  Loaded training log: {log_path}")
    
    return global_step, episode, metrics_log


def emergency_save():
    """Emergency save function called on unexpected termination."""
    global _global_agent, _global_metrics_log, _global_step, _global_episode
    
    if _global_agent is not None:
        print("\n" + "="*60)
        print("EMERGENCY SAVE - Saving current state...")
        print("="*60)
        try:
            save_checkpoint(
                _global_agent, 
                _global_metrics_log, 
                _global_step, 
                _global_episode,
                filename_prefix="dreamer_emergency",
                reason="emergency"
            )
        except Exception as e:
            print(f"Emergency save failed: {e}")


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    signal_names = {
        signal.SIGTERM: "SIGTERM",
        signal.SIGINT: "SIGINT (Ctrl+C)",
    }
    if hasattr(signal, 'SIGHUP'):
        signal_names[signal.SIGHUP] = "SIGHUP"
    
    sig_name = signal_names.get(signum, f"Signal {signum}")
    print(f"\n\nReceived {sig_name}. Saving and exiting gracefully...")
    
    emergency_save()
    sys.exit(0)


def setup_signal_handlers():
    """Set up signal handlers for graceful termination."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # SIGHUP is Unix-only
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)
    
    # Register emergency save on exit
    atexit.register(emergency_save)
    
    print("Signal handlers configured for graceful termination")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def preprocess(obs):
    """Preprocess observation: resize to 64x64, grayscale, normalize."""
    # Resize to 64x64
    img = cv2.resize(obs, (64, 64))
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalize to [0, 1] and add channel dimension
    return (img / 255.0).astype(np.float32).reshape(1, 64, 64)


def compute_lambda_returns(rewards, values, gamma=GAMMA, lambda_=LAMBDA):
    """
    Compute V_λ targets for actor-critic training.
    """
    H = len(rewards)
    
    # Start from the bootstrap value (last value in the sequence)
    returns = values[-1]  # v(s_{t+H})
    
    targets = []
    
    # Iterate backwards through the horizon
    for t in reversed(range(H)):
        # V_λ(s_t) = r_t + γ * [(1-λ) * v(s_{t+1}) + λ * V_λ(s_{t+1})]
        returns = rewards[t] + gamma * ((1 - lambda_) * values[t + 1] + lambda_ * returns)
        targets.insert(0, returns)
    
    return torch.stack(targets)  # (H, B, 1)


def process_action_for_env(action):
    """Process network output action for environment."""
    processed = action.copy()
    # Steering stays in [-1, 1] (already there from tanh)
    # Gas: map from [-1, 1] to [0, 1]
    processed[1] = (action[1] + 1) / 2
    # Brake: map from [-1, 1] to [0, 1]
    processed[2] = (action[2] + 1) / 2
    return processed


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_world_model(agent, obs_batch, action_batch, reward_batch):
    """Train the world model (encoder, decoder, RSSM, reward model)."""
    B, T = obs_batch.shape[:2]
    
    # Initialize RSSM states
    h, z = agent.get_initial_state(B)
    
    # Collect losses over sequence
    recon_losses = []
    reward_losses = []
    kl_losses = []
    free_nats_tensor = torch.tensor(FREE_NATS, device=DEVICE)
    
    for t in range(T):
        obs_t = obs_batch[:, t]  # (B, 1, H, W)
        action_t = action_batch[:, t]  # (B, action_dim)
        reward_t = reward_batch[:, t]  # (B, 1)
        
        # Encode observation
        embed_t = agent.encoder(obs_t)
        
        # Update RSSM state (returns both prior and posterior)
        h, z, prior_dist, posterior_dist = agent.rssm.observe(z, action_t, h, embed_t)
        
        # Get state feature
        state = agent.get_state_feature(h, z)
        
        # Reconstruction loss
        recon = agent.decoder(state)
        recon_loss = F.mse_loss(recon, obs_t, reduction='none')
        recon_loss = recon_loss.sum(dim=[1, 2, 3]).mean()  # Sum over pixels, mean over batch
        recon_losses.append(recon_loss)
        
        # Reward prediction loss
        pred_reward = agent.reward_model(state)
        reward_loss = F.mse_loss(pred_reward, reward_t)
        reward_losses.append(reward_loss)
        
        # KL divergence: KL(posterior || prior) with free nats
        kl = D.kl_divergence(posterior_dist, prior_dist)
        kl = kl.sum(dim=-1).mean()  # Sum over latent dims, mean over batch
        kl = torch.max(kl, free_nats_tensor)
        kl_losses.append(kl)
    
    # Aggregate losses
    recon_loss = torch.stack(recon_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    
    total_loss = recon_loss + reward_loss + KL_SCALE * kl_loss
    
    # Optimize
    agent.model_opt.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent.encoder.parameters()) + 
        list(agent.decoder.parameters()) +
        list(agent.rssm.parameters()) + 
        list(agent.reward_model.parameters()),
        GRAD_CLIP
    )
    agent.model_opt.step()
    
    metrics = {
        'model_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'reward_loss': reward_loss.item(),
        'kl_loss': kl_loss.item()
    }
    
    return total_loss.item(), metrics


def train_actor_critic(agent, obs_batch, action_batch):
    """Train actor and critic using imagination in latent space."""
    B, T = obs_batch.shape[:2]
    
    with torch.no_grad():
        h, z = agent.get_initial_state(B)
        
        posterior_states = []
        for t in range(T):
            embed_t = agent.encoder(obs_batch[:, t])
            h, z, _, _ = agent.rssm.observe(z, action_batch[:, t], h, embed_t)
            state = agent.get_state_feature(h, z)
            posterior_states.append(state)
        
        # Stack: (T, B, state_dim) -> flatten to (T*B, state_dim)
        start_states = torch.stack(posterior_states, dim=0)  # (T, B, state_dim)
        start_states = start_states.reshape(-1, agent.state_dim)  # (T*B, state_dim)
    
    # Split states back into h and z
    dream_h = start_states[:, :agent.det_dim].clone()
    dream_z = start_states[:, agent.det_dim:].clone()
    
    # Storage for imagination
    imagined_states = [start_states]  # Include starting states
    imagined_rewards = []
    
    for _ in range(HORIZON):  # dream steps
        # Current state
        state = agent.get_state_feature(dream_h, dream_z)
        
        # Get action from actor (with gradients)
        mean, std = agent.actor(state)
        dist = D.Normal(mean, std)
        action = torch.tanh(dist.rsample())  # rsample for reparameterization
        
        # Transition using prior (imagination)
        dream_h, dream_z, _ = agent.rssm.imagine(dream_z, action, dream_h)
        
        # Get next state
        next_state = agent.get_state_feature(dream_h, dream_z)
        
        # Predict reward for next state
        reward = agent.reward_model(next_state)
        
        imagined_states.append(next_state)
        imagined_rewards.append(reward)
    
    # Stack tensors
    imagined_states = torch.stack(imagined_states, dim=0)  # (H+1, T*B, state_dim)
    imagined_rewards = torch.stack(imagined_rewards, dim=0)  # (H, T*B, 1)
    
    with torch.no_grad():
        # Values for target computation (stop gradient for critic targets)
        imagined_values = agent.critic(imagined_states.reshape(-1, agent.state_dim))
        imagined_values = imagined_values.reshape(HORIZON + 1, -1, 1)  # (H+1, T*B, 1)
    
    # Compute V_λ targets using detached values
    lambda_targets = compute_lambda_returns(imagined_rewards, imagined_values, GAMMA, LAMBDA)

    # Actor loss
    actor_states = imagined_states[:-1]  # (H, T*B, state_dim) - exclude bootstrap state
    
    # Recompute values with gradients through the critic
    actor_values = agent.critic(actor_states.reshape(-1, agent.state_dim))
    actor_values = actor_values.reshape(HORIZON, -1, 1)  # (H, T*B, 1)
    
    # Actor maximize expected value
    actor_loss = -actor_values.mean()
    
    agent.actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(agent.actor.parameters(), GRAD_CLIP)
    agent.actor_opt.step()
    
    # Critic loss
    critic_states = imagined_states[:-1].detach()  # (H, T*B, state_dim)
    critic_targets = lambda_targets.detach()  # (H, T*B, 1)
    
    # Predict values
    pred_values = agent.critic(critic_states.reshape(-1, agent.state_dim))
    pred_values = pred_values.reshape(HORIZON, -1, 1)
    
    # MSE loss
    critic_loss = F.mse_loss(pred_values, critic_targets)
    
    agent.critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(agent.critic.parameters(), GRAD_CLIP)
    agent.critic_opt.step()
    
    return actor_loss.item(), critic_loss.item()


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def main():
    global _global_agent, _global_metrics_log, _global_step, _global_episode
    
    print("=" * 60)
    print(f"Dreamer Training on {ENV_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint directory: {_save_dir}")
    print("=" * 60)
    
    # Set up signal handlers for graceful termination
    setup_signal_handlers()
    ensure_save_dir()
    
    # Initialize environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = Dreamer(
        action_dim=action_dim,
        stoch_dim=STOCH_DIM,
        det_dim=DET_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        device=DEVICE
    )
    
    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=100000, obs_shape=(64, 64), action_dim=action_dim)
    
    # Initialize metrics logging
    metrics_log = {
        'step': [],
        'episode': [],
        'reward': [],
        'model_loss': [],
        'actor_loss': [],
        'critic_loss': []
    }
    
    # Initialize training state
    global_step = 0
    episode = 0
    
    # ==== Resume from checkpoint if specified ====
    if RESUME_PATH is not None:
        checkpoint_path = RESUME_PATH
        if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(_save_dir, RESUME_PATH)
        
        if os.path.exists(checkpoint_path):
            global_step, episode, loaded_metrics = load_checkpoint(agent, checkpoint_path)
            if loaded_metrics is not None:
                metrics_log = loaded_metrics
            print(f"Resuming training from step {global_step}, episode {episode}")
        else:
            print(f"Resume checkpoint not found: {checkpoint_path}")
            print("Starting fresh training...")
    
    # Set global references for signal handlers
    _global_agent = agent
    _global_metrics_log = metrics_log
    _global_step = global_step
    _global_episode = episode
    
    # Track best reward for checkpointing
    best_avg_reward = -float('inf')
    recent_rewards = []
    
    # ==== Phase 1: Seed buffer with random episodes ====
    if global_step == 0:  # Only seed if starting fresh
        print(f"\nSeeding buffer with {SEED_EPISODES} random episodes...")
        
        for ep in range(SEED_EPISODES):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action = env.action_space.sample()
                obs_processed = preprocess(obs)
                
                # Action repeat
                total_reward = 0
                for _ in range(ACTION_REPEAT):
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        done = True
                        break
                
                buffer.add(obs_processed, action, total_reward, done)
                obs = next_obs
                ep_reward += total_reward
            
            print(f"  Seed episode {ep+1}: reward = {ep_reward:.2f}")
        
        print(f"Buffer size: {len(buffer)}")
    else:
        print(f"\nSkipping buffer seeding (resuming from step {global_step})")
        print("Note: Buffer state is not saved/restored - collecting new experience...")
    
    # ==== Phase 2: Main training loop ====
    print(f"\nStarting main training loop...")
    
    episode_reward = 0
    
    # Reset environment
    obs, _ = env.reset()
    
    # Initialize agent states for interaction
    prev_action = torch.zeros(1, action_dim, device=DEVICE)
    prev_h, prev_z = agent.get_initial_state(batch_size=1)
    
    # Track losses for logging
    recent_model_loss = 0
    recent_actor_loss = 0
    recent_critic_loss = 0
    
    try:
        while global_step < TOTAL_ENV_STEPS:
            # ==== Collect experience ====
            for _ in range(COLLECT_INTERVAL):
                obs_processed = preprocess(obs)
                
                # Get action from agent
                action_tensor, prev_h, prev_z = agent.get_action(
                    obs_processed, prev_action, prev_h, prev_z,
                    deterministic=False
                )
                action = action_tensor.cpu().numpy()[0]
                
                # Add exploration noise
                action = action + np.random.normal(0, 0.3, size=action.shape)
                action = np.clip(action, -1, 1)
                
                # Process for environment
                env_action = process_action_for_env(action)
                
                # Step with action repeat
                total_reward = 0
                done = False
                for _ in range(ACTION_REPEAT):
                    next_obs, reward, terminated, truncated, _ = env.step(env_action)
                    total_reward += reward
                    if terminated or truncated:
                        done = True
                        break
                
                # Store transition
                buffer.add(obs_processed, action, total_reward, done)
                
                # Update state
                obs = next_obs
                prev_action = action_tensor
                global_step += 1
                episode_reward += total_reward
                
                # Update global state for signal handlers
                _global_step = global_step
                
                # ==== Autosave checkpoint ====
                if global_step % AUTOSAVE_INTERVAL == 0:
                    save_checkpoint(
                        agent, metrics_log, global_step, episode,
                        filename_prefix="dreamer_autosave",
                        reason="autosave"
                    )
                
                # Episode ended
                if done:
                    episode += 1
                    _global_episode = episode
                    
                    recent_rewards.append(episode_reward)
                    if len(recent_rewards) > 100:
                        recent_rewards.pop(0)
                    
                    avg_reward = np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else episode_reward
                    
                    # Log every episode
                    print(f"Step {global_step:7d} | Episode {episode:4d} | "
                          f"Reward: {episode_reward:7.2f} | Avg(10): {avg_reward:7.2f}")
                    
                    # Log metrics
                    metrics_log['step'].append(global_step)
                    metrics_log['episode'].append(episode)
                    metrics_log['reward'].append(episode_reward)
                    metrics_log['model_loss'].append(recent_model_loss)
                    metrics_log['actor_loss'].append(recent_actor_loss)
                    metrics_log['critic_loss'].append(recent_critic_loss)
                    
                    # Update global metrics for signal handlers
                    _global_metrics_log = metrics_log
                    
                    # Save best model
                    if len(recent_rewards) >= 10 and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        save_checkpoint(
                            agent, metrics_log, global_step, episode,
                            filename_prefix="dreamer_best",
                            reason="best_reward"
                        )
                    
                    # Reset for new episode
                    obs, _ = env.reset()
                    prev_action = torch.zeros(1, action_dim, device=DEVICE)
                    prev_h, prev_z = agent.get_initial_state(batch_size=1)
                    episode_reward = 0
            
            # ==== Training phase ====
            agent.train()
            
            model_losses = []
            actor_losses = []
            critic_losses = []
            
            for _ in range(TRAIN_STEPS):
                # Sample batch
                try:
                    obs_batch, action_batch, reward_batch = buffer.sample_sequence(
                        BATCH_SIZE, SEQ_LEN, DEVICE
                    )
                except ValueError as e:
                    print(f"Warning: {e}. Skipping training step.")
                    continue
                
                # Train world model
                model_loss, _ = train_world_model(agent, obs_batch, action_batch, reward_batch)
                model_losses.append(model_loss)
                
                # Train actor-critic
                actor_loss, critic_loss = train_actor_critic(agent, obs_batch, action_batch)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # Update recent losses for logging
            if model_losses:
                recent_model_loss = np.mean(model_losses)
                recent_actor_loss = np.mean(actor_losses)
                recent_critic_loss = np.mean(critic_losses)
            
            agent.eval()
            
            print(f"  Training @ step {global_step}: "
                  f"model={recent_model_loss:.4f}, "
                  f"actor={recent_actor_loss:.4f}, "
                  f"critic={recent_critic_loss:.4f}")
            
            # Periodic checkpoint save
            if global_step % SAVE_INTERVAL == 0:
                save_checkpoint(
                    agent, metrics_log, global_step, episode,
                    filename_prefix=f"dreamer_step{global_step}",
                    reason="periodic"
                )
                
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user (Ctrl+C)")
        print("="*60)
        save_checkpoint(
            agent, metrics_log, global_step, episode,
            filename_prefix="dreamer_interrupted",
            reason="user_interrupt"
        )
    
    except Exception as e:
        print("\n" + "="*60)
        print(f"Training crashed with error: {e}")
        print("="*60)
        save_checkpoint(
            agent, metrics_log, global_step, episode,
            filename_prefix="dreamer_crash",
            reason=f"crash_{type(e).__name__}"
        )
        raise  # Re-raise the exception after saving
    
    # ==== Training completed successfully ====
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    
    # Final save
    save_checkpoint(
        agent, metrics_log, global_step, episode,
        filename_prefix="dreamer_final",
        reason="training_complete"
    )
    
    # Also save to root directory for compatibility
    torch.save(agent.state_dict(), "dreamer.pth")
    df = pd.DataFrame(metrics_log)
    df.to_csv("training_log.csv", index=False)
    print("\nAlso saved to root directory:")
    print("  Model: dreamer.pth")
    print("  Log: training_log.csv")
    
    # Clear emergency save flag
    _global_agent = None
    
    env.close()


if __name__ == "__main__":
    main()