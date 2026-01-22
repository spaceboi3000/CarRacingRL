"""
Dreamer Training Script for CarRacing-v3

Based on Hafner et al. (2020) "Dream to Control: Learning Behaviors by Latent Imagination"

Training consists of three interleaved processes:
1. Environment Interaction: Collect experience using current policy
2. Dynamics Learning: Train world model on collected experience  
3. Behavior Learning: Train actor-critic by imagining in latent space
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


# Logging
LOG_INTERVAL = 10           # Log every N episodes
SAVE_INTERVAL = 50000       # Save checkpoint every N steps

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def preprocess(obs):
    # Resize to 64x64
    img = cv2.resize(obs, (64, 64))
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalize to [0, 1] and add channel dimension
    return (img / 255.0).astype(np.float32).reshape(1, 64, 64)


def compute_lambda_returns(rewards, values, gamma=GAMMA, lambda_=LAMBDA):
    """
    Compute V_λ 

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
    
    for _ in range(HORIZON): #dream steps
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
    # states: (H+1, T*B, state_dim) - H imagined + 1 starting state
    # rewards: (H, T*B, 1)
    imagined_states = torch.stack(imagined_states, dim=0)  # (H+1, T*B, state_dim)
    imagined_rewards = torch.stack(imagined_rewards, dim=0)  # (H, T*B, 1)
    
    with torch.no_grad():
        # Values for target computation (stop gradient for critic targets)
        imagined_values = agent.critic(imagined_states.reshape(-1, agent.state_dim))
        imagined_values = imagined_values.reshape(HORIZON + 1, -1, 1)  # (H+1, T*B, 1)
    
    # Compute V_λ targets using detached values
    lambda_targets = compute_lambda_returns(imagined_rewards, imagined_values, GAMMA, LAMBDA)  # (H, T*B, 1)

    # without gradients
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
    
  
    # Critic minimize MSE between v(s) and V_λ(s)
    # Use detached states and targets = no gradients through world model
    
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
    print(f"=" * 60)
    print(f"Dreamer Training on {ENV_NAME}")
    print(f"Device: {DEVICE}")
    print(f"=" * 60)
    
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
    
    # Metrics logging
    metrics_log = {
        'step': [],
        'episode': [],
        'reward': [],
        'model_loss': [],
        'actor_loss': [],
        'critic_loss': []
    }
    
    # Track best reward for checkpointing
    best_avg_reward = -float('inf')
    recent_rewards = []
    
    # ==== Phase 1: Seed buffer with random episodes ====
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
    
    # ==== Phase 2: Main training loop ====
    print(f"\nStarting main training loop...")
    
    global_step = 0
    episode = 0
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
                
                # Episode ended
                if done:
                    episode += 1
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
                    
                    # Save best model
                    if len(recent_rewards) >= 10 and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(agent.state_dict(), "dreamer_best.pth")
                        print(f"  --> New best average: {best_avg_reward:.2f}, saved to dreamer_best.pth")
                    
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
                checkpoint_path = f"dreamer_step{global_step}.pth"
                torch.save(agent.state_dict(), checkpoint_path)
                df = pd.DataFrame(metrics_log)
                df.to_csv(f"training_log_step{global_step}.csv", index=False)
                print(f"  --> Checkpoint saved: {checkpoint_path}")
    except KeyboardInterrupt:
        print("\nCaptured Ctrl+C. Stopping training and saving results")
    
    # ==== Save results ====
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    
    # Save metrics
    df = pd.DataFrame(metrics_log)
    df.to_csv("training_log.csv", index=False)
    print("Saved training log to 'training_log.csv'")
    
    # Save model
    torch.save(agent.state_dict(), "dreamer.pth")
    print("Saved model to 'dreamer.pth'")
    
    env.close()


if __name__ == "__main__":
    main()
