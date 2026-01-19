import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import cv2
import pandas as pd
import os

# --- IMPORTS ---
from replay import ReplayBuffer
from dreamer import Dreamer

# --- HYPERPARAMETERS ---
ENV_NAME = "CarRacing-v3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
STOCHASTIC_DIM = 30    # z dimension (stochastic latent)
DETERMINISTIC_DIM = 200  # h dimension (RNN hidden state)
HIDDEN_DIM = 300       # Hidden layer size for actor/critic/reward

# Dreamer Configuration
BATCH_SIZE = 50
SEQ_LEN = 32      # Reduced to 32 for stability and memory
HORIZON = 15      # Dream Horizon (H)
GAMMA = 0.99      # Discount factor
LAMBDA = 0.95     # GAE Parameter for Value Targets
FREE_NATS = 3.0   # KL divergence free threshold
KL_SCALE = 0.1    # KL loss coefficient

# Training Schedule
TOTAL_STEPS = 1000
TRAIN_EVERY = 50  # Interact for 50 steps...
TRAIN_STEPS = 50  # ...then train for 50 gradient steps
ACTION_REPEAT = 2 # Execute the same action 2x (Crucial for CarRacing)

class EarlyStopper:
    """
    Checks if the agent has stopped improving to save time.
    """
    def __init__(self, patience=10, min_delta=1.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False

    def __call__(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            print(f"  --> New Best Score: {self.best_score:.2f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def preprocess(obs):
    """Resize to 64x64, Grayscale, Normalize to 0-1"""
    # Obs is typically (96, 96, 3) for CarRacing
    img = cv2.resize(obs, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).reshape(1, 64, 64)

def compute_lambda_returns(rewards, values, gamma=GAMMA, lambda_=LAMBDA):
    """
    Computes V_lambda targets (Equation 6 in Dreamer paper).
    """
    # rewards: (Horizon, Batch, 1)
    # values: (Horizon, Batch, 1)
    returns = values[-1] # Bootstrap from last estimated value
    targets = []
    
    # Iterate backwards through time

    for t in reversed(range(len(rewards))):
        r_t = rewards[t]
        if t == len(rewards) - 1:
            # Last timestep: just the reward + discounted bootstrap
            returns = r_t + gamma * values[-1]
        else:
            v_next = values[t + 1]
            returns = r_t + gamma * ((1 - lambda_) * v_next + lambda_ * returns)
        targets.insert(0, returns)
    
    return torch.stack(targets)

def main():
    # 1. Setup
    print(f"Initializing Dreamer on {DEVICE}...")
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    agent = Dreamer(
        action_dim=3, 
        stochastic_dim=STOCHASTIC_DIM,
        deterministic_dim=DETERMINISTIC_DIM,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE
    )
    buffer = ReplayBuffer(capacity=10000)
    
    # --- METRICS LOGGER ---
    metrics = {
        "step":[],
        "reward":[],
        "model_loss":[],
        "actor_loss":[],
        "critic_loss": []
    }
    
    stopper = EarlyStopper(patience=15, min_delta=5.0) 

    # 2. Seed Data
    print("Collecting seed episodes...")
    obs, _ = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        obs_p = preprocess(obs)
        
        # Apply Action Repeat for Seeding
        total_reward = 0
        done = False
        for _ in range(ACTION_REPEAT):
            next_obs, r, term, trunc, _ = env.step(action)
            total_reward += r
            if term or trunc:
                done = True
                break
        
        buffer.add(obs_p, action, total_reward, done)
        obs = next_obs
        if done: obs, _ = env.reset()

    # 3. Main Loop
    print("Starting Training Loop...")
    obs, _ = env.reset()
    
    # Reset Internal States
    prev_act = torch.zeros(1, 3).to(DEVICE)
    prev_rnn, prev_z = agent.get_initial_state(batch_size=1)
    
    global_step = 0
    episode_reward = 0
    
    # Logging Placeholders
    last_m_loss = 0.0
    last_a_loss = 0.0
    last_c_loss = 0.0

    while global_step < TOTAL_STEPS:
        
        # --- PHASE 1: INTERACTION (WAKE) ---
        obs_p = preprocess(obs)
        
        # Get Action (Deterministic during inference is usually not done for training, 
        # but we use stochastic sampling in get_action by default)
        action_tensor, rnn, z = agent.get_action(obs_p, prev_act, prev_rnn, prev_z)
        action = action_tensor.cpu().numpy()[0]
        
        # CarRacing Specific: Map tanh output to proper ranges
        # Steering stays in [-1, 1]
        # Gas and Brake need [0, 1]

        action[1] = (action[1] + 1) / 2 # Gas
        action[2] = (action[2] + 1) / 2 # Brake
        
        # Action Repeat Step
        total_reward = 0
        done = False
        for _ in range(ACTION_REPEAT):
            next_obs, r, term, trunc, _ = env.step(action)
            total_reward += r
            if term or trunc:
                done = True
                break

        buffer.add(obs_p, action, total_reward, done)
        
        obs = next_obs
        prev_act = action_tensor
        prev_rnn = rnn
        prev_z = z
        global_step += 1
        episode_reward += total_reward
        
        if done:
            print(f"Step {global_step}: Episode Reward = {episode_reward:.2f}")
            
            # Logging
            metrics["step"].append(global_step)
            metrics["reward"].append(episode_reward)
            metrics["model_loss"].append(last_m_loss)
            metrics["actor_loss"].append(last_a_loss)
            metrics["critic_loss"].append(last_c_loss)
            
            # Check Convergence
            stopper(episode_reward)
            if stopper.early_stop:
                print(f"!!! CONVERGENCE REACHED !!!")
                break

            # Reset Env
            obs, _ = env.reset()
            prev_act = torch.zeros(1, 3).to(DEVICE)
            prev_rnn, prev_z = agent.get_initial_state(batch_size=1)
            episode_reward = 0

        # --- PHASE 2: TRAINING (SLEEP) ---
        if global_step % TRAIN_EVERY == 0:
            agent.train() # Switch to training mode
            
            m_losses, a_losses, c_losses = [], [], []

            for _ in range(TRAIN_STEPS):
                
                # ---------------------------------------------------------
                # A. DATA LOADING
                # ---------------------------------------------------------
                obs_b, act_b, rew_b = buffer.sample_sequence(BATCH_SIZE, SEQ_LEN, DEVICE)
                
                # Embed Images (Batch * Seq)
                B, T, C, H, W = obs_b.shape
                obs_flat = obs_b.view(B*T, C, H, W)
                embed_flat = agent.encoder(obs_flat)
                embed = embed_flat.view(B, T, -1)

                # ---------------------------------------------------------
                # B. DYNAMICS LEARNING (World Model)
                # ---------------------------------------------------------
                prev_rnn, prev_z = agent.get_initial_state(batch_size=B)
                
                posterior_states = []
                kl_losses = []
                
                # Unroll RSSM
                for t in range(T):
                    # 1. Posterior (Reality)
                    rnn_p, z_p, dist_p = agent.rssm.posterior(
                        prev_z, act_b[:, t], prev_rnn, embed[:, t]
                    )
                    
                    # 2. Prior (Dream) - For KL Calculation
                    _, _, dist_q = agent.rssm.transition(
                        prev_z, act_b[:, t], prev_rnn
                    )
                    
                    # 3. KL Divergence (Posterior || Prior) with free nats
                    kl = D.kl_divergence(dist_p, dist_q).mean()
                    kl = torch.maximum(kl, torch.tensor(FREE_NATS).to(DEVICE))
                    kl_losses.append(kl)
                    
                    # Save state for features
                    posterior_states.append(torch.cat([rnn_p, z_p], dim=1))
                    
                    prev_rnn = rnn_p
                    prev_z = z_p
                
                # Stack Features: (Batch, Seq, State_Dim)
                model_feat = torch.stack(posterior_states, dim=1)
                flat_feat = model_feat.view(B*T, -1)
                
                # Reconstruct
                recon_imgs = agent.decoder(flat_feat).view(B, T, 1, 64, 64)
                pred_rewards = agent.reward_model(flat_feat).view(B, T, 1)
                
                # Calculate Losses
                loss_recon = F.mse_loss(recon_imgs, obs_b)
                loss_reward = F.mse_loss(pred_rewards, rew_b)
                loss_kl = KL_SCALE * torch.mean(torch.stack(kl_losses))
                
                model_loss = loss_recon + loss_reward + loss_kl

                agent.model_opt.zero_grad()
                model_loss.backward()
                nn.utils.clip_grad_norm_(agent.rssm.parameters(), 100)
                agent.model_opt.step()
                
                m_losses.append(model_loss.item())

                # ---------------------------------------------------------
                # C. BEHAVIOR LEARNING (Actor-Critic)
                # ---------------------------------------------------------
                # 1. Detach World Model State
                # We start dreaming from the "Reality" states we just found.
                # Detach prevents Actor gradients from changing the World Model.
                start_states = model_feat.detach().view(B*T, -1)
                
                dream_rnn = start_states[:, :agent.det_dim]
                dream_z = start_states[:, agent.det_dim:]
                
                dream_rewards = []
                dream_values = []
                dream_states = []
                # 2. Dream Forward
                for step in range(HORIZON):
                    feat = torch.cat([dream_rnn, dream_z], dim=1)
                    dream_states.append(feat)
                    # Actor Action (with Gradients)
                    action_out = agent.actor(feat)
                    mu, std = action_out.chunk(2, dim=1)
                    dist_act = D.Normal(mu, std)
                    action = torch.tanh(dist_act.rsample()) # rsample keeps grads
                    
                    # World Model Transition (Prior)
                    dream_rnn, dream_z, _ = agent.rssm.transition(dream_z, action, dream_rnn)
                    
                    # Predict Reward & Value
                    dream_feat = torch.cat([dream_rnn, dream_z], dim=1)
                    r = agent.reward_model(dream_feat)
                    v = agent.critic(dream_feat)
                    
                    dream_rewards.append(r)
                    dream_values.append(v)
                
                # Stack Dreams
                dream_rewards = torch.stack(dream_rewards)  # (H, B*T, 1)
                dream_values = torch.stack(dream_values)    # (H, B*T, 1)
                dream_states = torch.stack(dream_states)    # (H, B*T, state_dim)
                
                # 3. Calculate Targets (V_lambda)
                target_values = compute_lambda_returns(dream_rewards, dream_values)
                
                # 4. Update Actor
                # Maximize Value -> Minimize Negative Value
                actor_loss = -torch.mean(target_values)
                
                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()


                
                # 5. Update Critic - Re-predict from detached states
                dream_states_detached = dream_states.detach()  # (H, B*T, state_dim)
                predicted_values = agent.critic(dream_states_detached.view(-1, agent.state_dim))
                predicted_values = predicted_values.view(HORIZON, -1, 1)
                
                critic_loss = F.mse_loss(predicted_values, target_values.detach())
                
                agent.critic_opt.zero_grad()
                critic_loss.backward()
                agent.critic_opt.step()


                
                a_losses.append(actor_loss.item())
                c_losses.append(critic_loss.item())

            # Update Log Variables
            last_m_loss = np.mean(m_losses)
            last_a_loss = np.mean(a_losses)
            last_c_loss = np.mean(c_losses)
            
            agent.eval() # Return to Eval mode

    # 4. Save Artifacts
    print("Saving metrics to 'training_log.csv'...")
    df = pd.DataFrame(metrics)
    df.to_csv("training_log.csv", index=False)
    
    print("Saving model to 'dreamer.pth'...")
    torch.save(agent.state_dict(), "dreamer.pth")
    env.close()

if __name__ == "__main__":
    main()
