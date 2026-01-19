import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from obsolete.dreamer_arch import WorldModel, Actor, Critic, RewardModel

# --- Configuration ---
ENV_NAME = "CarRacing-v3" 
SEQ_LEN = 50              # Batch sequence length for World Model training
HORIZON = 15              # Dream horizon length
GAMMA = 0.99              # Discount factor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper: Preprocess Image ---
def preprocess(obs):
    # Resize to 64x64, Grayscale, Normalize
    img = cv2.resize(obs, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).reshape(1, 64, 64)

def get_feat(rnn_state, z):
    # Concatenate deterministic and stochastic states
    return torch.cat([rnn_state, z], dim=-1)

# --- The Dreamer Agent ---
class DreamerAgent:
    def __init__(self):
        print(f"Initializing Dreamer on {DEVICE}...")
        
        # 1. The Brain (World Model)
        self.world_model = WorldModel().to(DEVICE)
        self.reward_model = RewardModel().to(DEVICE)
        
        # 2. The Personality (Actor-Critic)
        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        
        # Optimizers
        self.world_opt = torch.optim.Adam(
            list(self.world_model.parameters()) + list(self.reward_model.parameters()), 
            lr=6e-4
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        
        # Replay Buffer
        self.dataset_obs = []
        self.dataset_act = []
        self.dataset_rew = []

    # --- Real-World Interaction ---
    def get_action(self, obs, prev_rnn, prev_action, explore=False):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prev_action_tensor = torch.tensor(prev_action, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # 1. Encode image
            embed = self.world_model.encoder(obs_tensor)
            
            # 2. Update Recurrent State
            rnn_input = torch.cat([embed, prev_action_tensor], dim=1)
            rnn_state = self.world_model.gru(rnn_input, prev_rnn)
            
            # 3. Get Posterior (Current Belief)
            post_input = torch.cat([rnn_state, embed], dim=1)
            mu, _ = self.world_model.posterior_net(post_input).chunk(2, dim=1)
            z = mu # Use mean for stable action selection
            
            # 4. Actor selects action
            feat = get_feat(rnn_state, z)
            action = self.actor.get_action(feat, deterministic=not explore)
            
            return action.cpu().numpy()[0], rnn_state

    # --- Phase A: Learn Dynamics (and Reward) ---
    def train_world_model(self, iterations=100):
        self.world_model.train()
        self.reward_model.train()
        
        for _ in range(iterations):
            # Sample Random Sequence
            idx = np.random.randint(0, len(self.dataset_obs) - SEQ_LEN)
            obs_seq = torch.tensor(np.array(self.dataset_obs[idx:idx+SEQ_LEN]), dtype=torch.float32).to(DEVICE)
            act_seq = torch.tensor(np.array(self.dataset_act[idx:idx+SEQ_LEN]), dtype=torch.float32).to(DEVICE)
            rew_seq = torch.tensor(np.array(self.dataset_rew[idx:idx+SEQ_LEN]), dtype=torch.float32).to(DEVICE).unsqueeze(-1)
            
            rnn_state = torch.zeros(1, 200).to(DEVICE)
            total_loss = 0
            
            # Unroll Sequence
            for t in range(SEQ_LEN):
                img_t = obs_seq[t].unsqueeze(0)
                act_t = act_seq[t].unsqueeze(0)
                rew_t = rew_seq[t].unsqueeze(0)
                
                # Forward Pass
                recon, rnn_state, mu, logvar = self.world_model(img_t, act_t, rnn_state)
                
                # Reward Prediction
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                feat = get_feat(rnn_state, z)
                pred_rew = self.reward_model(feat)
                
                # Losses
                recon_loss = F.mse_loss(recon, img_t)
                reward_loss = F.mse_loss(pred_rew, rew_t)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Paper combines these: Reconstruction + Reward + KL
                loss = recon_loss + reward_loss + (0.1 * kl_loss)
                total_loss += loss

            self.world_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100)
            self.world_opt.step()

    # --- Phase B: Dream (Learn Behavior) ---
    def train_behavior(self, iterations=100):
        # Freeze World Model to save compute
        for p in self.world_model.parameters(): p.requires_grad = False
        for p in self.reward_model.parameters(): p.requires_grad = False
        
        for _ in range(iterations):
            # 1. Start from a real state in memory
            idx = np.random.randint(0, len(self.dataset_obs) - 1)
            obs_init = torch.tensor(preprocess(self.dataset_obs[idx]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            rnn_state = torch.zeros(1, 200).to(DEVICE) # Simplified: Start fresh RNN or store RNN states in buffer
            
            # Encode initial state
            with torch.no_grad():
                embed = self.world_model.encoder(obs_init)
                # Warm up RNN one step with zero action
                prev_act = torch.zeros(1, 3).to(DEVICE)
                rnn_state = self.world_model.gru(torch.cat([embed, prev_act], 1), rnn_state)
                post_in = torch.cat([rnn_state, embed], 1)
                mu, logvar = self.world_model.posterior_net(post_in).chunk(2, 1)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            
            # 2. Dream Forward
            rewards = []
            values = []
            log_probs = [] # Optional: for PPO-style, but we use Reparameterization (Dreamer)
            
            for t in range(HORIZON):
                feat = get_feat(rnn_state, z).detach() # Detach here is crucial for Straight-Through Gradients
                
                # Actor picks action
                action = self.actor.get_action(feat)
                
                # World Model predicts Next State (Prior)
                rnn_in = torch.cat([z, action], dim=1)
                rnn_state = self.world_model.gru(rnn_in, rnn_state)
                prior = self.world_model.prior_net(rnn_state)
                mu, logvar = prior.chunk(2, dim=1)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                
                # Predict Reward & Value
                feat_next = get_feat(rnn_state, z)
                pred_rew = self.reward_model(feat_next)
                pred_val = self.critic(feat_next)
                
                rewards.append(pred_rew)
                values.append(pred_val)
            
            # 3. Calculate lambda-targets (Dreamer Value Estimation)
            # Simplified: Just sum rewards + final value
            dream_rewards = torch.stack(rewards).squeeze()
            dream_values = torch.stack(values).squeeze()
            
            # Bootstrap value (Reverse accumulation)
            returns = []
            ret = dream_values[-1]
            for r in reversed(dream_rewards[:-1]):
                ret = r + GAMMA * ret
                returns.insert(0, ret)
            returns = torch.stack(returns)
            
            # 4. Update Actor & Critic
            # Critic Loss: Match the calculated returns
            critic_loss = F.mse_loss(dream_values[:-1], returns.detach())
            
            # Actor Loss: Maximize the expected returns
            # (In Dreamer, gradients flow back through the world model! "Analytic Gradients")
            actor_loss = -torch.mean(returns) 
            
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
        # Unfreeze
        for p in self.world_model.parameters(): p.requires_grad = True
        for p in self.reward_model.parameters(): p.requires_grad = True

# --- Main Training Loop ---
if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode="rgb_array") 
    agent = DreamerAgent()
    
    print("Collecting seed episodes...")
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs_p = preprocess(obs)
            obs, reward, term, trunc, _ = env.step(action)
            agent.dataset_obs.append(obs_p)
            agent.dataset_act.append(action)
            agent.dataset_rew.append(reward) # Capture Reward
            if term or trunc: break
            
    print("Starting Dreamer Training Loop...")
    
    for episode in range(100):
        obs, _ = env.reset()
        prev_act = np.zeros(3)
        rnn_state = torch.zeros(1, 200).to(DEVICE)
        
        total_reward = 0
        done = False
        
        while not done:
            # 1. Wake Phase: Act in Environment
            obs_p = preprocess(obs)
            
            # Explore heavily at first, then exploit
            if episode < 5:
                action = env.action_space.sample()
            else:
                action, rnn_state = agent.get_action(obs_p, rnn_state, prev_act)
            
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            agent.dataset_obs.append(obs_p)
            agent.dataset_act.append(action)
            agent.dataset_rew.append(reward)
            
            obs = next_obs
            prev_act = action
            total_reward += reward
        
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Memory: {len(agent.dataset_obs)}")
        
        # 2. Sleep Phase: Train on recent data
        if len(agent.dataset_obs) > SEQ_LEN + 1:
            agent.train_world_model(iterations=100) # Learn Physics
            agent.train_behavior(iterations=100)    # Dream