import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from convVAE import Encoder, Decoder
from RSSM import RSSM


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=300, min_std=0.1, init_std=5.0):
        super().__init__()
        self.min_std = min_std
        self.init_std = init_std
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.net(state)
        
        # Mean: scaled tanh allows the agent to saturate actions
        mean = self.mean_head(features)
        
        # Std: softplus ensures positive, with minimum for numerical stability
        std_raw = self.std_head(features)
        std = F.softplus(std_raw + self.init_std) + self.min_std
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        
        if deterministic:
            # Deterministic action
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Sample from Normal, then squash with tanh
            dist = D.Normal(mean, std)
            sample = dist.rsample()  # Reparameterized sample
            action = torch.tanh(sample)
            log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    # Value Model: v_ψ(s_t)
    def __init__(self, state_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class RewardModel(nn.Module):
    # q(r_t | s_t)
    def __init__(self, state_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class Dreamer(nn.Module):
    def __init__(self, action_dim=3, stoch_dim=30, det_dim=200, 
                 hidden_dim=300, embed_dim=1024, device="cpu"):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Dimensions
        self.stoch_dim = stoch_dim      # z dimension
        self.det_dim = det_dim          # h (RNN hidden) dimension
        self.state_dim = stoch_dim + det_dim  # Combined feature dimension
        self.embed_dim = embed_dim
        
        # World Model Components
        self.encoder = Encoder(in_channels=1).to(device)
        self.decoder = Decoder(state_dim=self.state_dim).to(device)
        self.rssm = RSSM(
            stoch_dim=stoch_dim, 
            det_dim=det_dim,
            action_dim=action_dim, 
            embed_dim=embed_dim
        ).to(device)
        self.reward_model = RewardModel(self.state_dim, hidden_dim).to(device)

        # Behavior Learning Components
        self.actor = Actor(self.state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(self.state_dim, hidden_dim).to(device)
        
        # Optimizers (from Appendix A)
        # World model: lr = 6e-4
        self.model_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.rssm.parameters()) + 
            list(self.reward_model.parameters()), 
            lr=6e-4
        )
        # Actor: lr = 8e-5
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        # Critic: lr = 8e-5
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

    def get_initial_state(self, batch_size=1):
        return self.rssm.initial_state(batch_size, self.device)

    def encode(self, obs):
        return self.encoder(obs)
    
    def decode(self, state):
        return self.decoder(state)

    def observe(self, obs, prev_action, prev_h, prev_z):
        embed = self.encoder(obs)
        h, z, prior_dist, posterior_dist = self.rssm.observe(
            prev_z, prev_action, prev_h, embed
        )
        return h, z, prior_dist, posterior_dist

    def imagine(self, prev_action, prev_h, prev_z):
        return self.rssm.imagine(prev_z, prev_action, prev_h)

    def get_state_feature(self, h, z):
        return torch.cat([h, z], dim=-1)

    @torch.no_grad()
    def get_action(self, obs, prev_action, prev_h, prev_z, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # Add batch dimension
        
        # Get embedding and update state
        embed = self.encoder(obs)
        h, z, _, _ = self.rssm.observe(prev_z, prev_action, prev_h, embed)
        
        # Get action from policy
        state = self.get_state_feature(h, z)
        action, _ = self.actor.get_action(state, deterministic=deterministic)
        
        return action, h, z

    def world_model_loss(self, obs_seq, action_seq, reward_seq, free_nats=3.0, kl_scale=1.0):
        B, T = obs_seq.shape[:2]
        device = obs_seq.device

        h, z = self.get_initial_state(B)
        
        # Storage for losses
        recon_losses = []
        reward_losses = []
        kl_losses = []
        
        # Process sequence
        for t in range(T):
            # Get current observation embedding
            obs_t = obs_seq[:, t]  # (B, 1, H, W)
            embed_t = self.encoder(obs_t)  # (B, embed_dim)
            
            # Update state with posterior (observe)
            h, z, prior_dist, posterior_dist = self.rssm.observe(
                z, action_seq[:, t], h, embed_t
            )
            
            state = self.get_state_feature(h, z)
            
            # Reconstruction loss
            recon = self.decoder(state)
            recon_loss = F.mse_loss(recon, obs_t, reduction='none').sum(dim=[1,2,3]).mean()
            recon_losses.append(recon_loss)
            
            # Reward prediction loss
            pred_reward = self.reward_model(state)
            reward_loss = F.mse_loss(pred_reward, reward_seq[:, t])
            reward_losses.append(reward_loss)
            
            # KL divergence: KL(posterior || prior)
            # With free nats: max(KL, free_nats)
            kl = D.kl_divergence(posterior_dist, prior_dist).sum(-1).mean()
            kl = torch.max(kl, torch.tensor(free_nats, device=device))
            kl_losses.append(kl)
        
        # Aggregate losses
        recon_loss = torch.stack(recon_losses).mean()
        reward_loss = torch.stack(reward_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()
        
        total_loss = recon_loss + reward_loss + kl_scale * kl_loss
        
        metrics = {
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics

    def imagine_trajectory(self, start_h, start_z, horizon):
        h, z = start_h, start_z
        
        states = []
        actions = []
        rewards = []
        
        for _ in range(horizon):
            # Current state
            state = self.get_state_feature(h, z)
            states.append(state)
            
            # Sample action from policy
            action, _ = self.actor.get_action(state, deterministic=False)
            actions.append(action)
            
            # Imagine next state (prior transition)
            h, z, _ = self.rssm.imagine(z, action, h)
            
            # Predict reward for new state
            next_state = self.get_state_feature(h, z)
            reward = self.reward_model(next_state)
            rewards.append(reward)
        
        # Add final state (for value bootstrap)
        states.append(self.get_state_feature(h, z))
        
        return states, actions, rewards
