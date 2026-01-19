"""
Dreamer Agent Implementation
Based on Hafner et al. (2020) "Dream to Control: Learning Behaviors by Latent Imagination"

Components:
- Encoder: Image -> Embedding
- Decoder: State -> Image (for reconstruction loss)
- RSSM: Latent dynamics model
- Reward Model: State -> Reward prediction
- Actor: State -> Action distribution
- Critic: State -> Value estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from convVAE import Encoder, Decoder
from RSSM import RSSM


class Actor(nn.Module):
    """
    Action Model (Policy): q_φ(a_t | s_t)
    
    Outputs a tanh-transformed Gaussian distribution over actions.
    Uses reparameterization for gradient flow through sampling.
    
    From Appendix A: "The action model outputs a tanh mean scaled by a factor of 5"
    """
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
        """
        Args:
            state: Model state, shape (B, state_dim)
        Returns:
            mean: Action mean (pre-tanh), shape (B, action_dim)
            std: Action std, shape (B, action_dim)
        """
        features = self.net(state)
        
        # Mean: scaled tanh allows the agent to saturate actions
        mean = self.mean_head(features)
        
        # Std: softplus ensures positive, with minimum for numerical stability
        std_raw = self.std_head(features)
        std = F.softplus(std_raw + self.init_std) + self.min_std
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            state: Model state, shape (B, state_dim)
            deterministic: If True, return mean action (no sampling)
        Returns:
            action: Sampled action in [-1, 1], shape (B, action_dim)
            log_prob: Log probability of action (for entropy bonus if needed)
        """
        mean, std = self.forward(state)
        
        if deterministic:
            # Deterministic action (mean of the tanh-squashed distribution)
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Sample from Normal, then squash with tanh
            dist = D.Normal(mean, std)
            sample = dist.rsample()  # Reparameterized sample
            action = torch.tanh(sample)
            
            # Log prob with tanh correction (for SAC-style entropy, if needed)
            # log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """
    Value Model: v_ψ(s_t)
    
    Estimates expected sum of future rewards from current state.
    Used to compute V_λ targets for actor optimization.
    """
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
        """
        Args:
            state: Model state, shape (B, state_dim)
        Returns:
            value: Estimated value, shape (B, 1)
        """
        return self.net(state)


class RewardModel(nn.Module):
    """
    Reward Predictor: q(r_t | s_t)
    
    Predicts immediate reward from model state.
    """
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
        """
        Args:
            state: Model state, shape (B, state_dim)
        Returns:
            reward: Predicted reward, shape (B, 1)
        """
        return self.net(state)


class Dreamer(nn.Module):
    """
    Complete Dreamer Agent
    
    Combines all components and provides methods for:
    - Environment interaction (observe, get_action)
    - Imagination (dream forward using prior)
    - Training (world model, actor, critic losses)
    """
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
        """
        Get initial (zero) states for h and z.
        Call this at the start of each episode.
        """
        return self.rssm.initial_state(batch_size, self.device)

    def encode(self, obs):
        """Encode observation to embedding."""
        return self.encoder(obs)
    
    def decode(self, state):
        """Decode state to reconstructed observation."""
        return self.decoder(state)

    def observe(self, obs, prev_action, prev_h, prev_z):
        """
        Process a real observation (posterior update).
        Used during environment interaction and training on real data.
        
        Args:
            obs: Observation tensor, shape (B, 1, 64, 64)
            prev_action: Previous action, shape (B, action_dim)
            prev_h: Previous deterministic state, shape (B, det_dim)
            prev_z: Previous stochastic state, shape (B, stoch_dim)
            
        Returns:
            h: New deterministic state
            z: New stochastic state (from posterior)
            prior_dist: Prior distribution (for KL loss)
            posterior_dist: Posterior distribution (for KL loss)
        """
        embed = self.encoder(obs)
        h, z, prior_dist, posterior_dist = self.rssm.observe(
            prev_z, prev_action, prev_h, embed
        )
        return h, z, prior_dist, posterior_dist

    def imagine(self, prev_action, prev_h, prev_z):
        """
        Imagine next state without observation (prior transition).
        Used during dreaming for behavior learning.
        
        Args:
            prev_action: Action to take, shape (B, action_dim)
            prev_h: Previous deterministic state, shape (B, det_dim)
            prev_z: Previous stochastic state, shape (B, stoch_dim)
            
        Returns:
            h: New deterministic state
            z: New stochastic state (from prior)
            prior_dist: Prior distribution
        """
        return self.rssm.imagine(prev_z, prev_action, prev_h)

    def get_state_feature(self, h, z):
        """Combine h and z into a single state feature."""
        return torch.cat([h, z], dim=-1)

    @torch.no_grad()
    def get_action(self, obs, prev_action, prev_h, prev_z, deterministic=False):
        """
        Get action for environment interaction.
        
        This method:
        1. Encodes the observation
        2. Updates the belief state using the posterior
        3. Samples an action from the policy
        
        Args:
            obs: Numpy array or tensor, shape (1, 64, 64) or (B, 1, 64, 64)
            prev_action: Previous action tensor, shape (B, action_dim)
            prev_h: Previous h state, shape (B, det_dim)
            prev_z: Previous z state, shape (B, stoch_dim)
            deterministic: If True, use mean action (no noise)
            
        Returns:
            action: Action tensor, shape (B, action_dim)
            h: Updated h state
            z: Updated z state
        """
        # Ensure observation is a tensor
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
        """
        Compute world model loss on a batch of sequences.
        
        Loss = reconstruction_loss + reward_loss + kl_scale * KL_loss
        
        Args:
            obs_seq: Observations, shape (B, T, 1, H, W)
            action_seq: Actions, shape (B, T, action_dim)
            reward_seq: Rewards, shape (B, T, 1)
            free_nats: KL divergence threshold (clip KL below this)
            kl_scale: Weight for KL loss term
            
        Returns:
            total_loss: Combined loss
            metrics: Dict with individual loss components
        """
        B, T = obs_seq.shape[:2]
        device = obs_seq.device
        
        # Initialize states
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
            
            # Get state feature
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
        """
        Imagine a trajectory forward using the prior (dreaming).
        Used for behavior learning.
        
        Args:
            start_h: Starting deterministic states, shape (B, det_dim)
            start_z: Starting stochastic states, shape (B, stoch_dim)
            horizon: Number of steps to imagine
            
        Returns:
            states: List of state features, length H+1 (includes start state)
            actions: List of actions taken, length H
            rewards: List of predicted rewards, length H
        """
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
