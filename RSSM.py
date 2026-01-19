"""
Recurrent State-Space Model (RSSM) for Dreamer
Based on Hafner et al. 2018 (PlaNet) and Hafner et al. 2020 (Dreamer)

The RSSM maintains two types of state:
- Deterministic state h_t (RNN hidden state)
- Stochastic state z_t (latent variable)

Combined model state: s_t = concat(h_t, z_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RSSM(nn.Module):
    def __init__(self, stoch_dim=30, det_dim=200, action_dim=3, embed_dim=1024, hidden_dim=200):
        """
        Args:
            stoch_dim: Dimension of stochastic latent z
            det_dim: Dimension of deterministic RNN hidden state h
            action_dim: Dimension of action space
            embed_dim: Dimension of image embedding from encoder
            hidden_dim: Hidden layer size for prior/posterior networks
        """
        super().__init__()
        self.stoch_dim = stoch_dim
        self.det_dim = det_dim
        
        # GRU for deterministic state transition
        # Input: concat(z_{t-1}, a_{t-1})
        # Hidden: h_{t-1} -> h_t
        self.gru = nn.GRUCell(stoch_dim + action_dim, det_dim)
        
        # Prior network: p(z_t | h_t)
        # Predicts z from deterministic state only (for imagination/dreaming)
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # Mean and std
        )
        
        # Posterior network: q(z_t | h_t, o_t)  
        # Infers z from deterministic state AND observation embedding (for training)
        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # Mean and std
        )

    def initial_state(self, batch_size, device):
        """Return initial zero states for h and z."""
        h = torch.zeros(batch_size, self.det_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _get_dist(self, mean, std_raw):
        """Convert raw network output to distribution."""
        std = F.softplus(std_raw) + 0.1  # Ensure positive std with minimum
        return D.Normal(mean, std)

    def observe(self, prev_z, prev_action, prev_h, obs_embed):
        """
        Posterior update: Given a real observation, infer the latent state.
        Used during training when we have access to real observations.
        
        This method computes BOTH prior and posterior from the same deterministic
        state to ensure consistency (important for KL divergence computation).
        
        Args:
            prev_z: Previous stochastic state z_{t-1}, shape (B, stoch_dim)
            prev_action: Previous action a_{t-1}, shape (B, action_dim)
            prev_h: Previous deterministic state h_{t-1}, shape (B, det_dim)
            obs_embed: Current observation embedding e_t, shape (B, embed_dim)
            
        Returns:
            h: New deterministic state h_t
            z: New stochastic state z_t (sampled from posterior)
            prior_dist: Prior distribution p(z_t | h_t)
            posterior_dist: Posterior distribution q(z_t | h_t, e_t)
        """
        # 1. Deterministic state update (same for both prior and posterior)
        gru_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        # 2. Prior distribution p(z_t | h_t) - what the model predicts without seeing observation
        prior_out = self.prior_net(h)
        prior_mean, prior_std_raw = prior_out.chunk(2, dim=-1)
        prior_dist = self._get_dist(prior_mean, prior_std_raw)
        
        # 3. Posterior distribution q(z_t | h_t, e_t) - refined with observation
        posterior_input = torch.cat([h, obs_embed], dim=-1)
        posterior_out = self.posterior_net(posterior_input)
        posterior_mean, posterior_std_raw = posterior_out.chunk(2, dim=-1)
        posterior_dist = self._get_dist(posterior_mean, posterior_std_raw)
        
        # 4. Sample z from posterior (use rsample for gradient flow)
        z = posterior_dist.rsample()
        
        return h, z, prior_dist, posterior_dist

    def imagine(self, prev_z, prev_action, prev_h):
        """
        Prior transition: Predict next state without observation.
        Used during imagination/dreaming when we don't have real observations.
        
        Args:
            prev_z: Previous stochastic state z_{t-1}, shape (B, stoch_dim)
            prev_action: Action a_{t-1}, shape (B, action_dim)
            prev_h: Previous deterministic state h_{t-1}, shape (B, det_dim)
            
        Returns:
            h: New deterministic state h_t
            z: New stochastic state z_t (sampled from prior)
            prior_dist: Prior distribution p(z_t | h_t)
        """
        # 1. Deterministic state update
        gru_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        # 2. Prior distribution
        prior_out = self.prior_net(h)
        prior_mean, prior_std_raw = prior_out.chunk(2, dim=-1)
        prior_dist = self._get_dist(prior_mean, prior_std_raw)
        
        # 3. Sample z from prior
        z = prior_dist.rsample()
        
        return h, z, prior_dist

    def get_feature(self, h, z):
        """Concatenate deterministic and stochastic states into model feature."""
        return torch.cat([h, z], dim=-1)
