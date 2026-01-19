import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import cv2
import matplotlib.pyplot as plt

from convVAE import Encoder, Decoder
from RSSM import RSSM

class Actor(nn.Module):
    """
    Custom Actor network that enforces positive standard deviation.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 2 * action_dim) # Output: [Mean, Std_Raw]
        )

    def forward(self, x):
        x = self.net(x)
        mean, std_raw = x.chunk(2, dim=1)
        
        # FIX: Force std to be positive using softplus + epsilon
        std = F.softplus(std_raw) + 0.1 
        
        # Return concatenated so .chunk(2,1) still works in train.py
        return torch.cat([mean, std], dim=1)


class Critic(nn.Module):
    """
    Value Network (Critic) v_psi(s)
    Predicts the expected sum of future rewards (V_lambda) from the latent state.
    """
    def __init__(self, input_dim, hidden_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1) # Outputs a scalar Value
        )

    def forward(self, x):
        return self.net(x)
    



class Dreamer(nn.Module):
    def __init__(self, action_dim=3, stochastic_dim=30, deterministic_dim=200, 
                 hidden_dim=300, device="cpu"):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        # Store dimensions for easy access
        self.stoch_dim = stochastic_dim      # z dimension
        self.det_dim = deterministic_dim     # h (RNN hidden) dimension
        self.state_dim = stochastic_dim + deterministic_dim  # Combined feature dimension
        self.hidden_dim = hidden_dim
        
        # 1. The Components
        self.encoder = Encoder().to(device)
        self.decoder = Decoder(latent_dim=self.state_dim).to(device)
        self.rssm = RSSM(latent_dim=stochastic_dim, rnn_dim = deterministic_dim,
                         action_dim=action_dim).to(device)
        
        # 2. Reward Predictor (q(r|s))
        self.reward_model = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        # 3. The Actor (Policy q(a|s))
        self.actor = Actor(input_dim=self.state_dim, 
                          action_dim=action_dim, 
                          hidden_dim=hidden_dim).to(device)
        
        # 4. The Critic (Value v(s)) 
        self.critic = Critic(input_dim=self.state_dim, 
                            hidden_dim=hidden_dim).to(device)
        
        # Optimizers
        self.model_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.rssm.parameters()) + 
            list(self.reward_model.parameters()), 
            lr=6e-4
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

    def get_initial_state(self, batch_size=1):
        """
        Returns initial (zero) states for RNN and stochastic latent.
        Useful for resetting between episodes.
        """
        rnn = torch.zeros(batch_size, self.det_dim).to(self.device)
        z = torch.zeros(batch_size, self.stoch_dim).to(self.device)
        return rnn, z

    def imagine(self, prev_action, prev_rnn, prev_z):
        """Helper for dreaming without images (Prior)"""
        return self.rssm.transition(prev_z, prev_action, prev_rnn)

    def observe(self, obs, prev_action, prev_rnn, prev_z):
        """Helper for seeing real images (Posterior)"""
        obs = obs.to(self.device)
        embed = self.encoder(obs)
        return self.rssm.posterior(prev_z, prev_action, prev_rnn, embed)
        
    def get_action(self, obs, prev_action, prev_rnn, prev_z, deterministic=False):
        """
        Interaction Helper:
        1. Encodes image
        2. Updates RNN (Posterior)
        3. Selects Action
        """
        with torch.no_grad():
            # 1. Encode & Observe
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            rnn, z, _ = self.observe(obs_tensor, prev_action, prev_rnn, prev_z)
            
            # 2. Select Action
            feat = torch.cat([rnn, z], dim=1)
            
            # Since self.actor now handles the softplus internally,
            # we just retrieve the already-positive std.
            mean, std = self.actor(feat).chunk(2, dim=1)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                # Create Normal distribution
                dist = D.Normal(mean, std)
                # Sample and THEN squash with Tanh (Squashed Normal)
                action = torch.tanh(dist.sample())
                
        return action, rnn, z
