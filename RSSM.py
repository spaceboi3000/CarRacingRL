import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class RSSM(nn.Module):
    def __init__(self, stoch_dim=30, det_dim=200, action_dim=3, embed_dim=1024, hidden_dim=200):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.det_dim = det_dim
        self.gru = nn.GRUCell(stoch_dim + action_dim, det_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # Mean and std
        )
        

        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)  # Mean and std
        )

    def initial_state(self, batch_size, device):
        
        h = torch.zeros(batch_size, self.det_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _get_dist(self, mean, std_raw): # network output to distribution
        std = F.softplus(std_raw) + 0.1  # force positive std with minimum
        return D.Normal(mean, std)

    def observe(self, prev_z, prev_action, prev_h, obs_embed):

        # state update
        gru_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        # Prior p(z_t | h_t) 
        prior_out = self.prior_net(h)
        prior_mean, prior_std_raw = prior_out.chunk(2, dim=-1)
        prior_dist = self._get_dist(prior_mean, prior_std_raw)
        
        # Prior q(z_t | h_t, e_t)
        posterior_input = torch.cat([h, obs_embed], dim=-1)
        posterior_out = self.posterior_net(posterior_input)
        posterior_mean, posterior_std_raw = posterior_out.chunk(2, dim=-1)
        posterior_dist = self._get_dist(posterior_mean, posterior_std_raw)
        
        # sample z from posterior (use rsample for gradient flow)
        z = posterior_dist.rsample()
        
        return h, z, prior_dist, posterior_dist

    def imagine(self, prev_z, prev_action, prev_h):
        # deterministic state update
        gru_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(gru_input, prev_h)
        
        #  Prior distribution
        prior_out = self.prior_net(h)
        prior_mean, prior_std_raw = prior_out.chunk(2, dim=-1)
        prior_dist = self._get_dist(prior_mean, prior_std_raw)
        
        # Sample z from prior
        z = prior_dist.rsample()
        
        return h, z, prior_dist

    def get_feature(self, h, z):
        return torch.cat([h, z], dim=-1)
