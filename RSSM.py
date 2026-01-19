import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class RSSM(nn.Module):
    def __init__(self, latent_dim=30, rnn_dim=200, action_dim=3, embed_dim=1024):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, rnn_dim)
        
       
        self.prior_net = nn.Sequential(
            nn.Linear(rnn_dim, 200), nn.ELU(),
            nn.Linear(200, 2 * latent_dim)
        )
        
        
        self.posterior_net = nn.Sequential(
            nn.Linear(rnn_dim + embed_dim, 200), nn.ELU(),
            nn.Linear(200, 2 * latent_dim)
        )

    def transition(self, prev_z, prev_action, prev_rnn):
        """Prior (Dreaming): Predicts z from h"""
        # 1. Deterministic Update (The RNN)
        rnn_input = torch.cat([prev_z, prev_action], dim=1)
        rnn_state = self.gru(rnn_input, prev_rnn)
        
        # 2. Stochastic Update (The Latent Z) - USE _net
        mean, std = self.prior_net(rnn_state).chunk(2, dim=1)
            
        # 3. Sample
        std = F.softplus(std) + 0.1
        dist = D.Normal(mean, std)
        z_state = dist.rsample()
        
        return rnn_state, z_state, dist

    def posterior(self, prev_z, prev_action, prev_rnn, obs_embed):
        """Posterior (Reality): Infers z from h + embedding"""
        # 1. Deterministic Update (The RNN)
        rnn_input = torch.cat([prev_z, prev_action], dim=1)
        rnn_state = self.gru(rnn_input, prev_rnn)
        
        # 2. Stochastic Update (The Latent Z) - USE _net
        post_in = torch.cat([rnn_state, obs_embed], dim=1)
        mean, std = self.posterior_net(post_in).chunk(2, dim=1)

        # 3. Sample
        std = F.softplus(std) + 0.1
        dist = D.Normal(mean, std)
        z_state = dist.rsample()
        
        return rnn_state, z_state, dist