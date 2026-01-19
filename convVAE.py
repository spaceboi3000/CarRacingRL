# ConvVAE model
# @article{Ha2018WorldModels,
#   author = {Ha, D. and Schmidhuber, J.},
#   title  = {World Models},
#   eprint = {arXiv:1803.10122},
#   doi    = {10.5281/zenodo.1207631},
#   url    = {https://worldmodels.github.io},
#   year   = {2018}
# }

import torch.nn as nn


# --- 1. The Encoder (Visual Cortex) ---
# Architecture from Ha & Schmidhuber (2018)
# Matches the TF code: 4 layers, specific kernels to get 2x2 output
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (1, 64, 64) -> Output: (32, 31, 31)
            nn.Conv2d(1, 32, kernel_size=4, stride=2), nn.ReLU(),
            
            # Input: (32, 31, 31) -> Output: (64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            
            # Input: (64, 14, 14) -> Output: (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            
            # Input: (128, 6, 6) -> Output: (256, 2, 2)
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(),
            
            nn.Flatten() # 256 * 2 * 2 = 1024
        )

    def forward(self, x):
        return self.net(x)

# --- 2. The Decoder (Dream Generator) ---
# Architecture from Ha & Schmidhuber (2018)
# Inverse of the Encoder to reconstruct 64x64
class Decoder(nn.Module):
    def __init__(self, latent_dim=128 + 200): 
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        self.net = nn.Sequential(
            nn.Unflatten(1, (1024, 1, 1)), # Reshape to 1x1 pixel with 1024 channels
            
            # ConvTranspose2d(in, out, kernel, stride)
            # 1x1 -> 5x5
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2), nn.ReLU(),
            
            # 5x5 -> 13x13
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(),
            
            # 13x13 -> 30x30
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ReLU(),
            
            # 30x30 -> 64x64 (Output)
            nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return self.net(x)
