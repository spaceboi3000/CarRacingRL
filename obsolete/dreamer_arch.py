import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# --- 1. The Encoder (Visual Cortex) ---
# Takes a 64x64 image -> Outputs a flat vector
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Conv2d(in_channels, out_channels, kernel, stride)
            nn.Conv2d(1, 32, 4, 2), nn.ReLU(),  # -> 31x31
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(), # -> 14x14
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),# -> 6x6
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),# -> 2x2
            nn.Flatten() # 256*2*2 = 1024 features
        )

    def forward(self, x):
        return self.net(x)

# --- 2. The Decoder (Dream Generator) ---
# Takes a feature vector -> Reconstructs 64x64 image
class Decoder(nn.Module):
    def __init__(self, latent_dim=128 + 200): # Input is (Deterministic + Stochastic)
        super().__init__()
        self.fc = nn.Linear(latent_dim, 1024)
        self.net = nn.Sequential(
            nn.Unflatten(1, (1024, 1, 1)), # Reshape to 1x1 image with 1024 channels
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 6, 2), nn.Sigmoid() # Output 0-1 (Pixel intensity)
        )

    def forward(self, x):
        x = self.fc(x)
        return self.net(x)