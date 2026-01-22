import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # (1, 64, 64) -> (32, 31, 31)
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # (32, 31, 31) -> (64, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # (64, 14, 14) -> (128, 6, 6)
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # (128, 6, 6) -> (256, 2, 2)
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # (256, 2, 2) -> (1024,)
            nn.Flatten()
        )
        
        # Output dimension
        self.embed_dim = 256 * 2 * 2  # = 1024

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, state_dim=230, out_channels=1):
        super().__init__()
        
        # Project state to spatial format
        self.fc = nn.Linear(state_dim, 1024)
        
        # Transposed convolutions to upsample
        self.net = nn.Sequential(
            # (1024,) -> (1024, 1, 1)
            nn.Unflatten(1, (1024, 1, 1)),
            
            # (1024, 1, 1) -> (128, 5, 5)
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            
            # (128, 5, 5) -> (64, 13, 13)
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            
            # (64, 13, 13) -> (32, 30, 30)
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            
            # (32, 30, 30) -> (1, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, state):
        x = self.fc(state)
        return self.net(x)
