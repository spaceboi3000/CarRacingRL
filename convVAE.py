"""
Convolutional Encoder and Decoder for Dreamer
Based on Ha & Schmidhuber (2018) "World Models"

The encoder compresses 64x64 grayscale images into a 1024-dim embedding.
The decoder reconstructs images from the model state (h, z).
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Visual Encoder: Image -> Embedding
    
    Architecture from Ha & Schmidhuber (2018):
    - 4 convolutional layers with stride 2
    - ReLU activations
    - Output: 1024-dimensional embedding (256 * 2 * 2)
    
    Input: (B, 1, 64, 64) grayscale image
    Output: (B, 1024) embedding
    """
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
        """
        Args:
            x: Image tensor, shape (B, 1, 64, 64)
        Returns:
            Embedding tensor, shape (B, 1024)
        """
        return self.net(x)


class Decoder(nn.Module):
    """
    Visual Decoder: Model State -> Image
    
    Inverse of the encoder, reconstructs 64x64 images from latent states.
    
    Input: (B, state_dim) model state where state_dim = stoch_dim + det_dim
    Output: (B, 1, 64, 64) reconstructed grayscale image
    """
    def __init__(self, state_dim=230, out_channels=1):
        """
        Args:
            state_dim: Dimension of model state (h, z). Default: 200 + 30 = 230
            out_channels: Number of output image channels (1 for grayscale)
        """
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
        """
        Args:
            state: Model state tensor, shape (B, state_dim)
        Returns:
            Reconstructed image, shape (B, 1, 64, 64)
        """
        x = self.fc(state)
        return self.net(x)
