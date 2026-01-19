import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=10000, obs_shape=(64, 64), action_dim=3):
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        
        # Pre-allocate huge arrays (Fast RAM access)
        # We store images as uint8 (0-255) to save 4x memory, convert to float later
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool_)

    def add(self, obs, action, reward, done):
        """Add a single step to the buffer."""
        # Convert 0.0-1.0 float image back to 0-255 uint8 to save space
        if obs.dtype == np.float32:
            obs = (obs * 255).astype(np.uint8)
            
        self.obs[self.ptr] = obs
        self.act[self.ptr] = action
        self.rew[self.ptr] = reward
        self.done[self.ptr] = done
        
        # Move pointer, wrap around if full (Circular Buffer)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0: self.full = True

    def sample_sequence(self, batch_size, seq_len, device):
        """
        Samples random sub-sequences for training.
        Returns tensors: (Batch, Seq_Len, ...)
        """
        # 1. Determine valid range
        limit = self.capacity if self.full else self.ptr
        # We need to ensure we don't pick an index so close to the end 
        # that the sequence runs off the edge of the array.
        valid_max = limit - seq_len
        if valid_max <= 0:
            raise ValueError(f"Not enough data: need {seq_len}, have {limit}")
        # 2. Randomly pick start indices
        # Shape: (Batch_Size, 1)
        start_idxs = np.random.randint(0, valid_max, size=(batch_size, 1))
        
        # 3. Vectorized Sequence Expansion (The Trick)
        # We create a range [0, 1, ... seq_len] and add it to start_idxs
        # Result: A matrix of all the indices we need.
        time_idxs = np.arange(seq_len)
        global_idxs = start_idxs + time_idxs # Broadcasting happens here
        
        # 4. Retrieve Data instantly
        # Normalize images back to 0.0-1.0 here
        obs = torch.tensor(self.obs[global_idxs], dtype=torch.float32, device=device) / 255.0
        # Add channel dimension: (Batch, Seq, H, W) -> (Batch, Seq, 1, H, W)
        obs = obs.unsqueeze(2) 
        
        act = torch.tensor(self.act[global_idxs], dtype=torch.float32, device=device)
        rew = torch.tensor(self.rew[global_idxs], dtype=torch.float32, device=device)
        
        return obs, act, rew