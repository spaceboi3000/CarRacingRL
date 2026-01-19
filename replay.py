"""
Experience Replay Buffer for Dreamer

Stores transitions and samples sequences for training.
Handles episode boundaries to avoid sampling across episodes.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer that stores full episodes and samples valid sequences.
    
    Key features:
    - Stores observations as uint8 to save memory (4x reduction)
    - Tracks episode boundaries to avoid sampling across episodes
    - Efficient vectorized sequence sampling
    """
    
    def __init__(self, capacity=100000, obs_shape=(64, 64), action_dim=3):
        """
        Args:
            capacity: Maximum number of transitions to store
            obs_shape: Shape of observations (H, W) for grayscale
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Current position and size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate storage
        # Store images as uint8 (0-255) to save memory
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        # Track episode start indices for valid sequence sampling
        self.episode_starts = [0]
        
    def add(self, obs, action, reward, done):
        """
        Add a single transition to the buffer.
        
        Args:
            obs: Observation, shape (1, H, W) or (H, W), values in [0, 1]
            action: Action, shape (action_dim,)
            reward: Scalar reward
            done: Boolean indicating episode termination
        """
        # Convert observation to uint8 if needed
        if isinstance(obs, np.ndarray):
            if obs.dtype == np.float32 or obs.dtype == np.float64:
                obs = (obs * 255).astype(np.uint8)
            if obs.ndim == 3:
                obs = obs.squeeze(0)  # Remove channel dimension for storage
        
        # Store transition
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track episode boundaries
        if done:
            self.episode_starts.append(self.ptr)
            # Clean up old episode starts if buffer has wrapped
            if self.size == self.capacity:
                self._cleanup_episode_starts()
    
    def _cleanup_episode_starts(self):
        """Remove episode starts that have been overwritten."""
        # Keep only episode starts that are still valid
        valid_starts = []
        for start in self.episode_starts:
            # An episode start is valid if it's ahead of the current pointer
            # (accounting for circular buffer wrap-around)
            if start >= self.ptr or start == 0:
                valid_starts.append(start)
        self.episode_starts = valid_starts if valid_starts else [0]
    
    def _get_valid_sequence_starts(self, seq_len):
        """
        Get indices where we can start a valid sequence.
        A valid sequence doesn't cross episode boundaries.
        
        Args:
            seq_len: Length of sequence to sample
            
        Returns:
            Array of valid starting indices
        """
        valid_starts = []
        
        # Sort episode starts
        sorted_starts = sorted(self.episode_starts)
        
        # Add current size as a boundary
        boundaries = sorted_starts + [self.size]
        
        # Find valid ranges between episode boundaries
        for i in range(len(boundaries) - 1):
            ep_start = boundaries[i]
            ep_end = boundaries[i + 1]
            ep_len = ep_end - ep_start
            
            # Can sample sequences of length seq_len from this episode
            if ep_len >= seq_len:
                # Valid starts are from ep_start to (ep_end - seq_len)
                for idx in range(ep_start, ep_end - seq_len + 1):
                    valid_starts.append(idx)
        
        return np.array(valid_starts)
    
    def sample_sequence(self, batch_size, seq_len, device):
        """
        Sample random sequences for training.
        Ensures sequences don't cross episode boundaries.
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            device: Torch device for output tensors
            
        Returns:
            obs: Observations, shape (B, T, 1, H, W)
            actions: Actions, shape (B, T, action_dim)
            rewards: Rewards, shape (B, T, 1)
        """
        # Get valid starting indices
        valid_starts = self._get_valid_sequence_starts(seq_len)
        
        if len(valid_starts) < batch_size:
            raise ValueError(
                f"Not enough valid sequences. Need {batch_size}, have {len(valid_starts)}. "
                f"Buffer size: {self.size}, seq_len: {seq_len}"
            )
        
        # Sample random starting indices
        start_indices = np.random.choice(valid_starts, size=batch_size, replace=False)
        
        # Build index array for all timesteps
        # Shape: (batch_size, seq_len)
        time_offsets = np.arange(seq_len)
        all_indices = start_indices[:, None] + time_offsets[None, :]
        
        # Gather data
        obs = self.obs[all_indices]  # (B, T, H, W)
        actions = self.actions[all_indices]  # (B, T, action_dim)
        rewards = self.rewards[all_indices]  # (B, T, 1)
        
        # Convert to tensors
        # Normalize observations to [0, 1] and add channel dimension
        obs = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
        obs = obs.unsqueeze(2)  # (B, T, 1, H, W)
        
        actions = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        return obs, actions, rewards
    
    def sample_sequence_simple(self, batch_size, seq_len, device):
        """
        Simple sequence sampling without episode boundary checking.
        Faster but may sample across episode boundaries.
        Use this only if you don't care about episode boundaries.
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of each sequence
            device: Torch device for output tensors
            
        Returns:
            obs: Observations, shape (B, T, 1, H, W)
            actions: Actions, shape (B, T, action_dim)
            rewards: Rewards, shape (B, T, 1)
        """
        # Maximum valid start index
        max_start = self.size - seq_len
        if max_start <= 0:
            raise ValueError(f"Not enough data: need {seq_len}, have {self.size}")
        
        # Sample random starts
        start_indices = np.random.randint(0, max_start, size=batch_size)
        
        # Build full index array
        time_offsets = np.arange(seq_len)
        all_indices = start_indices[:, None] + time_offsets[None, :]
        
        # Gather and convert
        obs = torch.tensor(self.obs[all_indices], dtype=torch.float32, device=device) / 255.0
        obs = obs.unsqueeze(2)
        actions = torch.tensor(self.actions[all_indices], dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards[all_indices], dtype=torch.float32, device=device)
        
        return obs, actions, rewards
    
    def __len__(self):
        return self.size


class EpisodeReplayBuffer:
    """
    Alternative replay buffer that stores complete episodes.
    More memory efficient for variable-length episodes.
    """
    
    def __init__(self, max_episodes=1000, obs_shape=(64, 64), action_dim=3):
        self.max_episodes = max_episodes
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        self.episodes = []
        self.total_steps = 0
        
    def add_episode(self, observations, actions, rewards):
        """
        Add a complete episode to the buffer.
        
        Args:
            observations: List or array of observations
            actions: List or array of actions
            rewards: List or array of rewards
        """
        episode = {
            'obs': np.array(observations, dtype=np.uint8),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'length': len(observations)
        }
        
        self.episodes.append(episode)
        self.total_steps += episode['length']
        
        # Remove old episodes if over capacity
        while len(self.episodes) > self.max_episodes:
            removed = self.episodes.pop(0)
            self.total_steps -= removed['length']
    
    def sample_sequence(self, batch_size, seq_len, device):
        """
        Sample sequences from stored episodes.
        """
        # Filter episodes that are long enough
        valid_episodes = [ep for ep in self.episodes if ep['length'] >= seq_len]
        
        if not valid_episodes:
            raise ValueError(f"No episodes with length >= {seq_len}")
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        
        for _ in range(batch_size):
            # Sample random episode
            ep = valid_episodes[np.random.randint(len(valid_episodes))]
            
            # Sample random start within episode
            start = np.random.randint(0, ep['length'] - seq_len + 1)
            
            batch_obs.append(ep['obs'][start:start+seq_len])
            batch_actions.append(ep['actions'][start:start+seq_len])
            batch_rewards.append(ep['rewards'][start:start+seq_len])
        
        # Stack and convert
        obs = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device) / 255.0
        obs = obs.unsqueeze(2)  # Add channel dim
        actions = torch.tensor(np.stack(batch_actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.stack(batch_rewards), dtype=torch.float32, device=device)
        
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        
        return obs, actions, rewards
    
    def __len__(self):
        return self.total_steps
