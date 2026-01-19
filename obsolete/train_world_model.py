import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dreamer_arch import WorldModel

# --- Configuration ---
BATCH_SIZE = 16
SEQ_LEN = 20 # How far back the RNN remembers
EPOCHS = 50
LR = 1e-3

# --- 1. Helper: Preprocess Image (64x64 Grayscale) ---
def preprocess(obs):
    # obs is (96, 96, 3)
    # Resize to 64x64
    img = cv2.resize(obs, (64, 64))
    # Convert to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Normalize 0-1 and add channel dim -> (1, 64, 64)
    return (img / 255.0).reshape(1, 64, 64)

# --- 2. Collect Data (The "Babbling" Phase) ---
def collect_data(env_name="CarRacing-v3", episodes=10):
    print(f"Collecting {episodes} episodes of random driving data...")
    env = gym.make(env_name, render_mode="rgb_array")
    buffer_obs = []
    buffer_act = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = preprocess(obs)
        done = False
        
        # Initialize zero action
        action = np.zeros(3) 
        
        while not done:
            buffer_obs.append(obs)
            buffer_act.append(action)
            
            # Sample random action (Babbling)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess(obs)
            done = terminated or truncated
            
    env.close()
    print(f"Collected {len(buffer_obs)} frames.")
    return np.array(buffer_obs), np.array(buffer_act)

# --- 3. Training Loop ---
def train():
    # Setup Device (CPU is fine for this Lite version)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    obs_data, act_data = collect_data(episodes=5)
    
    # Initialize Model
    model = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Prepare Tensors
    # (Total_Frames, 1, 64, 64)
    obs_tensor = torch.tensor(obs_data, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(act_data, dtype=torch.float32).to(device)
    
    print("Starting Training...")
    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0
        
        # We train on sequences
        # Iterate through data in chunks of SEQ_LEN
        for i in range(0, len(obs_data) - SEQ_LEN - 1, SEQ_LEN):
            
            # Get Sequence
            obs_seq = obs_tensor[i : i+SEQ_LEN]     # T x 1 x 64 x 64
            act_seq = act_tensor[i : i+SEQ_LEN]     # T x 3
            
            # Reset RNN state at start of sequence
            rnn_state = torch.zeros(1, 200).to(device)
            
            # Forward Pass through time
            recon_loss = 0
            kl_loss = 0
            
            for t in range(SEQ_LEN):
                # Input: Current Image, Previous Action, Prev RNN State
                # Note: In a full implementation, we process batch dimension. 
                # Here batch=1 for simplicity of understanding the loop.
                img_t = obs_seq[t].unsqueeze(0) # Add batch dim
                act_t = act_seq[t].unsqueeze(0)
                
                recon, rnn_state, mu, logvar = model(img_t, act_t, rnn_state)
                
                # 1. Reconstruction Loss (Did we imagine the right image?)
                recon_loss += F.mse_loss(recon, img_t)
                
                # 2. KL Divergence (Regularize the latent space)
                # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Combined Loss (ELBO-like)
            loss = recon_loss + (0.01 * kl_loss) # Small weight on KL for stability
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / (len(obs_data) / SEQ_LEN)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
        # --- Visualization every 10 epochs ---
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                # Visualize the last frame of the last sequence
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(obs_seq[-1].squeeze().cpu().numpy(), cmap='gray')
                ax[0].set_title("Real Reality")
                ax[1].imshow(recon.squeeze().cpu().numpy(), cmap='gray')
                ax[1].set_title("Robot's Dream")
                plt.show()

if __name__ == "__main__":
    train()