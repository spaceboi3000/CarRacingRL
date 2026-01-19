import gymnasium as gym
import torch
import cv2
import numpy as np
from dreamer import Dreamer

# --- CONFIGURATION ---
ENV_NAME = "CarRacing-v3"
MODEL_PATH = "dreamer.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs):
    """Resize to 64x64, Grayscale, Normalize to 0-1"""
    img = cv2.resize(obs, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).reshape(1, 64, 64)

def main():
    print(f"Loading Dreamer from {MODEL_PATH}...")
    
    # 1. Initialize Environment with 'human' render mode (Pop-up window)
    env = gym.make(ENV_NAME, render_mode="human")
    
    # 2. Initialize Agent and Load Weights
    agent = Dreamer(action_dim=3, device=DEVICE)
    try:
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: 'dreamer.pth' not found. Train the model first!")
        return

    # 3. Drive Loop
    while True:
        obs, _ = env.reset()
        done = False
        
        # Reset Internal States
        prev_act = torch.zeros(1, 3).to(DEVICE)
        prev_rnn = torch.zeros(1, 200).to(DEVICE)
        prev_z = torch.zeros(1, 30).to(DEVICE)
        
        total_reward = 0
        
        while not done:
            # Prepare observation
            obs_p = preprocess(obs)
            
            # Get Action (Deterministic = Best Guess, no random noise)
            action_tensor, rnn, z = agent.get_action(
                obs_p, prev_act, prev_rnn, prev_z, deterministic=True
            )
            action = action_tensor.cpu().numpy()[0]
            
            # Clip actions
            action[1] = np.clip(action[1], 0, 1) # Gas
            action[2] = np.clip(action[2], 0, 1) # Brake
            
            # Step Env
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            
            # Update States
            prev_act = action_tensor
            prev_rnn = rnn
            prev_z = z
            done = term or trunc
            
            # Optional: Press 'q' to quit early
            # (Gym usually handles closing the window, but this is a failsafe)
            
        print(f"Episode Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()