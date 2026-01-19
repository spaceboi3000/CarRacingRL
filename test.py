"""
Dreamer Evaluation Script

Loads a trained Dreamer agent and evaluates it in the CarRacing environment.
"""

import gymnasium as gym
import torch
import cv2
import numpy as np
import argparse
from dreamer import Dreamer


# Configuration
ENV_NAME = "CarRacing-v3"
DEFAULT_MODEL_PATH = "dreamer.pth"

# Must match training configuration
STOCH_DIM = 30
DET_DIM = 200
HIDDEN_DIM = 300
EMBED_DIM = 1024


def preprocess(obs):
    """Preprocess observation: resize, grayscale, normalize."""
    img = cv2.resize(obs, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).astype(np.float32).reshape(1, 64, 64)


def process_action_for_env(action):
    """Convert tanh action to CarRacing action space."""
    processed = action.copy()
    processed[1] = np.clip((action[1] + 1) / 2, 0, 1)  # Gas: [-1,1] -> [0,1]
    processed[2] = np.clip((action[2] + 1) / 2, 0, 1)  # Brake: [-1,1] -> [0,1]
    return processed


def evaluate(model_path, num_episodes=5, render=True, deterministic=True):
    """
    Evaluate a trained Dreamer agent.
    
    Args:
        model_path: Path to saved model weights
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Use mean action (True) or sample (False)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize environment
    render_mode = "human" if render else "rgb_array"
    env = gym.make(ENV_NAME, render_mode=render_mode)
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = Dreamer(
        action_dim=action_dim,
        stoch_dim=STOCH_DIM,
        det_dim=DET_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        device=device
    )
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        print(f"Loaded model from '{model_path}'")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train.py")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    agent.eval()
    
    # Evaluation loop
    episode_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        # Reset agent state
        prev_action = torch.zeros(1, action_dim, device=device)
        prev_h, prev_z = agent.get_initial_state(batch_size=1)
        
        total_reward = 0
        step_count = 0
        
        while not done:
            # Preprocess observation
            obs_processed = preprocess(obs)
            
            # Get action from agent
            with torch.no_grad():
                action_tensor, prev_h, prev_z = agent.get_action(
                    obs_processed, prev_action, prev_h, prev_z,
                    deterministic=deterministic
                )
            
            action = action_tensor.cpu().numpy()[0]
            
            # Process for environment
            env_action = process_action_for_env(action)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Update state
            prev_action = action_tensor
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Steps = {step_count}")
    
    # Summary statistics
    print(f"\n{'='*40}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*40}")
    print(f"Mean reward:   {np.mean(episode_rewards):.2f}")
    print(f"Std reward:    {np.std(episode_rewards):.2f}")
    print(f"Min reward:    {np.min(episode_rewards):.2f}")
    print(f"Max reward:    {np.max(episode_rewards):.2f}")
    
    env.close()
    return episode_rewards


def visualize_imagination(model_path, num_steps=50):
    """
    Visualize what the agent imagines vs reality.
    
    This shows:
    - Real observations
    - Reconstructed observations from the world model
    - Imagined future observations (predictions)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    action_dim = env.action_space.shape[0]
    
    agent = Dreamer(
        action_dim=action_dim,
        stoch_dim=STOCH_DIM,
        det_dim=DET_DIM,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBED_DIM,
        device=device
    )
    
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
    except:
        print("Could not load model. Using random weights.")
    
    agent.eval()
    
    obs, _ = env.reset()
    prev_action = torch.zeros(1, action_dim, device=device)
    prev_h, prev_z = agent.get_initial_state(batch_size=1)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for step in range(num_steps):
        obs_processed = preprocess(obs)
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Get action and update state
            action_tensor, prev_h, prev_z = agent.get_action(
                obs_processed, prev_action, prev_h, prev_z, deterministic=True
            )
            
            # Reconstruct current observation
            state = agent.get_state_feature(prev_h, prev_z)
            recon = agent.decoder(state)
            
            # Imagine future
            imagine_h, imagine_z = prev_h.clone(), prev_z.clone()
            imagined_frames = []
            for _ in range(5):
                imagine_state = agent.get_state_feature(imagine_h, imagine_z)
                mean, std = agent.actor(imagine_state)
                imagine_action = torch.tanh(mean)
                imagine_h, imagine_z, _ = agent.rssm.imagine(imagine_z, imagine_action, imagine_h)
                imagine_state = agent.get_state_feature(imagine_h, imagine_z)
                imagined_frames.append(agent.decoder(imagine_state))
        
        action = action_tensor.cpu().numpy()[0]
        env_action = process_action_for_env(action)
        obs, _, terminated, truncated, _ = env.step(env_action)
        prev_action = action_tensor
        
        if step % 10 == 0:
            # Visualize
            for ax in axes.flat:
                ax.clear()
            
            # Row 1: Real observation and reconstruction
            axes[0, 0].imshow(obs)
            axes[0, 0].set_title("Real (RGB)")
            axes[0, 1].imshow(obs_processed[0], cmap='gray')
            axes[0, 1].set_title("Real (Processed)")
            axes[0, 2].imshow(recon[0, 0].cpu().numpy(), cmap='gray')
            axes[0, 2].set_title("Reconstructed")
            axes[0, 3].axis('off')
            axes[0, 4].axis('off')
            
            # Row 2: Imagined future
            for i, frame in enumerate(imagined_frames):
                axes[1, i].imshow(frame[0, 0].cpu().numpy(), cmap='gray')
                axes[1, i].set_title(f"Imagined t+{i+1}")
            
            plt.suptitle(f"Step {step}")
            plt.tight_layout()
            plt.pause(0.1)
        
        if terminated or truncated:
            break
    
    plt.show()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dreamer agent")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize imagination vs reality")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_imagination(args.model)
    else:
        evaluate(
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic
        )
