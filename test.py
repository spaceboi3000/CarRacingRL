import gymnasium as gym
import torch
import cv2
import numpy as np
import argparse
from dreamer import Dreamer
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Configuration
ENV_NAME = "CarRacing-v3"
DEFAULT_MODEL_PATH = "dreamer_best.pth"

# Must match training configuration
STOCH_DIM = 30
DET_DIM = 200
HIDDEN_DIM = 300
EMBED_DIM = 1024


def preprocess(obs):
    # resize, grayscale, normalize
    img = cv2.resize(obs, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return (img / 255.0).astype(np.float32).reshape(1, 64, 64)


def process_action_for_env(action):
    # Convert tanh action to CarRacing action space
    processed = action.copy()
    processed[1] = np.clip((action[1] + 1) / 2, 0, 1)  # Gas: [-1,1] -> [0,1]
    processed[2] = np.clip((action[2] + 1) / 2, 0, 1)  # Brake: [-1,1] -> [0,1]
    return processed


def evaluate_with_dreams(model_path, num_episodes=5, deterministic=True, 
                         show_imagination_horizon=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
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
    
    # Setup visualization
    plt.ion()  # Interactive mode
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, show_imagination_horizon + 2, figure=fig)
    
    # Top row: Real observation and reconstruction
    ax_real = fig.add_subplot(gs[0, 0])
    ax_real.set_title("Real Observation (RGB)", fontsize=10)
    ax_real.axis('off')
    
    ax_processed = fig.add_subplot(gs[0, 1])
    ax_processed.set_title("Preprocessed", fontsize=10)
    ax_processed.axis('off')
    
    ax_recon = fig.add_subplot(gs[1, 0:2])
    ax_recon.set_title("Agent's Reconstruction (What it 'sees')", fontsize=10)
    ax_recon.axis('off')
    
    # Bottom row: Imagined future
    ax_imagined = []
    for i in range(show_imagination_horizon):
        ax = fig.add_subplot(gs[0, i + 2])
        ax.set_title(f"Dream t+{i+1}", fontsize=9)
        ax.axis('off')
        ax_imagined.append(ax)
    
    # Info text
    ax_info = fig.add_subplot(gs[1, 2:])
    ax_info.axis('off')
    
    plt.tight_layout()
    
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
        
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*50}")
        
        while not done:
            # Preprocess observation
            obs_processed = preprocess(obs)
            obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, 
                                     device=device).unsqueeze(0)
            
            # Get action and internal state from agent
            with torch.no_grad():
                action_tensor, prev_h, prev_z = agent.get_action(
                    obs_processed, prev_action, prev_h, prev_z,
                    deterministic=deterministic
                )
                
                # Get reconstruction
                state = agent.get_state_feature(prev_h, prev_z)
                recon = agent.decoder(state)
                
                # Imagine future steps
                imagine_h, imagine_z = prev_h.clone(), prev_z.clone()
                imagined_frames = []
                imagined_actions = []
                
                for _ in range(show_imagination_horizon):
                    imagine_state = agent.get_state_feature(imagine_h, imagine_z)
                    mean, std = agent.actor(imagine_state)
                    imagine_action = torch.tanh(mean) if deterministic else torch.tanh(mean + std * torch.randn_like(std))
                    imagine_h, imagine_z, _ = agent.rssm.imagine(imagine_z, imagine_action, imagine_h)
                    imagine_state = agent.get_state_feature(imagine_h, imagine_z)
                    imagined_frames.append(agent.decoder(imagine_state))
                    imagined_actions.append(imagine_action)
            
            # Update visualization every frame (or every N frames for performance)
            if step_count % 1 == 0:  # Change to % 2 or % 3 if too slow
                # Clear and update plots
                ax_real.clear()
                ax_real.imshow(obs)
                ax_real.set_title("Real Observation (RGB)", fontsize=10)
                ax_real.axis('off')
                
                ax_processed.clear()
                ax_processed.imshow(obs_processed[0], cmap='gray')
                ax_processed.set_title("Preprocessed", fontsize=10)
                ax_processed.axis('off')
                
                ax_recon.clear()
                ax_recon.imshow(recon[0, 0].cpu().numpy(), cmap='gray')
                ax_recon.set_title("Agent's Reconstruction (What it 'sees')", fontsize=10)
                ax_recon.axis('off')
                
                # Update imagined future
                for i, (ax, frame) in enumerate(zip(ax_imagined, imagined_frames)):
                    ax.clear()
                    ax.imshow(frame[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f"Dream t+{i+1}", fontsize=9)
                    ax.axis('off')
                
                # Update info
                ax_info.clear()
                ax_info.axis('off')
                action = action_tensor.cpu().numpy()[0]
                info_text = f"Episode: {ep+1}/{num_episodes} | Step: {step_count} | Reward: {total_reward:.1f}\n"
                info_text += f"Action: [Steer: {action[0]:+.2f}, Gas: {action[1]:+.2f}, Brake: {action[2]:+.2f}]"
                ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                           fontsize=11, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.draw()
                plt.pause(0.001)
            
            # Process action for environment
            action = action_tensor.cpu().numpy()[0]
            env_action = process_action_for_env(action)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            # Update state
            prev_action = action_tensor
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1} finished: Reward = {total_reward:.2f}, Steps = {step_count}")
    
    plt.ioff()
    plt.close()
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Mean reward:   {np.mean(episode_rewards):.2f}")
    print(f"Std reward:    {np.std(episode_rewards):.2f}")
    print(f"Min reward:    {np.min(episode_rewards):.2f}")
    print(f"Max reward:    {np.max(episode_rewards):.2f}")
    
    env.close()
    return episode_rewards


def evaluate(model_path, num_episodes=5, render=True, deterministic=True):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dreamer agent")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model weights")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (only for non-dream mode)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--show-dreams", action="store_true",
                        help="Show real-time dream visualization during inference")
    parser.add_argument("--imagination-horizon", type=int, default=5,
                        help="Number of future steps to imagine (for dream mode)")
    
    args = parser.parse_args()
    
    if args.show_dreams:
        print("Running evaluation with real-time dream visualization...")
        evaluate_with_dreams(
            model_path=args.model,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
            show_imagination_horizon=args.imagination_horizon
        )
    else:
        print("Running standard evaluation...")
        evaluate(
            model_path=args.model,
            num_episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic
        )