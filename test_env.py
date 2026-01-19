import gymnasium as gym

# 1. Create the environment
# render_mode="human" lets you see the window pop up
env = gym.make("CarRacing-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    # 2. Sample random actions (Steering, Gas, Brake)
    # This is the "Babbling" phase I mentioned earlier!
    action = env.action_space.sample() 
    
    # 3. Step through the world
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()