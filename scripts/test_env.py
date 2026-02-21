import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from env.trading_env import TradingEnv

# Create dummy price data
# 200 timesteps, 15 assets
dummy_prices = np.random.rand(200, 15) * 100

# Create environment
env = TradingEnv(dummy_prices)

# Reset environment
state, _ = env.reset()

print("Initial state shape:", state.shape)

# Run one full episode
done = False
step_count = 0

while not done:
    action = env.action_space.sample()  # Random action
    state, reward, done, _, _ = env.step(action)

    print(f"Step {step_count} | Reward: {reward:.4f}")
    
    # Check state shape
    assert state.shape == (229,), "State shape mismatch!"

    step_count += 1

print("Environment ran successfully.")