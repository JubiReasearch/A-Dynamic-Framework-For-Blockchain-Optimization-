# This file integrates Stable-Baselines3 PPO for learning MPC weights.
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import torch

class MPCWeightEnv(gym.Env):
    """Gym env where actions are MPC Q-diagonal weights and reward is composite performance."""
    def __init__(self, env_factory, mpc_factory, rollout_T=20):
        super().__init__()
        self.env_factory = env_factory
        self.mpc_factory = mpc_factory
        self.rollout_T = rollout_T
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=float)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=float)
    def reset(self):
        self.env = self.env_factory(); self.x = self.env.reset(); return np.zeros(8, dtype=float)
    def step(self, action):
        # action in [0,1]^4 -> map to Q diag via alpha
        alpha = np.array([200.0,100.0,250.0,20.0]); w_vec = action * alpha
        mpc = self.mpc_factory(w_vec); total_reward=0.0; x=self.x
        for _ in range(self.rollout_T):
            try:
                x_ref = getattr(self.env, 'x_base', x); u = mpc.solve(x, [x_ref for _ in range(mpc.Np)])
            except Exception:
                u = np.ones(4)*0.3
            x = self.env.step(x,u); scores = np.array([                        (x[0]/1000.0), (1 - x[1]/2000.0), x[2], (1 - x[3]/200.0)])
            total_reward += float(np.mean(np.clip(scores,0,1)))
        obs = np.zeros(8, dtype=float); done = True; info = {}; return obs, total_reward, done, info

    def render(self, mode='human'): pass

def train_sb3(env_factory, mpc_factory, timesteps=10000, model_path='data/ppo_mpc_weights'):
    gym_env = MPCWeightEnv(env_factory, mpc_factory)
    model = PPO('MlpPolicy', gym_env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    return model
