import os, torch, numpy as np
from models.rl_agent import REINFORCEAgent
from models.blockchain_env import SimpleBlockchainModel
from models.mpc_controller import MPCController

os.makedirs('data', exist_ok=True)

def env_factory():
    return SimpleBlockchainModel()
def mpc_factory_from_weights(w_diag):
    model = SimpleBlockchainModel()
    return MPCController(model=model, Np=8, Nc=4, Q_diag=w_diag)

def train_rl(n_episodes=200, log_interval=10):
    agent = REINFORCEAgent(obs_dim=8, act_dim=4, lr=3e-4)
    best_reward = -1e9
    for ep in range(1, n_episodes+1):
        r = agent.train_episode(env_factory, mpc_factory_from_weights)
        if ep % log_interval == 0 or ep==1:
            print(f'Episode {ep}, recent reward {r:.4f}')
        if r > best_reward:
            best_reward = r
            torch.save(agent.policy.state_dict(), 'data/best_policy.pth')
    print('Done. Best reward', best_reward)

if __name__ == '__main__':
    train_rl(n_episodes=200)
