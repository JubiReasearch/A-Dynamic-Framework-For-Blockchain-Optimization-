import numpy as np
class GameTheoryOptimizer:
    def __init__(self, n_agents=8):
        self.n_agents = n_agents
    def step(self, global_target):
        strategies = np.random.rand(self.n_agents, len(global_target))
        for _ in range(50):
            strategies += 0.1 * (global_target - strategies)
            strategies = np.clip(strategies, 0, 1)
        return strategies.mean(axis=0)
