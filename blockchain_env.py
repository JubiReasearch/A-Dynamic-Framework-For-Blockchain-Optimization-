import numpy as np

class SimpleBlockchainModel:
    def __init__(self, dt=1.0, data_source=None):
        self.nx = 4
        self.nu = 4
        self.dt = dt
        self.data_source = data_source
        self.A = np.array([
            [0.95,0.0,0.0,0.0],
            [0.05,0.9,0.0,0.0],
            [0.0,0.0,0.98,0.0],
            [0.0,0.01,0.0,0.97]
        ])
        self.B = np.array([
            [40.0,30.0,20.0,5.0],
            [10.0,8.0,-15.0,2.0],
            [-0.2,-0.1,-0.3,1.0],
            [0.5,0.4,0.2,1.2]
        ])
        self.x_base = np.array([100.0,200.0,0.8,50.0])
        self.noise_scale = np.array([2.0,5.0,0.01,1.0])

    def reset(self):
        return self.x_base.copy()

    def step(self, x, u):
        u = np.clip(u, 0.0, 1.0)
        x_next = self.A.dot(x - self.x_base) + self.B.dot(u) + self.x_base
        noise = np.random.normal(scale=self.noise_scale)
        x_next = x_next + noise
        x_next[0] = max(0.0, x_next[0])
        x_next[1] = max(1.0, x_next[1])
        x_next[2] = np.clip(x_next[2], 0.0, 1.0)
        x_next[3] = max(0.1, x_next[3])
        return x_next
