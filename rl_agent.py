import numpy as np, torch, torch.nn as nn, torch.optim as optim
from utils.metrics import compute_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last,h)); layers.append(nn.ReLU()); last=h
        layers.append(nn.Linear(last,act_dim))
        self.mean_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, obs):
        mu = self.mean_net(obs); std = torch.exp(self.log_std); return mu, std
class REINFORCEAgent:
    def __init__(self, obs_dim=8, act_dim=4, lr=3e-4, gamma=0.99):
        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr); self.gamma = gamma
    def sample_action(self, obs_np):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mu, std = self.policy(obs); dist = torch.distributions.Normal(mu, std)
        a = dist.rsample().squeeze(0); logp = dist.log_prob(a).sum()
        w_raw = torch.nn.functional.softplus(a); w_norm = (w_raw / (w_raw.sum() + 1e-8)).detach().cpu().numpy()
        return w_norm, logp
    def train_episode(self, env_factory, mpc_factory, obs_window=8, rollout_T=20):
        obs = np.zeros(obs_window, dtype=np.float32); log_probs = []; rewards=[]
        w_norm, logp = self.sample_action(obs); log_probs.append(logp)
        alpha = np.array([200.0,100.0,250.0,20.0])
        w_vec = w_norm * alpha; mpc = mpc_factory(w_vec); env = env_factory(); x = env.reset(); total_reward=0.0
        for t in range(rollout_T):
            try:
                x_ref = getattr(env, 'x_base', x); u = mpc.solve(x, [x_ref for _ in range(mpc.Np)])
            except Exception:
                u = np.ones(4)*0.3
            x = env.step(x,u)
            scores = compute_metrics(x); reward = float(np.mean(scores)); total_reward += reward
        returns = torch.tensor([total_reward], dtype=torch.float32, device=device)
        loss = -log_probs[0] * returns
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.0); self.optimizer.step()
        return total_reward
