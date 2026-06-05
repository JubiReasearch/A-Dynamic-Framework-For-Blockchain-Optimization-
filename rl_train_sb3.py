from models.rl_agent_sb3 import train_sb3
from models.blockchain_env import SimpleBlockchainModel
from models.mpc_controller import MPCController

def env_factory():
    return SimpleBlockchainModel()
def mpc_factory_from_weights(w_diag):
    model = SimpleBlockchainModel(); return MPCController(model=model, Np=8, Nc=4, Q_diag=w_diag)

if __name__ == '__main__':
    train_sb3(env_factory, mpc_factory_from_weights, timesteps=20000)
