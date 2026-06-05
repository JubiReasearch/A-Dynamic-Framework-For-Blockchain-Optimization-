from models.blockchain_env import SimpleBlockchainModel
from models.mpc_controller import MPCController
from models.pareto_optimizer import ParetoOptimizer
from models.game_theory import GameTheoryOptimizer
from utils.plots import plot_pareto, plot_objective_3d, compare_real_vs_simulated
from models.data_bridge import BlockchainDataBridge
import os

RESULTS_DIR = os.path.join('data','results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_full_simulation(use_real_data=False):
    data_bridge = None
    if use_real_data:
        data_bridge = BlockchainDataBridge(source='ethereum')
    env = SimpleBlockchainModel(data_source=data_bridge)
    mpc = MPCController(model=env, Np=8, Nc=4)
    pareto = ParetoOptimizer(n_pop=80, ngen=30)
    scores, pareto_solutions = pareto.evaluate(env, mpc)
    print('Pareto solutions found:', scores.shape[0])
    plot_pareto(scores, scores, out_dir=RESULTS_DIR)
    plot_objective_3d(scores, out_dir=RESULTS_DIR)
    game = GameTheoryOptimizer(n_agents=10)
    avg_strategy = game.step(global_target=scores.mean(axis=0)[:3])
    print('Average strategy:', avg_strategy)
    if data_bridge is not None and data_bridge.real_data is not None:
        compare_real_vs_simulated(data_bridge.real_data.values, scores, out_dir=RESULTS_DIR)

if __name__ == '__main__':
    run_full_simulation(use_real_data=False)
