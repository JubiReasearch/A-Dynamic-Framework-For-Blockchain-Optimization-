import matplotlib.pyplot as plt, os
def plot_pareto(scores, front, out_dir='data/results'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.scatter(scores[:,0], scores[:,1], c='gray', alpha=0.5, label='All')
    plt.scatter(front[:,0], front[:,1], c='red', label='Pareto')
    plt.xlabel('Scalability'); plt.ylabel('Decentralization')
    plt.title('Pareto: Scalability vs Decentralization'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'pareto_sc_vs_dec.png')); plt.close()
def plot_objective_3d(scores, out_dir='data/results'):
    from mpl_toolkits.mplot3d import Axes3D; import numpy as np
    os.makedirs(out_dir, exist_ok=True); fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=scores[:,3], cmap='viridis')
    ax.set_xlabel('Scalability'); ax.set_ylabel('Decentralization'); ax.set_zlabel('Security')
    fig.colorbar(p, label='Efficiency'); plt.title('Quadrilemma 3D Visualization'); plt.savefig(os.path.join(out_dir,'quad_3d.png')); plt.close()
def compare_real_vs_simulated(real_scores, sim_scores, out_dir='data/results'):
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6)); plt.scatter(sim_scores[:,0], sim_scores[:,1], c='gray', alpha=0.5, label='Simulated')
    plt.scatter(real_scores[:,0], real_scores[:,1], c='blue', alpha=0.7, label='Real'); plt.xlabel('Scalability'); plt.ylabel('Decentralization')
    plt.title('Real vs Simulated'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(out_dir,'real_vs_sim.png')); plt.close()
