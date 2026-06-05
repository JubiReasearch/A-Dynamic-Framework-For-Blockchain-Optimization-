import numpy as np, random
from deap import base, creator, tools
from utils.metrics import compute_metrics

class ParetoOptimizer:
    def __init__(self, n_pop=80, ngen=50, seed=42):
        self.n_pop = n_pop; self.ngen = ngen; self.seed = seed

    def evaluate(self, model, mpc_template):
        random.seed(self.seed); np.random.seed(self.seed)
        creator.create('FitnessMulti', base.Fitness, weights=(1.0,1.0,1.0,1.0))
        creator.create('Individual', list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register('attr_float', random.random)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=model.nu)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        def evaluate_ind(ind):
            u = np.clip(np.array(ind,dtype=float),0.0,1.0)
            x = model.reset(); acc = np.zeros(4); T=20
            for _ in range(T):
                x = model.step(x,u); acc += compute_metrics(x)
            return tuple((acc/T).tolist())
        toolbox.register('evaluate', evaluate_ind)
        toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
        toolbox.register('mutate', tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0/model.nu)
        toolbox.register('select', tools.selNSGA2)
        pop = toolbox.population(n=self.n_pop)
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind,fit in zip(invalid, fitnesses): ind.fitness.values = fit
        for gen in range(1,self.ngen+1):
            offspring = tools.selTournamentDCD(pop, len(pop)); offspring = [toolbox.clone(ind) for ind in offspring]
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= 0.9:
                    toolbox.mate(ind1, ind2); del ind1.fitness.values; del ind2.fitness.values
            for mutant in offspring:
                if random.random() <= 0.2:
                    toolbox.mutate(mutant); del mutant.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid)
            for ind,fit in zip(invalid, fitnesses): ind.fitness.values = fit
            pop = toolbox.select(pop + offspring, self.n_pop)
            if gen % 10 == 0 or gen==1:
                front0 = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
                print(f'[NSGA-II] Gen {gen}, front size {len(front0)}')
        pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        pareto_solutions = [(np.array(ind), np.array(ind.fitness.values)) for ind in pareto]
        try:
            del creator.FitnessMulti; del creator.Individual
        except Exception:
            pass
        return np.array([p[1] for p in pareto_solutions]), pareto_solutions
