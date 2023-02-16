from utils import *
from tqdm import tqdm

from agent_evolution import SisterEvolution

# a function to iterate over varying mutation and population sizes and collect the data for runs
def parameter_sweep(save: bool) -> None:
    # initial_energy

    simulation_params = {
            'n_individuals': 100,
            'n_generations': 50,
            'mutation_rate': 0.01,
            'initial_food': 100,
            'encounters': 100,
            'cut_off': 20
            }

    # list of different values
    mutation_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    number_individuals = np.arange(100, 505, 100)

    #define the network nodes
    N, D_in, H, D_out = 1, 2, 3, 1
    dims = {'output': D_out, 'hidden': H, 'input': D_in, 'num_examples': N}


    # iterate different initial seed environments
    for i, seed in enumerate(range(1,6)):
        seed_params = []
        print(f'Running evolution for seed {i}')
        # for each pop run for every mutation
        for n_individuals in tqdm(number_individuals):
            simulation_params['n_individuals'] = n_individuals #  change amount of indviduals to current iteration
            for mutation_rate in mutation_rates:
                simulation_params['mutation_rate'] = mutation_rate #  set mutation rate
                run_data = run_simulation(iterations=10, dims=dims, seed=seed, simulation_params=simulation_params) # run sim
                seed_params.append({'n_individuals': n_individuals, 'mutation_rate': mutation_rate, 'data': run_data}) # append run data


        # save to local disk
        if save:
            save_data(seed_params, f'/home/joshua/Documents/AL/Project/altruism_model/data/seed{i}-data.txt')


# run simulation with specfied parameters
def evolve_network(dims: dict[str:int], seed: int, simulation_params: dict) -> list[dict]:

    evolution = SisterEvolution(dims=dims, seed=seed, **simulation_params)

    data = evolution.run()

    return data

# for running once with preset params
def run_once():

    simulation_params = {
            'n_individuals': 100,
            'n_generations': 50,
            'mutation_rate': 0.01,
            'initial_food': 100,
            'encounters': 100,
            'cut_off': 20
            }

    N, D_in, H, D_out = 1, 2, 3, 1
    dims = {'output': D_out, 'hidden': H, 'input': D_in, 'num_examples': N}
    run_data = run_simulation(iterations=1, dims=dims, seed=1, simulation_params=simulation_params)


def run_simulation(iterations: int, dims: dict[str:int], seed: int, simulation_params: dict):

    run_data = [] # collect data over the runs of ga

    # loop for amount of iterations
    for _ in range(iterations):

        # evolve ga with given parameters
        data = evolve_network(dims=dims, seed=seed, simulation_params=simulation_params)

        run_data.append(data)

    return run_data

if __name__ == "__main__":
    # parameter_sweep(save=True)
    run_once()

