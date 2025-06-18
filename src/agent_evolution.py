import numpy as np
import random
import copy

from sister import Sister
from utils import unpack_genotype, get_random_genotype


#  the class that simualtes the social ecncounters of sister groups
class SisterEvolution:
    def __init__(
            self,
            dims: dict[str:int],
            seed: int,
            n_individuals: int,
            n_generations: int,
            mutation_rate: float,
            initial_food: int,
            encounters: int,
            cut_off: int,
            ) -> None:

        # global parameters, set by individual
        self.dims = dims  # neural network dimensions
        self.n_individuals = n_individuals  # individuals in the population
        self.n_generations = n_generations  # number of generations to be run
        self.mutation_rate = mutation_rate  # the limit for the gaussian value generation
        self.initial_food = initial_food  # the intial food every individual starts with
        self.encounters = encounters  # how many ecncouter every generation each individual has 
        self.cut_off = cut_off

        self.bloodlines = []  # bloodlines in the population
        self.gen_bloodline()  # generate intial bloodlines
        self.population = self.initialise_population() # initialise pop

        # metrics for the simulation
        self.gen_n = 0
        self.gen_data= []
        self.gen_metrics = []
        self.best_individuals= []
        self.gen_best_fitness_overtime = []
        self.best_historical_fitness = {'fitness': 0}

        self.inputs = {'related': np.array([1, 0]),
                       'non_related': np.array([0, 1]),}

        # set the different environents
        if seed:
            np.random.seed(seed)

    # generate the initial bloodlines
    def gen_bloodline(self) -> None:
        for i in range(int(self.n_individuals/5)):
            num = np.random.randint(1, 1000)

            # in case a duplicate number is generated
            while num in set(self.bloodlines):
                num = np.random.randint(1, 1000)

            self.bloodlines.append(num)

    # generate initial population with initial bloodlines
    def initialise_population(self) -> dict[Sister]:

        population = []

        # loop through all bloodlines
        for i, bloodline in enumerate(self.bloodlines):

            sister_geno = get_random_genotype(nodes=self.dims)  # generate initial network params
            sister_geno_unpacked = unpack_genotype(genotype=sister_geno, nodes=self.dims) #  unpack into numpy arrays of right shape

            # amount of duplicate 'sisters' has, intially 20 groups of 5
            # NEEDS TO BE EDITED TO HAVE CUSTOM AMOUNT OF BLOODLINES
            for dups in range(int(self.n_individuals/len(self.bloodlines))):
                sister = {'agent': self.generate_sister(sister_geno_unpacked, bloodline),
                          'genotype': sister_geno}
                population.append(sister)

        return population

    #  generate an indvidual given geno type and blood line
    def generate_sister(self, sister_geno: list[dict], bloodline: int) -> list[object]: # agents

        # contrust sister
        sister = Sister(W1=sister_geno['W1'],
                        W2=sister_geno['W2'],
                        B1=sister_geno['B1'],
                        B2=sister_geno['B2'],
                        dims=self.dims,
                        bloodline=bloodline,
                        num_food=self.initial_food)

        return sister

    # Simulate an ecounter between two networks
    def encounter(self, sister_1, sister_2) -> None:
        # if the networks are sisters
        if self.population[sister_1]['agent'].bloodline == self.population[sister_2]['agent'].bloodline:

            input = self.inputs['related']

            # sister 2 processing
            sis2out = self.population[sister_2]['agent'].output(input) #  generate output
            self.population[sister_2]['agent'].sis_encounter += 1 #  increment sister encounter

            # if sister 2 acts altruistically
            if sis2out == 1:
                self.population[sister_2]['agent'].alt_sis(1) #  increment altruism to sister's
                self.population[sister_1]['agent'].add_num_food(1) #  give sister food
                self.population[sister_2]['agent'].add_num_food(-1) # decrement own food
            else:
                # if sister 2 acts egotistcally by producing zero
                self.population[sister_2]['agent'].sis_ego +=1 # increment decision

            # generate sister 2's output
            sis1out = self.population[sister_1]['agent'].output(input)
            self.population[sister_1]['agent'].sis_encounter += 1 # increment encounter with sister

            # if sister 1 acts atruistically
            if sis1out == 1:
                self.population[sister_1]['agent'].alt_sis(1)
                self.population[sister_2]['agent'].add_num_food(1)
                self.population[sister_1]['agent'].add_num_food(-1)
            else:
                self.population[sister_1]['agent'].sis_ego +=1
        else:
            # generate output for networks that are not genetically related i.e. non sisters
            input = self.inputs['non_related']

            # sister 2 processing
            sis2out = self.population[sister_2]['agent'].output(input) #  get decision
            self.population[sister_2]['agent'].non_sis_encounter += 1 #  increment non sister iteraction encounter
            # give to non sister
            if sis2out == 1:
                self.population[sister_2]['agent'].alt_non_sis(1) # increment altrusim to non sis
                self.population[sister_1]['agent'].add_num_food(1) # give other network food
                self.population[sister_2]['agent'].add_num_food(-1)
            else:
                # do not give to non sister
                self.population[sister_2]['agent'].non_sis_ego +=1

            # sister 1 processing
            self.population[sister_1]['agent'].non_sis_encounter += 1
            sis1out = self.population[sister_1]['agent'].output(input)
            if sis1out == 1:
                self.population[sister_1]['agent'].alt_non_sis(1)
                self.population[sister_2]['agent'].add_num_food(1)
                self.population[sister_1]['agent'].add_num_food(-1)
            else:
                self.population[sister_1]['agent'].non_sis_ego +=1

    # generate metrics for each generation
    def generation_metrics(self):

        all_metrics = []  # all metrics
        num_bloodlines = 0  # number of blood lines in the current population
        weights_bloodlines = {k:[] for k in self.bloodlines}

        # enumerate population and append the indviduals states
        for i, agent in enumerate(self.population):

            all_metrics.append(
                    [
                        agent["agent"].sis_ego,
                        agent["agent"].non_sis_ego,
                        agent["agent"].sis,
                        agent["agent"].non_sis,
                        agent["agent"].num_food,
                        agent["agent"].sis_encounter,
                        agent["agent"].non_sis_encounter
                    ]
                    )

        # adding the weights matrices to their repective blood lines for analysis
        for i, agent in enumerate(self.population):
            weights_bloodlines[agent["agent"].bloodline].append({'W1': agent['agent']._W1, 'W2': agent['agent']._W2})

        all_metrics = np.asarray(all_metrics)
        gen_metrics_mean = np.mean(all_metrics, axis=0)
        num_bloodlines = len(self.bloodlines)

        # save generation data
        self.gen_data.append(
                {
                    'gen_n': self.gen_n,
                    'metric_means': list(gen_metrics_mean),
                    'gen_metrics': all_metrics,
                    'num_bloodlines': num_bloodlines,
                    'weights_bloodlines': weights_bloodlines,
                    'fittest': self.best_individuals[-1]
                }
                )

    def calculate_fitness(self) -> None:
        # calculate the top 20 fitnesses after 50 generations

        # store all info and fitness of every indvidual
        fitnesses = []

        # grab important stats
        for i, sister in enumerate(self.population):
            fitnesses.append(
                    {
                        'genotype': sister['genotype'],
                        'non_sis_encounter': sister['agent'].non_sis_encounter,
                        'non_sis': sister['agent'].non_sis,
                        'non_sis_ego': sister['agent'].non_sis_ego,
                        'sis': sister['agent'].sis,
                        'sis_ecounter': sister['agent'].sis_encounter,
                        'sis_ego': sister['agent'].sis_ego,
                        'fitness': sister['agent'].num_food,
                        'bloodline': sister['agent'].bloodline,
                        'W1': sister['agent']._W1,
                        'W2': sister['agent']._W2
                        }
                    )

        # sort dictionary on fitness
        sorted_fitnesses = sorted(fitnesses, key=lambda x: x['fitness'], reverse=True)[:self.cut_off]

        # keep track of best individuals every generation
        self.gen_best_fitness_overtime.append(sorted_fitnesses[0]['fitness'])

        # keep track of best fitness acheived across generations
        if sorted_fitnesses[0]['fitness'] > self.best_historical_fitness['fitness']:
            self.best_historical_fitness = {'fitness': sorted_fitnesses[0]['fitness'], 'genotype': sorted_fitnesses[0]['genotype']}

        # store the best individuals every generation
        self.best_individuals.append(sorted_fitnesses)

    def reproduce(self) -> list[Sister]:
        # repoduce from best individuals

        # self.gen_data.append(self.population)
        population = []
        new_bloodlines = [] # new bloodline for new population

        # generate new pop from best individuals in last generation
        for i, individual in enumerate(self.best_individuals[-1]):
            new_bloodlines.append(individual['bloodline'])

            # create correct amount of dupes
            for dups in range(int(self.n_individuals/len(self.best_individuals[-1]))):

                # deep copy geno of parent
                sister_geno = copy.deepcopy(individual['genotype'])

                # mutate all network parameters
                for j, _ in enumerate(sister_geno):
                    sister_geno[j] += random.gauss(0.0, self.mutation_rate)

                # unpack genos into separate matrices and vectors
                sister_geno_unpacked = unpack_genotype(genotype=sister_geno,
                                                       nodes=self.dims)

                # add new individual
                sister = {
                        'agent': self.generate_sister(sister_geno_unpacked, bloodline=individual['bloodline']),
                        'genotype': sister_geno
                        }

                population.append(sister)

        self.bloodlines = set(new_bloodlines)

        self.population = population

    # simulate one round of encounters
    def run_simulation_once(self) -> None:

        # random shuffle the population.
        np.random.shuffle(self.population)

        # loop over half the pop, every individual gets 1 interactions per cycle (100 per gen)
        for i in range(int(self.n_individuals / 2)):
            # collect in pairs [n, n+1]
            idx_1 = i * 2
            idx_2 = i * 2 + 1
            self.encounter(idx_1, idx_2)

    # run simulation of encounteres given the set amount
    def run_simulation(self) -> None:
        for _ in range(self.encounters):
            self.run_simulation_once()

    def run(self):

        # genotypes = self.initialise_population()
        # genotype = load_data('/home/joshua/Documents/university/al-data.txt')
        # all_best = []

        while self.gen_n < self.n_generations:
            self.run_simulation()
            self.calculate_fitness()
            self.generation_metrics()
            self.reproduce()

            self.gen_n += 1

        return self.gen_data
