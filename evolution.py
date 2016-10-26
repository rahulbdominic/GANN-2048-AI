from random import *
import numpy as np
from network import *
from boardnew import *
from deap import creator, base, tools, algorithms

class GeneticAlgorithm(object):

    def __init__(self, genetics):
        self.genetics = genetics
        pass

    def run(self):
        population = self.genetics.initial()
        while True:
            fits_pops = [(self.genetics.fitness(ch), ch) for ch in
                         population]
            if self.genetics.check_stop(fits_pops):
                break
            population = self.next(fits_pops)
            pass
        return population

    def next(self, fits):
        parents_generator = self.genetics.parents(fits)
        size = len(fits)
        nexts = []
        while len(nexts) < size:
            parents = next(parents_generator)
            cross = random.random() \
                < self.genetics.probability_crossover()
            children = \
                (self.genetics.crossover(parents) if cross else parents)
            for ch in children:
                mutate = random.random() \
                    < self.genetics.probability_mutation()
                nexts.append((self.genetics.mutation(ch) if mutate else ch))
                pass
            pass
        return nexts[0:size]

    pass


class GeneticFunctions(object):

    def probability_crossover(self):
        r"""returns rate of occur crossover(0.0-1.0)"""

        return 1.0

    def probability_mutation(self):
        r"""returns rate of occur mutation(0.0-1.0)"""

        return 0.0

    def initial(self):
        r"""returns list of initial population
        """

        return []

    def fitness(self, chromosome):
        r"""returns domain fitness value of chromosome
        """

        return len(chromosome)

    def check_stop(self, fits_populations):
        r"""stop run if returns True
        - fits_populations: list of (fitness_value, chromosome)
        """

        return False

    def parents(self, fits_populations):
        r"""generator of selected parents
        """

        gen = iter(sorted(fits_populations))
        while True:
            (f1, ch1) = next(gen)
            (f2, ch2) = next(gen)
            yield (ch1, ch2)
            pass
        return

    def crossover(self, parents):
        r"""breed children
        """

        return parents

    def mutation(self, chromosome):
        r"""mutate chromosome
        """

        return chromosome

    pass

class EvolveNet(GeneticFunctions):

    def __init__(
        self,
        net,
        matrix,
        limit=50000,
        size=50,
        prob_crossover=0.9,
        prob_mutation=0.3,
        ):

        # self.target = self.text2chromo(target_text)

        self.matrix = matrix
        self.counter = 0
        self.net = net
        self.limit = limit
        self.size = size
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        pass

    # GeneticFunctions interface impls

    def probability_crossover(self):
        return self.prob_crossover

    def probability_mutation(self):
        return self.prob_mutation

    def initial(self):
        return [self.random_chromo() for j in range(self.size)]

    def fitness(self, chromo):

        '''REMEMBER TO PUT CHROMOSOME IN'''
        # larger is better, matched == 0
        # Feed data through the neural network and get the fitness score
        # as a result

        # 1) Construct Neural Net
        net = Network(self.net.getsizes(), self.counter)
        weights, biases = self.chromotowb(chromo)
        net.setbiases(biases)
        net.setweights(weights)

        # 2) Evaluate matrix till 5 illegal moves or lost
        matrix = [[self.matrix[x][y] for y in range(len(self.matrix[0]))] for x in range(len(self.matrix))]
        fitness = 0
        grid = GameGrid(matrix)
        illegal_moves = 0
        done = False
        while ((illegal_moves < 1) and (done == False)):
            activation_matrix = net.evaluate(np.array(grid.scalar_multiply()).flatten())
            if (activation_matrix[0] == 0):
                (matrix, done, invalid) = grid.move('w')
            if (activation_matrix[0] == 1):
                (matrix, done, invalid) = grid.move('s')
            if (activation_matrix[0] == 2):
                (matrix, done, invalid) = grid.move('a')
            if (activation_matrix[0] == 3):
                (matrix, done, invalid) = grid.move('d')
            if done == True:
                if invalid == True:
                    illegal_moves += 1
            else:
                if invalid == True:
                    illegal_moves += 1

        # Calculate fitness score
        fitness = sum(x for x in np.array(matrix).flatten())
        return fitness

    def check_stop(self, fits_populations):
        self.counter += 1
        if self.counter % 5 == 0:
            best_match = list(sorted(fits_populations))[-1][1]
            fits = [f for (f, ch) in fits_populations]
            best = max(fits)
            worst = min(fits)
            ave = sum(fits) / len(fits)
            print '[G %3d] score=(%4d, %4d, %4d)' \
                % (self.counter, best, ave, worst)
            pass
        return self.counter >= self.limit

    def parents(self, fits_populations):
        while True:
            father = self.tournament(fits_populations)
            mother = self.tournament(fits_populations)
            yield (father, mother)
            pass
        pass

    def crossover(self, parents):
        (father, mother) = parents
        index1 = random.randint(1, len(np.array(self.matrix).flatten()) - 2)
        index2 = random.randint(1, len(np.array(self.matrix).flatten()) - 2)
        if index1 > index2:
            (index1, index2) = (index2, index1)
        child1 = father[:index1] + mother[index1:index2] \
            + father[index2:]
        child2 = mother[:index1] + father[index1:index2] \
            + mother[index2:]
        return (child1, child2)

    def mutation(self, chromosome):
        mutated = list(chromosome)
        for i in range(70):
            index = random.randint(0, len(np.array(self.matrix).flatten()) - 1)
            vary = random.uniform(-0.3, 0.3)
            mutated[index] += vary
        return mutated

    # internals

    def tournament(self, fits_populations):
        (alicef, alice) = self.select_random(fits_populations)
        (bobf, bob) = self.select_random(fits_populations)
        return (alice if alicef > bobf else bob)

    def select_random(self, fits_populations):
        return fits_populations[random.randint(0, \
                                len(fits_populations) - 1)]

    def wbtochromo(self, weights, biases):
        biases = [np.random.randn(y, 1) for y in self.net.getsizes()[1:]]
        weights = [np.random.randn(y, x) for (x, y) in \
                   zip(self.net.getsizes()[:-1], self.net.getsizes()[1:])]
        chrom = biases.flatten() + weights.flatten()
        return chrom

    def reshape_weights(self, weights):
        c = 0
        new_weights = []
        for (k, l) in zip(self.net.getsizes()[1:], self.net.getsizes()[:-1]):
            arr_main = np.random.randn(l, k)

            for i in range(k):
                c += l
                np.insert(arr_main, 0, [weights[c - l: c]], axis = 1)

            np.delete(arr_main, 0, 0)
            new_weights.append(arr_main)

        return new_weights

    def reshape_biases(self, biases):
        c = 0
        new_biases = []
        for k in self.net.getsizes()[1:]:
            c += k
            new_biases.append(np.array(biases[c - k: c]))

        return new_biases

    def sizeweights(self):
        size = 0
        for (k, l) in zip(self.net.getsizes()[1:], self.net.getsizes()[:-1]):
            size += k * l
        return size

    def sizebiases(self):
        size = 0
        for k in self.net.getsizes()[1:]:
            size += k
        return size

    def chromotowb(self, chromo):
        # Weights and biases are arrays of numpy arrays. Get each separately
        # and then encode.
        weights = chromo[:self.sizeweights()]
        self.reshape_weights(weights)
        biases = chromo[self.sizebiases():]
        self.reshape_biases(biases)

        return weights, biases

    def random_chromo(self):
        return [random.random() for i in range(self.sizeweights() + self.sizebiases())]
    pass
