from math import *
from random import *

from globals import *
from genome import *

class Species:

    def __init__(self, identifier, current_generation, genome):

        assert genome is not None

        self.identifier = identifier
        self.starting_generation = current_generation

        self.ancestors = []
        self.genomes = []
        self.elites = []
        self.misfits = []

        self.fitness_history = []
        self.fitness = None
        self.champion = deepcopy(genome)

    def step_generation(self):

        # set fitness metadata
        # self.fitness = max([genome.fitness for genome in self.genomes] + [0])
        # self.fitness_history.append( self.fitness )

        # sort genomes by fitness descending
        self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
        if self.genomes[0].fitness >= self.champion.fitness:
            self.champion = deepcopy(self.genomes[0])

        # update ancestors
        self.ancestors.clear()
        for genome in self.genomes:
            self.ancestors.append(genome)

        # clear genomes
        self.genomes.clear()

    def reproduce(self, num_children, next_genome_identifier):

        assert len(self.ancestors) >= 2

        # Population object will re-add genomes to species
        self.genomes.clear()

        # we need at least 2 parents, and we can have at most len(self.ancestors) parents
        num_parents = int(max(2, len(self.ancestors) * reproduction_elitism))
        if num_parents > len(self.ancestors):
            num_parents = len(self.ancestors)

        potential_parents = self.ancestors[0 : num_parents]

        # print("in species: ancestors = {}".format(self.ancestors))
        # print("in species: num_parents = {}".format(num_parents))
        # print("in species: potential_parents = {}".format(potential_parents))
        # print("in species: addresses: {}".format([id(ancestor) for ancestor in self.ancestors]))

        # create children
        for i in range(num_children):

            # randomly choose parents
            parent_1 = choice(potential_parents)
            parent_2 = choice([potential_parent for potential_parent in potential_parents if potential_parent is not parent_1])

            # create a new child
            child = Genome.crossover(parent_1, parent_2, next_genome_identifier)
            child.random_mutation()

            # increment identifier
            next_genome_identifier += 1

            # place child inside or outside the species based on its similarity to best-performing individual
            # if self.champion.similarity(child) > species_similarity_threshold:
            #     self.genomes.append(child)
            # else:
            #     self.misfits.append(child)
            self.genomes.append(child)

        return next_genome_identifier

    def reproduce_deprecated(self, num_individuals, next_genome_identifier):

        # Population will re-add genomes to species
        self.genomes.clear()

        sorted_ancestors = sorted(self.ancestors, key=lambda genome : genome.fitness)

        num_parents = ceil(num_individuals * reproduction_elitism)

        potential_parents = []
        for i in range(min(num_parents, len(self.ancestors) - 1)):
            if sorted_ancestors[i] not in potential_parents:
                potential_parents.append(sorted_ancestors[i])

        self.elites = []
        for i in range(min(elites_to_keep, len(self.ancestors) - 1)):
            self.elites.append(sorted_ancestors[i])
            self.genomes.append(sorted_ancestors[i])

        self.children.clear()
        num_children = (num_individuals - elites_to_keep)
        print("in Species.reproduce, num_parents={}, num_children={}".format(num_parents, num_children))
        if num_children < 0:
            num_children = 0

        if len(potential_parents) > 1:
            for i in range(num_children):

                parent_1 = choice(potential_parents)
                parent_2 = choice([parent for parent in potential_parents if parent is not parent_1])

                new_genome = Genome.crossover(parent_1, parent_2, next_genome_identifier)
                self.children.append(new_genome)

                next_genome_identifier += 1

        elif len(potential_parents) == 1:
            new_genome = deepcopy(potential_parents[0])
            self.children.append(new_genome)

        print("num children: {}; made {} children".format(num_children, len(self.children)))

        return next_genome_identifier

    def add_genome(self, genome):

        self.genomes.append(genome)

    def is_compatible_with(self, genome):

        result = None

        if self.champion.similarity(genome) > species_similarity_threshold:
            result = True
        else:
            result = False

        return result

    def is_stagnated(self):

        stagnated = False

        if len(self.fitness_history) > stagnation_time:

            fitness_to_beat = self.fitness_history[-stagnation_time]
            starting_index = -stagnation_time

            improvements = [fitness > fitness_to_beat for fitness in self.fitness_history[starting_index:]]

            stagnated = not any(improvements)

        return stagnated

    def add_fitness(self, fitness):

        self.fitness = fitness
        self.fitness_history.append(fitness)

    def average_fitness(self):

        if len(self.fitness_history) < species_average_fitness_time:

            fitness_time = len(self.fitness_history)

        else:

            fitness_time = species_average_fitness_time

        return sum(self.fitness_history[-fitness_time:]) / fitness_time

    def size(self):

        return len(self.genomes)

    def __str__(self):

        representation = "Species {}, age {} generations, fitness {}.".format(self.identifier, len(self.fitness_history), self.fitness)

        return representation

    def information_entry(self):

        return "%6s%6s%10s%15s" % (self.identifier, len(self.fitness_history), len(self.genomes), round(self.fitness, 2))