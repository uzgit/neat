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
        self.children = []

        self.fitness_history = []
        self.fitness = None
        self.champion = genome

    def step_generation(self):

        self.fitness = max([genome.fitness for genome in self.genomes] + [0])

        self.fitness_history.append( self.fitness )
        sorted_genomes = sorted(self.genomes, key=lambda genome : genome.fitness)
        if len(sorted_genomes) > 0 and sorted_genomes[0].fitness > self.champion.fitness:
            self.champion = sorted_genomes[0]

        self.ancestors.clear()
        for genome in self.genomes:
            self.ancestors.append(genome)

    def reproduce(self, num_individuals, next_genome_identifier):

        # Population will re-add genomes to species
        self.genomes.clear()


        sorted_ancestors = sorted(self.ancestors, key=lambda genome : genome.fitness)

        num_parents = floor(num_individuals * reproduction_elitism)

        potential_parents = []
        for i in range(min(num_parents, len(self.ancestors) - 1)):
            potential_parents.append(sorted_ancestors[i])

        print(len(sorted_ancestors))
        self.elites = []
        for i in range(min(elites_to_keep, num_individuals)):
            self.elites.append(sorted_ancestors[i])

        self.children.clear()
        num_children = (num_individuals - elites_to_keep)
        if num_children < 0:
            num_children = 0
        for i in range(num_children):

            parent_1 = choice(potential_parents)
            parent_2 = choice([parent for parent in potential_parents if parent is not parent_1])

            new_genome = Genome.crossover(parent_1, parent_2, next_genome_identifier)

            self.children.append(new_genome)

            next_genome_identifier += 1

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

        oldest_fitness_index = max(len(self.genomes) - (stagnation_time - 1), 0)
        fitness_to_beat = self.fitness_history[oldest_fitness_index]

        improvements = [fitness > fitness_to_beat for fitness in self.fitness_history[oldest_fitness_index:]]

        return not any(improvements)

    def size(self):

        return len(self.genomes)