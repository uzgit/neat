from genome import *

class Species:

    def __init__(self, genome, current_generation, identifier=None):

        self.identifier = identifier
        if self.identifier is None:
            self.set_identifier()

        self.starting_generation = current_generation

        self.ancestors = []
        self.genomes   = []
        self.elites    = []
        self.misfits   = []

        self.fitness_history = []
        self.fitness = None
        self.champion = deepcopy(genome)
        self.representative = deepcopy(genome)
        self.genomes.append(genome)
        self.age = 0

    def set_identifier(self):

        self.identifier = max(global_species_identifiers) + 1
        global_species_identifiers.append(self.identifier)

    # This function assumes that all genomes have been evaluated.
    def step_generation(self):

        assert len(self.genomes) > 0

        # Sort genomes by fitness descending.
        self.genomes.sort(key = lambda genome : genome.fitness, reverse = True)

        if (self.champion.fitness is None or self.genomes[0].fitness >= self.champion.fitness):
            self.champion = deepcopy(self.genomes[0])

        self.fitness = max([genome.fitness for genome in self.genomes])
        self.fitness_history.append(self.fitness)

        # Set current genomes as ancestors, then clear current genomes.
        self.ancestors = self.genomes.copy()

        # Choose a random representative from the previous generation
        self.representative = choice(self.ancestors)

        self.age += 1

    # This function assumes that step_generation() has already been called since the last call of reproduce()
    def reproduce(self, num_children):

        self.genomes.clear()
        self.misfits.clear()

        num_parents = int(max(2, species_reproduction_elitism * len(self.ancestors)))
        potential_parents = self.ancestors[0:num_parents]

        for i in range(num_children):
            parent_1 = choice(potential_parents)
            parent_2 = choice([potential_parent for potential_parent in potential_parents if potential_parent is not parent_1])

            child = Genome.crossover(parent_1, parent_2)
            child.random_mutation()

            if self.is_compatible_with(child):
                self.genomes.append(child)
            else:
                self.misfits.append(child)

        assert (len(self.genomes) + len(self.misfits)) == num_children

    def is_extinct(self):

        return self.size() < 2

    def add_genome(self, genome):

        self.genomes.append(genome)

    def is_compatible_with(self, genome):

        return Genome.similarity(self.representative, genome) >= species_similarity_threshold

    def is_stagnated(self):

        stagnated = False
        if self.age > species_stagnation_time:
            fitness_to_beat = self.fitness_history[-species_stagnation_time:]
            improvements = [fitness > fitness_to_beat for fitness in self.fitness_history[-species_stagnation_time:]]
            stagnated = not any(improvements)
        return stagnated

    def average_fitness(self):

        fitness_time = min(self.age, species_average_fitness_time)
        return sum(self.fitness_history[-fitness_time:]) / fitness_time

    def size(self):

        return len(self.genomes)

    def __str__(self):

        return "Species {}, age: {} generations, fitness: {}".format(self.identifier, self.age, self.fitness)

    def information_entry(self):

        return "%6s%6s%10s%15s%15s%10s" % (
        self.identifier, self.age, len(self.ancestors), round(self.average_fitness(), 2),
        round(self.fitness, 2), round(numpy.std([genome.fitness for genome in self.ancestors]), 3))