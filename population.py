from globals import *
from genome import *
from neural_network import *

class Population:

    def __init__(self, population_size=default_population_size):

        self.innovation_number = 1
        self.genomes = []
        self.species = []