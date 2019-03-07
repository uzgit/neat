#!/usr/bin/env python3

import sys
import graphviz

from genome import *
from neural_network import *
from visualize import *

assert len(sys.argv) == 2, "You must provide a .genome file to be visualized."

path = sys.argv[1]

genome = Genome.from_file(path)
network = FeedForwardNeuralNetwork(genome)

draw_neural_network_full(network, path + ".vis")
