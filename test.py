#!/usr/bin/env python3

from genome import *
from neural_network import *

nodes = []
nodes.append(NodeGene(1, "sum", "identity", is_input_node=True))
nodes.append(NodeGene(4, "sum", "identity", is_input_node=True))
nodes.append(NodeGene(2, "sum", "sigmoid", is_output_node=True))
#nodes.append(NodeGene(3, "sum", "relu"))
nodes.append(NodeGene(5, "sum", "identity"))
#nodes.append(NodeGene(6, "sum", "lelu"))
#nodes.append(NodeGene(7, "sum", "sigmoid"))
nodes.append(NodeGene(8, "sum", "identity", is_input_node=True))

#nodes.append(NodeGene(20, "sum", "identity"))

edges = []
# edges.append(EdgeGene(1, 1, 1, 3, 10))
# edges.append(EdgeGene(3, 2, 3, 2, 1))
# edges.append(EdgeGene(2, 3, 4, 2, 1))
# edges.append(EdgeGene(4, 4, 1, 5, 5))
# edges.append(EdgeGene(5, 5, 5, 2, 3, is_enabled=False))
# edges.append(EdgeGene(6, 6, 5, 6, 1))
# edges.append(EdgeGene(7, 7, 6, 2, 5))
# edges.append(EdgeGene(8, 8, 8, 2, 1))

edges.append(EdgeGene(1, 1, 1, 2, 2))
edges.append(EdgeGene(2, 2, 4, 2, 1))
edges.append(EdgeGene(3, 3, 8, 2, 1))
edges.append(EdgeGene(4, 4, 1, 5, 2))
edges.append(EdgeGene(5, 5, 5, 2, 2))

# edges.append(EdgeGene(21, 21, 5, 20, 2))
# edges.append(EdgeGene(22, 22, 20, 2, 100))

genome1 = Genome(1, nodes, edges)

neural_network1 = FeedForwardNeuralNetwork(genome1)
print(neural_network1)
print()

#genome1.mutate_remove_node()
#genome1.mutate_add_node(20, 21, 22)
genome1.mutate_reset_weight(-2, 2)

neural_network2 = FeedForwardNeuralNetwork(genome1)
print(neural_network2)

print( neural_network1.activate([0, 1, 1]), neural_network2.activate([0, 1, 1]) )
print( neural_network1.activate([0, 2, 2]), neural_network2.activate([0, 2, 2]) )
