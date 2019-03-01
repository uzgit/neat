#!/usr/bin/env python3
import os
import sys

from xor import *

from population import *
from visualize import *

# nodes = []
# nodes.append(NodeGene(1, "sum", "identity", is_input_node=True))
# nodes.append(NodeGene(4, "sum", "identity", is_input_node=True))
# nodes.append(NodeGene(2, "sum", "sigmoid", is_output_node=True))
# #nodes.append(NodeGene(3, "sum", "relu"))
# nodes.append(NodeGene(5, "sum", "identity"))
# #nodes.append(NodeGene(6, "sum", "lelu"))
# nodes.append(NodeGene(7, "sum", "sigmoid", is_output_node=True))
# nodes.append(NodeGene(8, "sum", "identity", is_input_node=True))

#nodes.append(NodeGene(20, "sum", "identity"))

# edges = []
# edges.append(EdgeGene(1, 1, 1, 3, 10))
# edges.append(EdgeGene(3, 2, 3, 2, 1))
# edges.append(EdgeGene(2, 3, 4, 2, 1))
# edges.append(EdgeGene(4, 4, 1, 5, 5))
# edges.append(EdgeGene(5, 5, 5, 2, 3, is_enabled=False))
# edges.append(EdgeGene(6, 6, 5, 6, 1))
# edges.append(EdgeGene(7, 7, 6, 2, 5))
# edges.append(EdgeGene(8, 8, 8, 2, 1))

# edges.append(EdgeGene(1, 1, 1, 2, -2))
# edges.append(EdgeGene(2, 2, 4, 2, -1))
# edges.append(EdgeGene(3, 3, 8, 2, 1))
# edges.append(EdgeGene(4, 4, 1, 5, 2))
# edges.append(EdgeGene(5, 5, 5, 2, 2))
#
# edges.append(EdgeGene(6, 6, 4, 7, 2))
# edges.append(EdgeGene(7, 7, 8, 5, 2))
# edges.append(EdgeGene(8, 8, 4, 5, 2))

# edges.append(EdgeGene(21, 21, 5, 20, 2))
# edges.append(EdgeGene(22, 22, 20, 2, 100))

# genome1 = Genome(1, nodes, edges)
# genome2 = Genome(2, nodes, edges)

# neural_network1 = FeedForwardNeuralNetwork(genome1)

#print(genome1)

# print(neural_network1)
# print()

# genome1.mutate_add_node(9, 21, 22)
# genome1.mutate_add_node(23, 24, 25)
# genome1.mutate_add_node(26, 27, 28)
# genome1.mutate_remove_node(5)
# genome1.mutate_remove_node(26)
# genome1.mutate_reset_weight()
# genome1.mutate_scale_weight()
# for i in range(10):
#     genome1.mutate_add_edge(66, weight=23.795 - i)
# genome1.mutate_remove_edge(2)
# genome1.mutate_change_aggregation_function()
# genome1.mutate_change_activation_function()

# genome1.save("genome1.genome")
# genome5 = Genome.from_file("genome1.genome")
# neural_network2 = FeedForwardNeuralNetwork(genome5)
# os.remove("genome1.genome")

#print(neural_network1)

# neural_network_1 = FeedForwardNeuralNetwork(genome1)
# neural_network_2 = FeedForwardNeuralNetwork(genome2)
#
# draw_neural_network_full(neural_network_1, "images/network1")
# draw_neural_network_full(neural_network_2, "images/network2")
#
# genome3 = Genome.crossover(genome2, genome1, 3)
# neural_network_3 = FeedForwardNeuralNetwork(genome3)
# draw_neural_network_full(neural_network_3, "images/network3")

# print( neural_network1.activate([0, 1, 1]), neural_network2.activate([0, 1, 1]) )
# print( neural_network1.activate([0, 2, 2]), neural_network2.activate([0, 2, 2]) )

# print(neural_network2.genome)
# print(neural_network2)

# draw_neural_network_active(neural_network2, "images/network_2_active")
# draw_neural_network_full(  neural_network2, "images/network_2_full")
# draw_neural_network_active(neural_network2, "images/network 2")

# genome2 = Genome.default(2, 5, 2, num_hidden_nodes=10, mode="full")
# network3 = FeedForwardNeuralNetwork(genome2)
# draw_neural_network_active(network3)

# genome5.mutate_remove_edge()
# genome5.mutate_remove_node()

# neural_network3 = FeedForwardNeuralNetwork(genome5)
# draw_neural_network_full(neural_network3, "images/network3_full")
# print(genome1.similarity(genome5))

# genome with no connections
# genome10 = Genome(10, nodes, [])
# neural_network10 = FeedForwardNeuralNetwork(genome10)
# print(neural_network10.activate([1, 2, 3]))
# draw_neural_network_active(neural_network10)

# genome4 = Genome.default(4, 4, 2, num_hidden_nodes=3, mode="fully connected", output_activation_function="binary_step")
# print(genome4)
# neural_network4 = FeedForwardNeuralNetwork(genome4)
# draw_neural_network_active(neural_network4)
# draw_neural_network_full(neural_network4)

# print(neural_network4.activate([1, 2, 3, 4]))
# print(neural_network4.activate([1, 2, 3, 4]))
# print(neural_network4.activate([1, 2, 3, 4]))
# print(neural_network4.activate([3, 7, 8, 5]))

# population = Population(150, 500, 2, 1, max_num_hidden_nodes=10, output_activation_function="binary_step")

# for genome in population.genomes:
#     print(genome)
# print(population)

# for i in range(50):
#     print(i)
#     population.mutate_all_genomes()

# neural_network50 = FeedForwardNeuralNetwork(population.genomes[5])

# for i in range(20):
#     inputs = [uniform(-10, 10) for ii in range(2)]
#     print(neural_network50.activate(inputs))

# draw_neural_network_full(neural_network50, "images/complex")

# max_fitness = 0
# while max_fitness < 4:
#
#     population = Population(150, 500, 2, 1, max_num_hidden_nodes=10, output_activation_function="binary_step")
#     best_genome = population.run_with_local_fitness_function(test_xor, num_generations=1000, fitness_goal=4)
#     best_neural_network = FeedForwardNeuralNetwork(best_genome)
#
#     max_fitness = best_neural_network.genome.fitness
#
# draw_neural_network_active(best_neural_network, "images/xor_active")
# draw_neural_network_full(best_neural_network, "images/xor_full")
#
# print(best_genome)
# print("fitness:", test_xor_print(best_neural_network))





# genome1 = Genome.default(1, 2, 1, num_hidden_nodes=3)
# # neural_network1 = FeedForwardNeuralNetwork(genome1)
# # draw_neural_network_full(neural_network1, "images/neural_newtork")
#
# for i in range(5):
#
#     genome = deepcopy(genome1)
#     # genome.random_mutation()
#     genome.mutate_add_edge()
#     print(type(genome.mutate_add_node()) == type(genome.nodes[0]))
#     # neural_network = FeedForwardNeuralNetwork(genome)
#     # draw_neural_network_full( neural_network, "images/nerual_network_{}".format(i))


population = Population(10, 500, 2, 1, max_num_hidden_nodes=10, output_activation_function="binary_step")
population.initialize_genomes()
population.initial_mutation()

population.set_species()
print("num species: {}".format(population.num_species()))
print("num individuals: {}".format(population.size()))

for genome in population.genomes:
    # genome.fitness = randint(1, 20)
    genome.fitness = 1

population.set_species_fitnesses()
population.step_generation()

num_species_individuals = 0
for species in population.species:
    for genome in species.genomes:
        num_species_individuals += 1
print("num individuals available through species: {}".format(num_species_individuals))

for species in population.species:
    for genome in species.genomes:
        neural_network = FeedForwardNeuralNetwork(genome)
        draw_neural_network_full( neural_network, "images/nerual_network_{}_{}".format(species.identifier, genome.identifier))

# best_genome = population.run_with_local_fitness_function(test_xor)#, num_generations=1000, fitness_goal=4)
# best_neural_network = FeedForwardNeuralNetwork(best_genome)
#
# draw_neural_network_active(best_neural_network, "images/best_active")
# draw_neural_network_full(best_neural_network, "images/best_full")
#
# print(best_genome)
# print("fitness:", test_xor_print(best_neural_network))

