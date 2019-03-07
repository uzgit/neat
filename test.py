import os
import sys

# For use in contexts where this file is imported from outside this directory.
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from genome import *
from neural_network import *
from visualize import *
from species import *
from population import *
from xor import *

# edge1 = EdgeGene(1, 2, 112312)
# edge2 = EdgeGene(1, 2, 212319)
# edge3 = EdgeGene(2, 3, 312312)
# edge4 = EdgeGene(1, 3, 412312)
#
# print(edge1)
# print(edge2)
# print(edge3)
# print(edge4)

# genome1 = Genome.default(identifier=1, num_inputs=2, num_outputs=1, num_hidden_nodes=1, output_activation_function=step, mode="unconnected")
# random_edge = genome1.get_possible_edge()
# print(random_edge)
# genome1.add_edge(random_edge)
#
# for i in range(5):
#     # print(genome1)
#     random_edge = genome1.get_possible_edge()
#     # print(random_edge)
#     if random_edge is not None:
#         genome1.add_edge(random_edge)
#
# print(genome1)
#
# for i in range(20):
#     random_edge = genome1.get_random_existing_edge()
#     if random_edge is not None:
#         genome1.remove_edge(random_edge)

# removed_node = genome1.get_node(4)
# genome1.remove_node(removed_node)

# random_node = genome1.get_possible_node()
# print(random_node)
# genome1.add_node(random_node)

# print(genome1)

# genome1.num_adds = 0
# genome1.num_removes = 0
# genome1.num_agg = 0
# genome1.num_act = 0
# genome1.num_per = 0
# # genome1.mutate_add_edge()
#
# for i in range(2):
#     genome1.mutate_add_edge()
#     genome1.mutate_add_node()

# for i in range(10):
#     genome1.random_mutation()

# print(genome1)

# network = FeedForwardNeuralNetwork(genome1)
# print(network)

# genome1.remove_node(genome1.get_node(6))
# genome1.remove_node(genome1.get_node(5))

# print("num adds:", genome1.num_adds)
# print("num removes:", genome1.num_removes)
# print("num agg:", genome1.num_agg)
# print("num act:", genome1.num_act)
# print("num per:", genome1.num_per)

# print(network)
#
# # draw_genome_full(genome1, "images/genome1test")
# draw_neural_network_full(network, "images/network1")
#
# print(network.activate([1, 2]))

# genome2 = Genome.default(identifier=1, num_inputs=2, num_outputs=1, num_hidden_nodes=2, aggregation_function=average, activation_function=identity, weights="", mode="fully connected", output_activation_function=identity)
# print(genome2)
# # draw_genome_full(genome2)
# network = FeedForwardNeuralNetwork(genome2)
# print(network)
# draw_neural_network_full(network)
# print(network.activate([2, 5]))

# genome1 = Genome.default(2, 1, num_hidden_nodes=3, mode="fully connected")
# genome2 = Genome.default(2, 1, num_hidden_nodes=3, mode="fully connected")
#
# for i in range(2):
#     genome1.mutate_remove_node()
#
# genome1.fitness = 2
# genome2.fitness = 1
#
# print(genome1)
# print(genome2)
#
# # print(Genome.similarity(genome1, genome2))
# child = Genome.crossover(genome1, genome2)
# print(child)
#
# network1 = FeedForwardNeuralNetwork(genome1)
# network2 = FeedForwardNeuralNetwork(genome2)
# network3 = FeedForwardNeuralNetwork(child)

# draw_genome_full(genome1, "images/parent1")
# draw_genome_full(genome2, "images/parent2")
# draw_genome_full(child,   "images/child")

# draw_neural_network_full(network1, "images/network1")
# draw_neural_network_full(network2, "images/network2")
# draw_neural_network_full(network3, "images/network3")

population = Population(num_inputs=2, num_outputs=1, initial_num_hidden_nodes=0, max_num_hidden_nodes=1, output_activation_function=sigmoid, population_size=150, num_initial_mutations=1)
champion = population.run_skeleton(evaluation_function=test_xor_sigmoid, fitness_goal=3.999)#, num_generations=100)
population.save("save_test.population")
# population2 = Population.from_file("save_test.population")

# print(population2.champion)
# input()
# population2.run_skeleton(evaluation_function=test_xor_sigmoid, fitness_goal=3.98)#, num_generations=100)

# network = FeedForwardNeuralNetwork(champion)
# draw_neural_network_full(network)
# test_xor_sigmoid_print(network)
# print("max num hidden nodes in population: {}".format(max([genome.num_hidden_nodes() for genome in population.genomes])))

# population = Population.from_file("save_test.population")
# champion = population.run_skeleton(evaluation_function=test_xor_sigmoid, fitness_goal=3.99)#, num_generations=100)
