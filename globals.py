import os
import sys

# For use in contexts where this file is imported from outside this directory.
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from functions import *

# Adjustable parameters for configuring the evolution

# general variables for node biases
random_initial_bias = True
bias_min = -10
bias_max = 10
initial_weight_min = -5
initial_weight_max = 5
global_bias_min = -20
global_bias_max = 20

# general variables for edges
global_weight_min = -10
global_weight_max = 10

# variables for mutating edges with Genome.mutate_add_edge
initial_weight_min = -2
initial_weight_max = 2

# variables for NodeGenes
# default_aggregation_function = "sum"
# default_activation_function = "sigmoid"
default_aggregation_function = sum
default_activation_function = sigmoid

# variables for input NodeGenes (NodeGene.is_input_node == True)
# default_input_aggregation_function = "sum"
# default_input_activation_function = "identity"
default_input_aggregation_function = sum
default_input_activation_function = identity

# variables for output NodeGenes
default_output_activation_function = "sigmoid"
default_output_activation_function = sigmoid

# variables for creating a default Genome
default_genome_mode = "unconnected"
# default_genome_mode = "fully connected"

# variables for comparing genomes
node_gene_similarity_measure = 0.8
node_bias_similarity_measure = 0
node_aggregation_function_similarity_measure = 0
node_activation_function_similarity_measure  = 0
edge_gene_similarity_measure = 0.8
edge_weight_similarity_measure = 0

# variables for mutating genomes
max_num_mutations_per_individual_per_generation = 1
mutate_add_node_probability = 0.2
mutate_remove_node_probability = 0.1
mutate_perturb_bias_probability = 0.2
mutate_set_bias_probability = 0.2
mutate_add_edge_probability = 0.6
mutate_remove_edge_probability = 0.4
mutate_reset_weight_probability = 0.2
mutate_scale_weight_probability = 0.2
mutate_perturb_weight_probability = 0.3
mutate_change_aggregation_function_probability = 0.05
mutate_change_activation_function_probability = 0.05

# variables for Species
species_similarity_threshold = 0.75
species_stagnation_time = 2000 # number of generations without fitness improvement before a species is considered stagnant
species_elitism = 0.1 # this proportion of the number of children will be the elites from the previous generation
species_reproduction_elitism = 0.4 # percentage of individuals who will reproduce to create the next generation of the species
species_average_fitness_time = 20 # number of generations over which to calculate average fitness

# variables for Populations
default_population_size = 150
default_max_num_hidden_nodes = 15
default_num_initial_mutations = 1

# Set to True if the population size must be exactly equal to the size set by the user. The actual population size
# varies throughout execution in order to maintain a per-species population of at least 2, which is necessary for
# reproduction. This is for use in comparing this algorithm to other algorithms with respect to performance after
# the evaluation of a certain number of genomes.
exact_population_size = True

########################################################################################################################
# the following are critical variables for NEAT, so it's best not to touch them, but you should definitely touch them
########################################################################################################################

# innovations are formatted as "{input_node}->{output_node}" : innovation_number
global_genome_identifiers = [0]
global_innovations = {None : 0}
global_species_identifiers = [0]
