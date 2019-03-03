# general variables for node biases
random_initial_bias = True
initial_weight_min = -5
initial_weight_max = 5
global_bias_min = -20
global_bias_max = 20

# general variables for edges
global_weight_min = -30
global_weight_max = 30

# variables for mutating edges with Genome.mutate_add_edge
initial_weight_min = -2
initial_weight_max = 2

# variables for mutating edges with Genome.mutate_scale_weight
weight_scale_min = -5
weight_scale_max = 5

# variables for NodeGenes
default_aggregation_function = "sum"
default_activation_function = "sigmoid"

# variables for input NodeGenes (NodeGene.is_input_node == True)
default_input_aggregation_function = "sum"
default_input_activation_function = "identity"

# variables for output NodeGenes
default_output_activation_function = "sigmoid"

# variables for creating a default Genome
default_genome_mode = "unconnected"
#default_genome_mode = "fully connected"

# variables for comparing genomes
node_gene_similarity_coefficient = 0.5
edge_gene_similarity_coefficient = 0.8
edge_weight_similarity_coefficient = 0.5

# variables for mutating genomes
max_num_mutations_per_individual_per_generation = 1
mutate_add_node_probability = 0.2
mutate_remove_node_probability = 0.1
mutate_set_bias_probability = 0.2
mutate_add_edge_probability = 0.9
mutate_remove_edge_probability = 0.4
mutate_reset_weight_probability = 0.2
mutate_scale_weight_probability = 0.2
mutate_change_aggregation_function_probability = 0.05
mutate_change_activation_function_probability = 0.05

# variables for Species
species_similarity_threshold = 0.6
stagnation_time = 2000 # number of generations without fitness improvement before a species is considered stagnant
elites_to_keep = 0 # doesn't work yet
reproduction_elitism = 0.4 # percentage of individuals who will reproduce to create the next generation of the species
species_average_fitness_time = 20 # number of generations over which to calculate average fitness

# variables for Populations
default_population_size = 150
default_max_num_hidden_nodes = 10
default_num_initial_mutations = 1