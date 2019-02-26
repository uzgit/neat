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

# variables for creating a default Genome
default_genome_mode = "unconnected"
#default_genome_mode = "fully connected"

# variables for comparing genomes
node_gene_coefficient = 0.6
edge_gene_coefficient = 0.9
edge_weight_coefficient = 0.2

# variables for Populations
default_population_size = 150