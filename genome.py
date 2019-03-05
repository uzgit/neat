from copy import deepcopy
from random import *
import pickle

from functions import *
from globals import *

class NodeGene:

    def __init__(self, identifier, aggregation_function=default_aggregation_function, bias=None, activation_function=default_activation_function, is_input_node=False, is_output_node=False, is_enabled=True):

        self.identifier = identifier
        self.aggregation_function = aggregation_function
        self.activation_function = activation_function
        self.is_input_node = is_input_node
        self.is_output_node = is_output_node
        self.is_enabled = is_enabled

        self.bias = bias
        if self.bias is None and random_initial_bias and not(self.is_input_node or self.is_output_node):
            self.bias = uniform(bias_min, bias_max)

    def sanity_check(self):

        assert self.identifier is not None
        assert self.aggregation_function in aggregation_functions
        assert self.activation_function in activation_functions
        assert not (self.is_input_node and not self.is_enabled)
        assert not (self.is_output_node and not self.is_enabled)
        assert not (self.is_input_node and self.is_output_node)

    def __str__(self):

        representation = "node {}: {}, bias: {}, {}".format(self.identifier, function_names[self.aggregation_function], self.bias if self.bias is None else round(self.bias, 2), function_names[self.activation_function])

        if not self.is_enabled:
            representation += " (disabled) "
        if self.is_input_node:
            representation += " (input)"
        elif self.is_output_node:
            representation += " (output)"

        return representation

    def __repr__(self):

        return self.__str__()

class EdgeGene:

    def __init__(self, input_node_identifier, output_node_identifier, identifier=None, innovation_number=None, weight=None, is_enabled=True):

        # These should never change after being set.
        self.identifier = identifier
        self.innovation_number = innovation_number
        self.input_node_identifier = input_node_identifier
        self.output_node_identifier = output_node_identifier

        # These can change after being set.
        self.weight = weight
        self.is_enabled = is_enabled

        if self.weight is None:
            self.weight = uniform(initial_weight_min, initial_weight_max)

        self.innovation = "{}->{}".format(self.input_node_identifier, self.output_node_identifier)
        if self.innovation_number is None:
            self.set_innovation_number()

        self.sanity_check()

    # The global_innovations dictionary contains a list of all unique edges that have ever been created, along with
    # their innovation numbers. This is to avoid the competing conventions problem.
    def set_innovation_number(self):

        if self.innovation in global_innovations:
            self.innovation_number = global_innovations[self.innovation]

        else:
            self.innovation_number = max(global_innovations.values()) + 1
            global_innovations.update({self.innovation : self.innovation_number})

    def sanity_check(self):

        assert self.identifier is not None
        assert self.input_node_identifier is not None
        assert self.output_node_identifier is not None
        assert self.input_node_identifier is not self.output_node_identifier, "{} and {}".format(self.input_node_identifier, self.output_node_identifier)
        assert self.innovation_number is not None
        assert self.is_enabled is not None
        assert self.weight is not None

    def __str__(self):

        representation = "edge {} ({}): {}->{}, {}".format(self.identifier, self.innovation_number, self.input_node_identifier, self.output_node_identifier, round(self.weight, 2))

        if self.is_enabled:
            representation += " enabled"
        else:
            representation += " disabled"

        return representation

    def __repr__(self):

        return self.__str__()

class Genome:

    # Basic Genome constructor.
    def __init__(self, num_inputs, num_outputs, nodes=None, edges=None, max_num_hidden_nodes=default_max_num_hidden_nodes, identifier=None):

        self.identifier = identifier
        if identifier is None:
            self.set_identifier()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_num_hidden_nodes = max_num_hidden_nodes

        self.nodes = []
        if nodes is not None:
            self.nodes = deepcopy(nodes)

        self.edges = []
        if edges is not None:
            self.edges = deepcopy(edges)

        self.fitness = None

    def set_identifier(self):

        self.identifier = max(global_genome_identifiers) + 1
        global_genome_identifiers.append(self.identifier)

    # Easier way to create a Genome with nodes and possible edges.
    @classmethod
    def default(cls, num_inputs, num_outputs, num_hidden_nodes=0, aggregation_function=default_aggregation_function, activation_function=default_activation_function, input_aggregation_function=default_input_aggregation_function, input_activation_function=default_input_activation_function, output_activation_function=default_output_activation_function, mode=default_genome_mode, weights="randomized", max_num_hidden_nodes=default_max_num_hidden_nodes, identifier=None):

        nodes = []
        edges = []

        genome = Genome(identifier=identifier, num_inputs=num_inputs, num_outputs=num_outputs, max_num_hidden_nodes=max_num_hidden_nodes)

        for i in range(num_inputs):
            genome.add_node( NodeGene(genome.next_node_identifier(), aggregation_function=input_aggregation_function, activation_function=input_activation_function, is_input_node=True, bias=None) )

        for i in range(num_outputs):
            genome.add_node( NodeGene(genome.next_node_identifier(), aggregation_function=aggregation_function, activation_function=output_activation_function, is_output_node=True, bias=None) )

        for i in range(num_hidden_nodes):
            genome.add_node( NodeGene(genome.next_node_identifier(), aggregation_function=aggregation_function, activation_function=activation_function) )

        if mode == "fully connected":

            input_nodes  = [node for node in genome.nodes if node.is_input_node]
            hidden_nodes = [node for node in genome.nodes if not node.is_input_node and not node.is_output_node]
            output_nodes = [node for node in genome.nodes if node.is_output_node]

            if num_hidden_nodes != 0:
                for input_node in input_nodes:
                    for hidden_node in hidden_nodes:

                        if weights == "randomized":
                            weight = None
                        else:
                            weight = 1

                        # input_node_identifier, output_node_identifier, identifier=None, innovation_number=None, weight=None, is_enabled=True, genome=None):
                        genome.add_edge( EdgeGene(input_node.identifier, hidden_node.identifier, weight=weight, identifier=genome.next_edge_identifier()) )

                for hidden_node in hidden_nodes:
                    for output_node in output_nodes:

                        if weights == "randomized":
                            weight = None
                        else:
                            weight = 1

                        genome.add_edge( EdgeGene(hidden_node.identifier, output_node.identifier, weight=weight, identifier=genome.next_edge_identifier()) )
            else:
                for input_node in input_nodes:
                    for output_node in output_nodes:

                        if weights == "randomized":
                            weight = uniform(global_weight_min, global_weight_max)
                        else:
                            weight = 1

                        genome.add_edge( EdgeGene(input_node.identifier, output_node.identifier, weight=weight, identifier=genome.next_edge_identifier()) )

        return genome

    @classmethod
    def crossover(cls, genome1, genome2):

        assert genome1.fitness is not None
        assert genome2.fitness is not None
        # assert Genome.similarity(genome1, genome2) >= species_similarity_threshold

        better_parent = genome1 if genome1.fitness >= genome2.fitness else genome2
        worse_parent  = genome2 if genome2.fitness <  genome1.fitness else genome1

        worse_parent_edges = {edge.innovation : edge for edge in worse_parent.edges}
        edges = []
        for edge in better_parent.edges:
            if edge.innovation in worse_parent_edges:

                possible_edges = [edge, worse_parent_edges[edge.innovation]]
                edges.append( choice( possible_edges ) )

            else:
                edges.append(edge)
        edges.sort(key=lambda edge : edge.innovation_number)

        # We have to ensure that all nodes with enabled edges are also enabled
        used_node_identifiers = {edge.input_node_identifier for edge in edges if edge.is_enabled}
        used_node_identifiers = used_node_identifiers.union(edge.output_node_identifier for edge in edges if edge.is_enabled)

        nodes = []
        better_parent_nodes = {node.identifier : node for node in better_parent.nodes}
        worse_parent_nodes = {node.identifier : node for node in worse_parent.nodes}
        for node in better_parent.nodes:
            if node.identifier in worse_parent_nodes:

                possible_nodes = [node, worse_parent_nodes[node.identifier]]

                # The node must be enabled.
                if node.identifier in used_node_identifiers:
                    truly_possible_nodes = [node for node in possible_nodes if node.is_enabled]
                    nodes.append( choice(truly_possible_nodes) )
                else:
                    nodes.append( choice(possible_nodes) )

            else:
                nodes.append(node)

        # node_identifiers = [node.identifier for node in nodes]
        # for node_identifier in used_node_identifiers:
        #     if node_identifier not in node_identifiers:
        #
        #         if node_identifier in better_parent_nodes:
        #             nodes.append(better_parent_nodes[node_identifier])
        #             print("added node {} from better parent")
        #         elif node.identifier in worse_parent_nodes:
        #             nodes.append(worse_parent_nodes[node_identifier])
        #         else:
        #             raise RuntimeError

        assert used_node_identifiers.issubset(set([node.identifier for node in nodes]))

        child = Genome(num_inputs=genome1.num_inputs, num_outputs=genome1.num_outputs, nodes=nodes)
        for edge in edges:
            child.add_edge(edge)

        return child

    @classmethod
    def similarity(cls, genome1, genome2):

        numerator   = 0
        denominator = 0

        genome1_node_identifiers = [(node.identifier, node.is_enabled) for node in genome1.nodes]
        genome2_node_identifiers = [(node.identifier, node.is_enabled) for node in genome2.nodes]

        numerator += node_gene_similarity_measure * len( set(genome1_node_identifiers).intersection(set(genome2_node_identifiers)) )
        denominator += node_gene_similarity_measure * max(len(genome1.nodes), len(genome2.nodes))

        genome1_edge_identifiers = [(edge.innovation, edge.is_enabled) for edge in genome1.edges]
        genome2_edge_identifiers = [(edge.innovation, edge.is_enabled) for edge in genome2.edges]

        numerator += edge_gene_similarity_measure * len( set(genome1_edge_identifiers).intersection(set(genome2_edge_identifiers)) )
        denominator += edge_gene_similarity_measure * max(len(genome1.edges), len(genome2.edges))

        if denominator == 0:
            denominator = 1

        return numerator / denominator

    def random_mutation(self):

        theoretically_possible_mutations = Genome.mutations.copy()

        # determine which structural mutations are possible
        # (nonstructural mutations are always possible

        # We cannot add a node to the Genome if we have already reached the maximum number of hidden nodes, or if there
        # are no edges to split.
        if self.num_edges() == 0 or (self.max_num_hidden_nodes is not None and self.num_hidden_nodes() >= self.max_num_hidden_nodes):
            theoretically_possible_mutations.pop(Genome.mutate_add_node)
            # print("removing add node")

        # We cannot modify nodes if we have 0 hidden nodes.
        if self.num_hidden_nodes() == 0:
            # print("removing remove node")
            theoretically_possible_mutations.pop(Genome.mutate_remove_node)
            theoretically_possible_mutations.pop(Genome.mutate_change_aggregation_function)
            theoretically_possible_mutations.pop(Genome.mutate_change_activation_function)

            # theoretically_possible_mutations.pop(Genome.mutate_perturb_bias)

        # We cannot add an edge if the graph is fully connected, or if the addition of any new edge would result
        # in a cycle, since we are using only feed-forward networks.
        if self.get_possible_edge() == None:
            theoretically_possible_mutations.pop(Genome.mutate_add_edge)

        # We cannot modify an edge if the Genome contains no edges.
        if self.num_edges() == 0:
            theoretically_possible_mutations.pop(Genome.mutate_remove_edge)
            theoretically_possible_mutations.pop(Genome.mutate_perturb_weight)

        # Ensure that the random number is greater than the probability of at least one of the mutations.
        # This way we know that at least one mutation is guaranteed to happen.
        random_number_max = max(theoretically_possible_mutations.values()) - 0.00001
        random_number = uniform(0, random_number_max)

        # Filter possible mutations by probability according to the randomly generated number.
        possible_mutations = [mutation for mutation in theoretically_possible_mutations if theoretically_possible_mutations[mutation] > random_number]

        # Choose a random element from the possible mutations and carry it out.
        mutation = choice(possible_mutations)
        mutation(self)

    # Adds a random enabled node to the Genome.
    def mutate_add_node(self):

        new_node = self.get_possible_node()
        edge = self.get_random_existing_edge()

        assert new_node is not None
        assert edge.is_enabled

        self.remove_edge(edge)
        input_edge  = EdgeGene(identifier=self.next_edge_identifier(),
                               input_node_identifier = edge.input_node_identifier,
                               output_node_identifier = new_node.identifier,
                               weight = 1)
        output_edge = EdgeGene(identifier=self.next_edge_identifier(),
                               input_node_identifier = new_node.identifier,
                               output_node_identifier = edge.output_node_identifier,
                               weight = edge.weight)
        self.add_node(new_node)
        self.add_edge(input_edge)
        self.add_edge(output_edge)


    # Selects a random enabled node and disables it.
    def mutate_remove_node(self):

        removed_node = self.get_random_existing_hidden_node()
        self.remove_node(removed_node)

    # Selects a random enabled node and changes its bias to a random number constrained by global_bias_min and global_bias_max.
    def mutate_perturb_bias(self):

        perturbed_node = self.get_random_existing_hidden_or_output_node()
        self.perturb_bias(perturbed_node)

    # Selects a random enabled node and changes its aggregation function to a random different function in the list of
    # aggregation functions.
    def mutate_change_aggregation_function(self):

        changed_node = self.get_random_existing_hidden_node()
        possible_aggregation_functions = aggregation_functions.copy()
        possible_aggregation_functions.remove(changed_node.aggregation_function)
        changed_node.aggregation_function = choice(possible_aggregation_functions)

    # Selects a random enabled node and changes its activation function to a random different function in the list of
    # activation functions.
    def mutate_change_activation_function(self):

        changed_node = self.get_random_existing_hidden_node()
        possible_activation_functions = activation_functions.copy()
        possible_activation_functions.remove(changed_node.activation_function)
        changed_node.activation_function = choice(possible_activation_functions)

    # Creates and adds a random edge.
    def mutate_add_edge(self):

        new_edge = self.get_possible_edge()
        self.add_edge(new_edge)

    # Selects and disables a random enabled edge.
    def mutate_remove_edge(self):

        removed_edge = self.get_random_existing_edge()
        self.remove_edge(removed_edge)

    # Selects an edge and changes its weight to a random number constrained by global_weight_min and global_weight_max.
    def mutate_perturb_weight(self):

        perturbed_edge = self.get_random_existing_edge()
        self.perturb_weight(perturbed_edge)

    # Adds a given node to the Genome.
    def add_node(self, node):

        assert node is not None
        assert node.identifier not in [node.identifier for node in self.nodes]
        self.nodes.append(node)

    # generates the next possible node, if the genome has not already reached max_nodes number of nodes.
    # does not add the node to the genome. this is an intermediate function that should not be called from the outside.
    def get_possible_node(self):

        new_node = None

        if self.max_num_hidden_nodes is None or self.num_hidden_nodes() < self.max_num_hidden_nodes:
            new_node = NodeGene(self.next_node_identifier())
        return new_node

    # Adds a given edge to the Genome.
    def add_edge(self, edge):

        edge = deepcopy(edge)

        edge.identifier = self.next_edge_identifier()

        # Sanity check
        assert edge.innovation not in [edge.innovation for edge in self.edges if edge.is_enabled]
        if edge.is_enabled:
            for node in self.nodes:
                if node.identifier == edge.input_node_identifier or node.identifier == edge.output_node_identifier:
                    assert node.is_enabled

        existing_edge = ([existing_edge for existing_edge in self.edges if existing_edge.innovation == edge.innovation] + [None])[0]
        if existing_edge is None:
            self.edges.append(edge)
        else:
            existing_edge.is_enabled = True
            existing_edge.weight = edge.weight

    # generates a random possible edge, if there is any pair of nodes in the genome for which an edge can be created.
    # this function avoids creating duplicate edges and cycles. does not add the edge to the genome.
    # this is an intermediate function that should not be called from the outside.
    def get_possible_edge(self):

        new_edge = None

        possible_input_nodes = [node for node in self.nodes if not node.is_output_node and node.is_enabled]
        possible_output_nodes = [node for node in self.nodes if not node.is_input_node and node.is_enabled]

        shuffle(possible_input_nodes)
        shuffle(possible_output_nodes)

        found_edge = False
        possible_input_node_index = 0
        while possible_input_node_index < len(possible_input_nodes) and not found_edge:
            predecessors = self.get_predecessors(possible_input_nodes[possible_input_node_index])

            possible_output_node_index = 0
            while possible_output_node_index < len(possible_output_nodes) and not found_edge:

                input_node_identifier = possible_input_nodes[possible_input_node_index].identifier
                output_node_identifier = possible_output_nodes[possible_output_node_index].identifier
                innovation = "{}->{}".format(input_node_identifier, output_node_identifier)

                if innovation not in [edge.innovation for edge in self.edges if edge.is_enabled] and possible_output_nodes[possible_output_node_index].identifier not in predecessors:

                    new_edge = EdgeGene(input_node_identifier, output_node_identifier, identifier=self.next_edge_identifier())
                    found_edge = True

                possible_output_node_index += 1

            possible_input_node_index += 1

        return new_edge

    # Returns a random, enabled, hidden node.
    def get_random_existing_hidden_node(self):

        assert self.num_hidden_nodes() > 0
        return choice([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled])

    def get_random_existing_hidden_or_output_node(self):

        return choice([node for node in self.nodes if (node.is_output_node or not node.is_input_node)])

    # Nothing is ever removed from the genome. Instead, the node and connected edges are disabled.
    def remove_node(self, node):

        assert not (node.is_input_node or node.is_output_node)

        node.is_enabled = False
        for edge in self.edges:
            if edge.input_node_identifier == node.identifier or edge.output_node_identifier == node.identifier:
                self.remove_edge(edge)

    # Returns a random enabled edge.
    def get_random_existing_edge(self):

        random_edge = None
        if self.num_edges() > 0:
            random_edge = choice([edge for edge in self.edges if edge.is_enabled])
        return random_edge

    # Disables the given edge. No component is removed from a Genome. "Removed" components are simply disabled.
    def remove_edge(self, edge):

        edge.is_enabled = False

    def perturb_weight(self, edge):

        edge.weight = uniform(global_weight_min, global_weight_max)

    def perturb_bias(self, node):

        node.bias = uniform(global_bias_min, global_bias_max)

    # Returns the number of enabled edges in the Genome.
    def num_edges(self):

        return len([edge for edge in self.edges if edge.is_enabled])

    # This function returns a list of the identifiers of nodes whose output propagates to the parameter node. This is
    # used to avoid creating cycles when adding a new edge. It is implemented this way to add as little to the genome
    # structure as possible.
    def get_predecessors(self, node):

        predecessors = [node.identifier]
        node_stack = [node]
        edge_stack = []

        while len(node_stack) > 0:

            # Add all input edges from each node to the node stack.
            for current_node in node_stack:

                # print([edge for edge in self.edges if edge.output_node_identifier == current_node.identifier and edge.input_node_identifier not in predecessors])
                edge_stack += [edge for edge in self.edges if edge.output_node_identifier == current_node.identifier and edge.input_node_identifier not in predecessors]

                # Accumulate the predecessors.
                if current_node.identifier not in predecessors:
                    predecessors.append(current_node.identifier)

            # Add all input nodes from each edge in the edge stack to the node stack.
            node_stack.clear()
            for edge in edge_stack:
                node_stack.append(self.get_node(edge.input_node_identifier))

            # Reset the edge stack to an empty list.
            edge_stack.clear()

        return predecessors

    # This function gets the NodeGene object that is identified by the identifier parameter.
    def get_node(self, identifier):

        result = None
        if identifier in [node.identifier for node in self.nodes]:
            result = [node for node in self.nodes if node.identifier == identifier][0]
        return result

    # Returns the number of enabled hidden nodes in the Genome.
    def num_hidden_nodes(self):

        return len([node for node in self.nodes if not node.is_input_node and not node.is_output_node and node.is_enabled])

    def num_all_nodes(self):

        return len([node for node in self.nodes if node.is_enabled])

    # Get the identifier of any node added to the Genome at this point.
    def next_node_identifier(self):

        return max([node.identifier for node in self.nodes] + [0]) + 1

    # Get the identifier of any edge added to the Genome at this point.
    def next_edge_identifier(self):

        return max([edge.identifier for edge in self.edges] + [0]) + 1

    def __str__(self):

        representation = "Genome {}, fitness {}:\n".format(self.identifier, self.fitness)

        representation += "\tNodes: {} input, {} active hidden, {} output\n".format(self.num_inputs, self.num_hidden_nodes(), self.num_outputs)

        for node in self.nodes:
            representation += "\t\t" + str(node) + "\n"

        representation += "\tEdges: {} enabled, {} disabled\n".format(len([edge for edge in self.edges if edge.is_enabled]), len([edge for edge in self.edges if not edge.is_enabled]))
        for edge in self.edges:
            representation += "\t\t" + str(edge) + "\n"

        representation += "\n"

        return representation

    def __repr__(self):

        return self.__str__()

# A dictionary of all Genome mutations, paired with their respective probabilities.
Genome.mutations = {
    Genome.mutate_add_node                      : mutate_add_node_probability,
    Genome.mutate_remove_node                   : mutate_remove_node_probability,
    Genome.mutate_perturb_bias                  : mutate_perturb_bias_probability,
    Genome.mutate_add_edge                      : mutate_add_edge_probability,
    Genome.mutate_remove_edge                   : mutate_remove_edge_probability,
    Genome.mutate_perturb_weight                : mutate_perturb_weight_probability,
    Genome.mutate_change_aggregation_function   : mutate_change_aggregation_function_probability,
    Genome.mutate_change_activation_function    : mutate_change_activation_function_probability,
}

Genome.mutation_names = {
    Genome.mutate_add_node                      : "add node",
    Genome.mutate_remove_node                   : "remove node",
    Genome.mutate_perturb_bias                  : "perturb bias",
    Genome.mutate_add_edge                      : "add edge",
    Genome.mutate_remove_edge                   : "remove edge",
    Genome.mutate_perturb_weight                : "perturb weight",
    Genome.mutate_change_aggregation_function   : "change aggregation function",
    Genome.mutate_change_activation_function    : "change activation function",
}