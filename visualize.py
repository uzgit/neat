import graphviz
import matplotlib.pyplot as plt

def draw_neural_network2(network):

    top_level_graph = graphviz.Digraph(format="svg")

    subgraphs = []

    for layer in range(1, network.num_layers + 1):

        subgraph = graphviz.Digraph("layer {}".format(layer))

        if layer == 1:
            subgraph.graph_attr.update(rank="min")
        elif layer == network.num_layers:
            subgraph.graph_attr.update(rank="max")
        else:
            subgraph.graph_attr.update(rank="same")

        for node in [node for node in network.nodes if node.layer == layer]:
            subgraph.node(str(node.identifier))

        subgraphs.append(subgraph)

    # draw enabled edges
    for edge in [edge for edge in network.genome.edges if edge.is_enabled]:

        style = "solid"
        color = "black" if edge.weight > 0 else "grey"
        width = str(0.1 + abs(edge.weight / 5.0))

        top_level_graph.edge(str(edge.input_node_identifier), str(edge.output_node_identifier), _attributes={"style" : style, "color" : color, "penwidth" : width})

    for subgraph in subgraphs:
        top_level_graph.subgraph(subgraph)

    top_level_graph.render(view=True)

def draw_neural_network(network):

    graph = graphviz.Digraph(format="svg")
    input_node_subgraph = graphviz.Digraph("input nodes")
    input_node_subgraph.graph_attr.update(rank="min")

    hidden_node_subgraph = graphviz.Digraph("hidden nodes")
    hidden_node_subgraph.graph_attr.update(rank="same")

    output_node_subgraph = graphviz.Digraph("output nodes")
    output_node_subgraph.graph_attr.update(rank="max")

    for input_node in network.input_nodes:
        print(input_node.identifier)
        input_node_subgraph.node(str(input_node.identifier), color="red")

    for hidden_node in network.hidden_nodes:
        hidden_node_subgraph.node(str(hidden_node.identifier), color="yellow")

    for output_node in network.output_nodes:
        output_node_subgraph.node(str(output_node.identifier), color="red")

    # draw enabled edges
    for edge in [edge for edge in network.genome.edges if edge.is_enabled]:

        style = "solid"
        color = "black" if edge.weight > 0 else "grey"
        width = str(0.1 + abs(edge.weight / 5.0))

        graph.edge(str(edge.input_node_identifier), str(edge.output_node_identifier), _attributes={"style" : style, "color" : color, "penwidth" : width, "label" : str(edge.weight)})

    graph.subgraph(input_node_subgraph)
    graph.subgraph(output_node_subgraph)

    graph.render(view=True)

def draw_neural_network1(network):

    node_attributes = \
    {
        "shape" : "circle",
        "fontsize" : '9',
        "height" : "0.2",
        "width" : "0.2",
        #"rank" : "same",
    }

    image = graphviz.Digraph(format="svg", node_attr=node_attributes)

    # subgraphs = []
    #
    # for layer in range(1, network.num_layers + 1):
    #
    #     current_layer_nodes = [node for node in network.nodes if node.layer == layer]
    #     print([node.identifier for node in current_layer_nodes])
    #     for node in current_layer_nodes:
    #         subgraph = graphviz.Digraph("layer {}".format(layer))
    #         #subgraph.graph_attr.update(rank="max", rankdir="LR")
    #         subgraph.node(str(node.identifier), _attributes=node_attributes)
    #
    #         #image.node(str(node.identifier))
    #
    #     subgraphs.append(subgraph)

    # # draw nodes
    # for node in network.nodes:
    #     single_node_attributes = \
    #         {
    #             "shape": "circle",
    #             "fontsize": '9',
    #             "height": "0.2",
    #             "width": "0.2",
    #             "rank" : "same",
    #         }
    #
    #     image.node(str(node.identifier), _attributes=single_node_attributes)
    #
    # for node in network.hidden_nodes:
    #     image.node(str(node.identifier), _attributes=node_attributes)
    #
    # for node in network.output_nodes:
    #     image.node(str(node.identifier), _attributes=node_attributes)

    # draw enabled edges
    for edge in [edge for edge in network.genome.edges if edge.is_enabled]:

        style = "solid"
        color = "black" if edge.weight > 0 else "grey"
        width = str(0.1 + abs(edge.weight / 5.0))

        image.edge(str(edge.input_node_identifier), str(edge.output_node_identifier))#, _attributes={"style" : style, "color" : color, "penwidth" : width})

    # for subgraph in subgraphs:
    #     subgraph.graph_attr.update(rank="same")
    #     image.subgraph(subgraph)

    image.render(view=True)