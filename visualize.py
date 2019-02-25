import graphviz

def draw_neural_network(network, filename=None):

    input_node_attributes = \
    {
        "shape" : "circle",
        "fontsize" : '12',
        "height" : "0.2",
        "width" : "0.2",
        "color" : "red",
    }

    hidden_node_attributes = \
    {
        "shape" : "circle",
        "fontsize" : '12',
        "height" : "0.2",
        "width" : "0.2",
    }

    output_node_attributes = \
    {
        "shape" : "circle",
        "fontsize" : '12',
        "height" : "0.2",
        "width" : "0.2",
        "color" : "red",
    }

    graph = graphviz.Digraph(format="svg")
    input_node_subgraph = graphviz.Digraph("input nodes")
    input_node_subgraph.graph_attr.update(rank="min")

    hidden_node_subgraph = graphviz.Digraph("hidden nodes")
    # hidden_node_subgraph.graph_attr.update(rank="same")

    output_node_subgraph = graphviz.Digraph("output nodes")
    output_node_subgraph.graph_attr.update(rank="max")

    for input_node in network.input_nodes:
        input_node_subgraph.node(str(input_node.identifier), _attributes=input_node_attributes)
        # input_node_subgraph.node_attr.

    for hidden_node in network.hidden_nodes:
        hidden_node_subgraph.node(str(hidden_node.identifier), _attributes=hidden_node_attributes)

    for output_node in network.output_nodes:
        output_node_subgraph.node(str(output_node.identifier), _attributes=output_node_attributes)

    # draw enabled edges
    for edge in network.edges:

        style = "solid"
        color = "black" if edge.weight > 0 else "red"
        width = str(0.1 + abs(edge.weight / 5.0))

        graph.edge(str(edge.input_node_identifier), str(edge.output_node_identifier),
                   _attributes={"style": style, "color": color, "label": str("%0.2f" % edge.weight)})

    graph.subgraph(input_node_subgraph)
    graph.subgraph(hidden_node_subgraph)
    graph.subgraph(output_node_subgraph)

    graph.render(view=True, filename=filename)