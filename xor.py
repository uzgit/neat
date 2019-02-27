def xor(inputs):

    result = None

    if inputs[0] == 0 and inputs[1] == 0:
        result = [0]
    elif inputs[0] == 0 and inputs[1] == 1:
        result = [1]
    elif inputs[0] == 1 and inputs[1] == 0:
        result = [1]
    elif inputs[0] == 1 and inputs[1] == 1:
        result = [0]

    return result

def test_xor(neural_network):

    fitness = 0

    for i in range(4):

        input_1 = 1 if i > 1 else 0
        input_2 = i % 2

        inputs = [input_1, input_2]

        network_output = neural_network.activate(inputs)
        true_output = xor(inputs)

        if network_output == true_output:
            fitness += 1

    neural_network.genome.fitness = fitness

def test_xor_print(neural_network):

    fitness = 0

    for i in range(4):

        input_1 = 1 if i > 1 else 0
        input_2 = i % 2

        inputs = [input_1, input_2]

        network_output = neural_network.activate(inputs)
        true_output = xor(inputs)

        print(inputs, "->", network_output)

        if network_output == true_output:
            fitness += 1

    return fitness