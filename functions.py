import math
import numpy

# aggregation_functions



# sum is already defined
# min is already defined
# max is already defined

# activation functions



def arctan(x):

    return math.atan(x)

def average(inputs):

    return sum(inputs) / len(inputs)

def binary_step(x):

    result = 0
    if x >= 0:
        result = 1

    return result

def identity(x):

    return x

def lelu(x):

    leaky = 0.005
    return x if x > 0.0 else leaky * x

def relu(x):

    result = 0
    if x >= 0:
        result = x

    return result

def sigmoid(x):

    # We get an overflow error if x < 709 (not 708, but still clipped to -708 min just in case)
    x = max(x, -708)
    return 1 / (1 + math.exp(-x))

def softplus(x):

    x = min(x, 708)
    return math.log(1 + (math.e ** x))

def sign(x):

    return 1 if x >= 0 else -1

def closeness(x, y, interval_minimum, interval_maximum):

    return 1 - abs((x-y)/(interval_maximum - interval_minimum))

def step(x):

    return 0 if x < 0.5 else 1

def tanh(x):

    x = max(-353, x)
    return (2.0 / (1 + (math.e ** (-2 * x)))) - 1

aggregation_functions = [sum, min, max, average]
activation_functions = [arctan, binary_step, identity, lelu, relu, sigmoid, softplus, step, tanh]

function_names = {
    arctan : "arctan",
    average : "average",
    binary_step : "binary_step",
    identity : "identity",
    lelu : "lelu",
    min : "min",
    max : "max",
    relu : "relu",
    sigmoid : "sigmoid",
    softplus : "softplus",
    step : "step",
    sum : "sum",
    tanh : "tanh",
}

function_names = {
    arctan : "atan",
    average : "ave",
    binary_step : "step",
    identity : "id",
    lelu : "lelu",
    min : "min",
    max : "max",
    relu : "relu",
    sigmoid : "sig",
    softplus : "splus",
    step : "step",
    sum : "sum",
    tanh : "tanh",
}