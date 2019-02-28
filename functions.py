import math
import numpy

# aggregation_functions

#sum is already defined

# activation functions
def arctan(x):
    return math.atan(x)

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

def logistic(x):

    # We get an overflow error if x < 709 (not 708, but still clipped to -708 min just in case)
    x = max(x, -708)
    return 1.0 / (1 + (math.e ** (-x)))

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

def step(x):
    return 0 if x < 0.5 else 1

def tanh(x):

    x = max(-353, x)
    return (2.0 / (1 + (math.e ** (-2 * x)))) - 1

aggregation_functions = {
    "sum" : sum,
    "min" : min,
    "max" : max,
}

activation_functions = {
    "arctan": arctan,
    "binary_step": binary_step,
    "identity": identity,
    "lelu" : lelu,
    "logistic": logistic,
    "relu": relu,
    "sigmoid" : sigmoid,
    "softplus": softplus,
    "step" : step,
    "tanh": tanh,
}

aggregation_function_names = list(aggregation_functions.keys())
activation_function_names = list(activation_functions.keys())

# print(activation_function_names)

# print( activation_functions["binary_step"](5) )