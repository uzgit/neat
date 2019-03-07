import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from population import *
import numpy
import matplotlib.pyplot as plt

portion_elites = 0.25
regression_degree = 2

run_data = Population.load_run_data(sys.argv[1])

generation_average_fitnesses = []
generation_elite_average_fitnesses = []
generation_best_fitnesses = []

for generation_list in run_data:

    generation_list.sort(reverse=True)

    generation_average_fitnesses.append( sum(generation_list) / len(generation_list) )

    num_elites = int(portion_elites * len(generation_list))
    generation_elite_average_fitnesses.append( sum(generation_list[:num_elites]) / num_elites )

    generation_best_fitnesses.append( max(generation_list) )

list_for_regression = generation_elite_average_fitnesses
regression_function = numpy.poly1d( numpy.polyfit(x=range(1, len(list_for_regression) + 1), y=list_for_regression, deg=regression_degree) )
regression_points = regression_function( range(len(run_data)) )

plt.plot(generation_best_fitnesses, label="Best Fitness")
plt.plot(generation_elite_average_fitnesses, label="Elite (Top {}%) Average Fitness".format(portion_elites * 100))
plt.plot(regression_points, label="Elite Fitness Regression (Degree {})".format(regression_degree), linestyle="dashed")
plt.plot(generation_average_fitnesses, label="Average Fitness")

plt.legend(loc="upper left")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()
