import numpy as np


def function(x):
    return x**2 + 2*x + 1


def derivative(x):
    return 2*x + 2


def gradient(points):

    gradients = []

    for x in points:
        gradients.append(derivative(x))

    return gradients