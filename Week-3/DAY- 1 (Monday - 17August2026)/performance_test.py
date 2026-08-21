import time
import numpy as np


def python_loop():

    numbers = list(range(1000000))

    start = time.time()

    squares = []

    for i in numbers:
        squares.append(i * i)

    end = time.time()

    return end - start


def numpy_loop():

    numbers = np.arange(1000000)

    start = time.time()

    squares = numbers ** 2

    end = time.time()

    return end - start