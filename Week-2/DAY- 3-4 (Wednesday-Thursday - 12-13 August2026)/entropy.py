import math


def entropy(probabilities):

    result = 0

    for p in probabilities:

        if p > 0:
            result -= p * math.log2(p)

    return result