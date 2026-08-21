import numpy as np


def normal_distribution():

    values = np.random.normal(
        loc=0,
        scale=1,
        size=1000
    )

    return values



def bayes_theorem(prior, likelihood, evidence):

    posterior = (likelihood * prior) / evidence

    return posterior