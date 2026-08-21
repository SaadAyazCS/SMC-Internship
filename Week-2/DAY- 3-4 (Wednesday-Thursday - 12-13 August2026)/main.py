from derivatives import gradient
from probability import (
    normal_distribution,
    bayes_theorem
)

from entropy import entropy

from visualization import (
    plot_derivative,
    plot_gradient_descent
)



print("========== DERIVATIVES ==========")


points = [-3,-2,-1,0,1,2,3]

print(
    "Gradients:",
    gradient(points)
)



print("\n========== PROBABILITY ==========")


data = normal_distribution()

print(
    "Sample Probability Data:",
    data[:10]
)



posterior = bayes_theorem(
    prior=0.5,
    likelihood=0.8,
    evidence=0.6
)


print(
    "Bayes Posterior:",
    posterior
)



print("\n========== ENTROPY ==========")


print(
    "Entropy:",
    entropy(
        [0.5,0.5]
    )
)



print("\n========== VISUALIZATION ==========")


plot_derivative()

plot_gradient_descent()