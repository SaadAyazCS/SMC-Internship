import matplotlib.pyplot as plt

from derivatives import function
from gradient_descent import gradient_descent



def plot_derivative():

    x_values = []

    y_values = []


    for x in range(-10,11):

        x_values.append(x)
        y_values.append(function(x))


    plt.plot(
        x_values,
        y_values
    )

    plt.title(
        "Derivative Visualization"
    )

    plt.xlabel("X")
    plt.ylabel("f(x)")


    plt.show()



def plot_gradient_descent():

    values = gradient_descent(
        0.1,
        20
    )


    plt.plot(values)


    plt.title(
        "Gradient Descent Optimization"
    )

    plt.xlabel(
        "Iterations"
    )

    plt.ylabel(
        "Value of X"
    )


    plt.show()