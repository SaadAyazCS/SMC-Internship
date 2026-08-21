def gradient_descent(
        learning_rate,
        iterations
):

    x = 10

    history = []


    for i in range(iterations):

        gradient = 2*x


        x = x - learning_rate * gradient


        history.append(x)


    return history