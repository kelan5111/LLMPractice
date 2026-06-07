import numpy as np


def get_loss():
    return 0


def get_curr_parameters():
    # ŷ = wx + b
    w = np.random.randint(-0.1, 0.1)  # Initial weight value
    loss = get_loss()
    gradient = get_gradient()

    return w, loss, gradient


def get_gradient():
    return 0


def gradient_desc():
    # Optimization technique

    learning_rate = 0.01
    w, loss, gradient = get_curr_parameters()

    while loss > 0.01:
        w -= (learning_rate * gradient)

    return w


def main():
    gradient_desc()


if __name__ == "__main__":
    main()
