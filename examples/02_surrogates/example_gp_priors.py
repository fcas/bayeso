#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np

from bayeso.gp import gp


def cosine(X):
    return np.cos(X)


def linear_down(X):
    list_up = []
    for elem_X in X:
        list_up.append([-0.5 * np.sum(elem_X)])
    return np.array(list_up)


def linear_up(X):
    list_up = []
    for elem_X in X:
        list_up.append([0.5 * np.sum(elem_X)])
    return np.array(list_up)


def main(fun_prior, str_prior):
    X_train = np.array(
        [
            [-3.0],
            [-2.0],
            [-1.0],
        ]
    )
    Y_train = np.cos(X_train) + 2.0
    num_test = 200
    X_test = np.linspace(-3, 6, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 2.0

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(
        X_train, Y_train, X_test, prior_mu=fun_prior
    )

    print(str_prior, mu.shape, sigma.shape, Sigma.shape)


if __name__ == "__main__":
    main(cosine, "cosine")
    main(linear_down, "linear_down")
    main(linear_up, "linear_up")
