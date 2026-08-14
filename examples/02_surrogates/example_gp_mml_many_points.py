#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np

from bayeso.gp import gp


def main():
    num_train = 200
    num_test = 1000
    X_train = np.random.randn(num_train, 1) * 5.0
    Y_train = np.cos(X_train) + 10.0
    X_test = np.linspace(-10, 10, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) + 10.0

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(
        X_train, Y_train, X_test, debug=True
    )

    print(mu.shape, sigma.shape, Sigma.shape)


if __name__ == "__main__":
    main()
