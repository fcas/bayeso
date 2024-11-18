#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np

from bayeso.gp import gp


def main(scale):
    X_train = np.array(
        [
            [-3.0],
            [-2.0],
            [-1.0],
            [2.0],
            [1.2],
            [1.1],
        ]
    )
    Y_train = np.cos(X_train) * scale
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test) * scale

    mu, sigma, Sigma = gp.predict_with_optimized_hyps(
        X_train, Y_train, X_test, fix_noise=False, debug=True
    )

    print(scale, mu.shape, sigma.shape, Sigma.shape)


if __name__ == "__main__":
    main(0.01)
    main(100000.0)
