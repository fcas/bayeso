#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np

from bayeso.tp import tp


list_str_covs = ["se", "eq", "matern32", "matern52"]


def main(str_cov):
    np.random.seed(42)
    X_train = np.array(
        [
            [-3.0],
            [-1.0],
            [3.0],
            [1.0],
            [2.0],
        ]
    )
    Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.2
    num_test = 200
    X_test = np.linspace(-3, 3, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test)

    nu, mu, sigma, Sigma = tp.predict_with_optimized_hyps(
        X_train, Y_train, X_test, str_cov=str_cov, fix_noise=False, debug=True
    )

    print(str_cov, nu, mu.shape, sigma.shape, Sigma.shape)


if __name__ == "__main__":
    for str_cov in list_str_covs:
        main(str_cov)
