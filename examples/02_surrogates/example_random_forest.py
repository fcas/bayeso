#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np
import time

from bayeso.trees import trees_random_forest
from bayeso.trees import trees_common


def main():
    time_start = time.time()

    np.random.seed(42)
    X_train = np.array(
        [
            [-3.0],
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
            [4.0],
        ]
    )
    Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.2
    num_test = 10000
    X_test = np.linspace(-5, 5, num_test)
    X_test = X_test.reshape((num_test, 1))
    Y_test = np.cos(X_test)

    num_trees = 100
    depth_max = 5
    size_min_leaf = 2
    num_features = 1

    trees = trees_random_forest.get_random_forest(
        X_train, Y_train, num_trees, depth_max, size_min_leaf, num_features
    )

    mu, sigma = trees_common.predict_by_trees(X_test, trees)

    time_end = time.time()
    print("time consumed: {:.4f}".format(time_end - time_start))
    print(mu.shape, sigma.shape)


if __name__ == "__main__":
    main()
