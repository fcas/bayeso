#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 18, 2024
#

import numpy as np

from bayeso.gp import gp


np.random.seed(42)

X_train = np.array(
    [
        [-3],
        [-1],
        [1],
        [2],
    ]
)
Y_train = np.cos(X_train) + np.random.randn(X_train.shape[0], 1) * 0.1
num_test = 200

X_test = np.linspace(-3, 3, num_test)
X_test = X_test.reshape((num_test, 1))
Y_test = np.cos(X_test)
hyps = {
    "signal": 0.5,
    "lengthscales": 0.5,
    "noise": 0.02,
}
mu, sigma, Sigma = gp.predict_with_hyps(X_train, Y_train, X_test, hyps)

print(mu.shape, sigma.shape, Sigma.shape)
