#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 20, 2024
#

import numpy as np

from bayeso.bo import bo_w_gp


def fun_target(X):
    return 4.0 * np.cos(X) + 0.1 * X + 2.0 * np.sin(X) + 0.4 * (X - 0.5) ** 2


str_acq = "ei"
num_iter = 10
debug = True
X_train = np.array(
    [
        [-5],
        [-1],
        [1],
        [2],
    ]
)
Y_train = fun_target(X_train)
num_init = X_train.shape[0]

model_bo = bo_w_gp.BOwGP(
    np.array([[-6.0, 6.0]]), str_acq=str_acq, normalize_Y=False, debug=debug
)
X_test = np.linspace(-6, 6, 400)
X_test = np.reshape(X_test, (400, 1))

for ind_ in range(1, num_iter + 1):
    next_x, dict_info = model_bo.optimize(
        X_train, fun_target(X_train), str_sampling_method="uniform"
    )
    cov_X_X = dict_info["cov_X_X"]
    inv_cov_X_X = dict_info["inv_cov_X_X"]
    hyps = dict_info["hyps"]

    mu_test, sigma_test = model_bo.compute_posteriors(
        X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps
    )
    acq_test = model_bo.compute_acquisitions(
        X_test, X_train, Y_train, cov_X_X, inv_cov_X_X, hyps
    )

    mu_test = np.expand_dims(mu_test, axis=1)
    sigma_test = np.expand_dims(sigma_test, axis=1)
    acq_test = np.expand_dims(acq_test, axis=1)

    X_train = np.vstack((X_train, next_x))
    Y_train = fun_target(X_train)

Y_train = np.squeeze(Y_train)
Y_train = np.array([[Y_train]])

print(X_train.shape, Y_train.shape)
