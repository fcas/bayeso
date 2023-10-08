#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: August 20, 2023
#

import numpy as np
import os

from bayeso import bo
from bayeso.utils import utils_plotting


def fun_target(X):
    return 4.0 * np.cos(X) + 0.1 * X + 2.0 * np.sin(X) + 0.4 * (X - 0.5)**2

path_save = None

if path_save is not None and not os.path.isdir(path_save):
    os.makedirs(path_save)

str_acq = 'aei'
num_iter = 10
X_train = np.array([
    [-5],
    [-1],
    [1],
    [2],
])
num_init = X_train.shape[0]

model_bo = bo.BO(np.array([[-6., 6.]]), str_acq=str_acq, normalize_Y=False)

X_test = np.linspace(-6, 6, 400)
X_test = X_test[..., np.newaxis]

for ind_ in range(1, num_iter + 1):
    Y_train = fun_target(X_train)
    next_x, dict_info = model_bo.optimize(X_train, fun_target(X_train), str_sampling_method='uniform', seed=ind_)

    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']

    mu_test, sigma_test = model_bo.compute_posteriors(X_train, Y_train, X_test, cov_X_X, inv_cov_X_X, hyps)
    acq_test = model_bo.compute_acquisitions(X_test, X_train, Y_train, cov_X_X, inv_cov_X_X, hyps)

    mu_test = np.expand_dims(mu_test, axis=1)
    sigma_test = np.expand_dims(sigma_test, axis=1)
    acq_test = np.expand_dims(acq_test, axis=1)

    X_train = np.vstack((X_train, next_x))
    Y_train = fun_target(X_train)

    str_postfix = f'bo_{str_acq}_{ind_}' if path_save is not None else None

    utils_plotting.plot_bo_step(X_train, Y_train, X_test, fun_target(X_test), mu_test, sigma_test, path_save=path_save, str_postfix=str_postfix, num_init=num_init)
    utils_plotting.plot_bo_step_with_acq(X_train, Y_train, X_test, fun_target(X_test), mu_test, sigma_test, acq_test, path_save=path_save, str_postfix=str_postfix, num_init=num_init)
