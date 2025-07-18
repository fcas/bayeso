#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 22, 2024
#
"""It defines a Gaussian process upper confidence bound acquisition function.

(i) Srinivas, N., Krause, A., Kakade, S. M. & Seeger, M. (2010). Gaussian\
Process Optimization in the Bandit Setting: No Regret and Experimental Design.\
In Proceedings of the 27th International Conference on International Conference\
on Machine Learning, pp. 1015--1022."""

import typing
import numpy as np

from bayeso.utils import utils_common


@utils_common.validate_types
def acq_fun(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    Y_train: typing.Optional[np.ndarray] = None,
    kappa: float = 2.0,
    increase_kappa: bool = True,
) -> np.ndarray:
    """
    It is a Gaussian process upper confidence bound criterion.

    :param pred_mean: posterior predictive mean function over `X_test`.
        Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over
        `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray, optional
    :param kappa: trade-off hyperparameter between exploration and
        exploitation.
    :type kappa: float, optional
    :param increase_kappa: flag for increasing a kappa value as `Y_train`
        grows. If `Y_train` is None, it is ignored, which means `kappa` is
        fixed.
    :type increase_kappa: bool., optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, (np.ndarray, type(None)))
    assert isinstance(kappa, float)
    assert isinstance(increase_kappa, bool)
    assert pred_mean.ndim == 1
    assert pred_std.ndim == 1
    if Y_train is not None:
        assert Y_train.ndim == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    if increase_kappa and Y_train is not None:
        kappa_ = kappa * np.log(Y_train.shape[0] + 1.0)
    else:
        kappa_ = kappa
    return -pred_mean + kappa_ * pred_std
