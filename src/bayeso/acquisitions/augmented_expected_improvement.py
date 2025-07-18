#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 22, 2024
#
"""It defines an augmented expected improvement acquisition function.

(i) Huang, D., Allen, T. T., Notz, W. I., & Zeng, N. (2006). Global \
Optimization of Stochastic Black-Box Systems via Sequential Kriging \
Meta-Models. *Journal of Global Optimization*, 34(3), pp. 441--466."""

import numpy as np
import scipy.stats as scis

from bayeso import constants
from bayeso.utils import utils_common


@utils_common.validate_types
def acq_fun(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    Y_train: np.ndarray,
    noise: float,
    jitter: float = constants.JITTER_ACQ,
) -> np.ndarray:
    """
    It is an augmented expected improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`. Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param noise: noise for augmenting exploration.
    :type noise: float
    :param jitter: jitter for `pred_std`.
    :type jitter: float, optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(noise, float)
    assert isinstance(jitter, float)
    assert pred_mean.ndim == 1
    assert pred_std.ndim == 1
    assert Y_train.ndim == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    pred_std = np.maximum(pred_std, jitter)
    val_z = (np.min(Y_train) - pred_mean) / pred_std

    val_ei = (np.min(Y_train) - pred_mean) \
        * scis.norm.cdf(val_z) + pred_std * scis.norm.pdf(val_z)
    val_aei = val_ei * (1.0 - noise / np.sqrt(pred_std**2 + noise**2))

    return val_aei
