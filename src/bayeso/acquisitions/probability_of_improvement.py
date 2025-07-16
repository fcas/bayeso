#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 22, 2024
#
"""It defines a probability of improvement acquisition function."""

import numpy as np
import scipy.stats as scis

from bayeso import constants
from bayeso.utils import utils_common


@utils_common.validate_types
def acq_fun(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    Y_train: np.ndarray,
    jitter: float = constants.JITTER_ACQ,
) -> np.ndarray:
    """
    It is a probability of improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`.
        Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over
        `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray
    :param Y_train: outputs of `X_train`. Shape: (n, 1).
    :type Y_train: numpy.ndarray
    :param jitter: jitter for `pred_std`.
    :type jitter: float, optional

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_mean, np.ndarray)
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(Y_train, np.ndarray)
    assert isinstance(jitter, float)
    assert pred_mean.ndim == 1
    assert pred_std.ndim == 1
    assert Y_train.ndim == 2
    assert pred_mean.shape[0] == pred_std.shape[0]

    pred_std = np.maximum(pred_std, jitter)
    val_z = (np.min(Y_train) - pred_mean) / pred_std

    return scis.norm.cdf(val_z)
