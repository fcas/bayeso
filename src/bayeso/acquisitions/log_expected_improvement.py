#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: July 15, 2025
#
"""It defines a log expected improvement acquisition function.

(i) Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E.
(2023). Unexpected Improvements to Expected Improvement for Bayesian
Optimization. In Advances in Neural Information Processing Systems,
36, pp. 20577--20612."""

import numpy as np
import scipy.stats as scist
import scipy.special as scisp

from bayeso import constants
from bayeso.utils import utils_common


def log1mexp(x):
    return np.log(-np.expm1(x))


def erfcx(x):
    return scisp.erfcx(x)


@utils_common.validate_types
def acq_fun(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    Y_train: np.ndarray,
    jitter: float = constants.JITTER_ACQ,
) -> np.ndarray:
    """
    It is a log expected improvement criterion.

    :param pred_mean: posterior predictive mean function over `X_test`. Shape: (l, ).
    :type pred_mean: numpy.ndarray
    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
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

    eps = 1e-3
    pred_std = np.maximum(pred_std, jitter)
    val_z = (np.min(Y_train) - pred_mean) / pred_std

    c_1 = np.log(2 * np.pi) / 2
    c_2 = np.log(np.pi / 2) / 2

    term_first = np.zeros_like(val_z)

    indices = -1 < val_z
    term_first[indices] = np.log(
        scist.norm.pdf(val_z[indices]) + val_z[indices] * scist.norm.cdf(val_z[indices])
    )

    indices = np.logical_and(-1 / np.sqrt(eps) < val_z, val_z <= -1)
    term_first[indices] = -val_z[indices]**2 / 2 - c_1 \
        + log1mexp(np.log(erfcx(-val_z[indices] / np.sqrt(2)) * np.abs(val_z[indices])) + c_2)

    indices = val_z <= -1 / np.sqrt(eps)
    term_first[indices] = -val_z[indices]**2 / 2 - c_1 - 2 * np.log(np.abs(val_z[indices]))

    term_second = np.log(pred_std)

    return term_first + term_second
