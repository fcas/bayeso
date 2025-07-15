#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 22, 2024
#
"""It defines a pure exploration acquisition function."""

import numpy as np

from bayeso.utils import utils_common


@utils_common.validate_types
def acq_fun(pred_std: np.ndarray) -> np.ndarray:
    """
    It is a pure exploration criterion.

    :param pred_std: posterior predictive standard deviation function over `X_test`. Shape: (l, ).
    :type pred_std: numpy.ndarray

    :returns: acquisition function values. Shape: (l, ).
    :rtype: numpy.ndarray

    :raises: AssertionError

    """

    assert isinstance(pred_std, np.ndarray)
    assert pred_std.ndim == 1

    return pred_std
