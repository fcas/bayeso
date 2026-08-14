#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: July 16, 2025
#
"""test_acquisitions_log_expected_improvement"""

import typing
import pytest
import numpy as np
import scipy.special as scisp

from bayeso.acquisitions import log_expected_improvement as package_target
from bayeso.acquisitions import expected_improvement


def test_expm1():
    def fun_expm1(x):
        return np.exp(x) - 1

    np.testing.assert_allclose(np.expm1(-100.0), fun_expm1(-100.0))
    np.testing.assert_allclose(np.expm1(-10.0), fun_expm1(-10.0))
    np.testing.assert_allclose(np.expm1(-1.0), fun_expm1(-1.0))
    np.testing.assert_allclose(np.expm1(0.0), fun_expm1(0.0))
    np.testing.assert_allclose(np.expm1(1.0), fun_expm1(1.0))
    np.testing.assert_allclose(np.expm1(10.0), fun_expm1(10.0))
    np.testing.assert_allclose(np.expm1(100.0), fun_expm1(100.0))


def test_log1mexp():
    def fun_log1mexp(x):
        return np.log(1 - np.exp(x))

    np.testing.assert_allclose(package_target.log1mexp(-1000.0), fun_log1mexp(-1000.0))
    np.testing.assert_allclose(package_target.log1mexp(-100.0), fun_log1mexp(-100.0))
    np.testing.assert_allclose(package_target.log1mexp(-10.0), fun_log1mexp(-10.0))
    np.testing.assert_allclose(package_target.log1mexp(-1.0), fun_log1mexp(-1.0))


def test_erfcx():
    def fun_erfcx_1(x):
        return np.exp(x**2) * (1 - scisp.erf(x))

    def fun_erfcx_2(x):
        return np.exp(x**2) * scisp.erfc(x)

    np.testing.assert_allclose(package_target.erfcx(-10.0), fun_erfcx_1(-10.0))
    np.testing.assert_allclose(package_target.erfcx(-1.0), fun_erfcx_1(-1.0))
    np.testing.assert_allclose(package_target.erfcx(0.0), fun_erfcx_1(0.0))
    np.testing.assert_allclose(package_target.erfcx(1.0), fun_erfcx_1(1.0))
    np.testing.assert_allclose(package_target.erfcx(2.0), fun_erfcx_1(2.0))

    np.testing.assert_allclose(package_target.erfcx(-10.0), fun_erfcx_2(-10.0))
    np.testing.assert_allclose(package_target.erfcx(-1.0), fun_erfcx_2(-1.0))
    np.testing.assert_allclose(package_target.erfcx(0.0), fun_erfcx_2(0.0))
    np.testing.assert_allclose(package_target.erfcx(1.0), fun_erfcx_2(1.0))
    np.testing.assert_allclose(package_target.erfcx(2.0), fun_erfcx_2(2.0))


def test_acq_fun():
    random_state = np.random.RandomState(42)

    num_data = 10

    pred_mean = random_state.randn(num_data)
    pred_std = random_state.uniform(size=num_data)
    Y_train = random_state.randn(4, 1)

    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(pred_mean, pred_std, "abc")
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(pred_mean, "abc", Y_train)
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun("abc", pred_std, Y_train)
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(pred_mean, pred_std, Y_train, jitter="abc")
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(pred_mean, pred_std, random_state.randn(10))
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(pred_mean, random_state.uniform(size=4), Y_train)
    with pytest.raises(AssertionError) as error:
        package_target.acq_fun(random_state.randn(4), pred_std, Y_train)

    val_acq = package_target.acq_fun(pred_mean, pred_std, Y_train)
    truth_val_acq = np.array(
        [
            -63.20290871,
            -31.16940939,
            -29.20683632,
            -20.87946532,
            -7.87453097,
            -13.47734946,
            -16.75476422,
            -131.89468679,
            -10.08565436,
            -19.79594326,
        ]
    )
    print(val_acq)

    assert val_acq.ndim == 1
    assert val_acq.shape[0] == num_data

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_compare_log_ei_to_ei():
    random_state = np.random.RandomState(42)

    num_data = 100

    pred_mean = random_state.randn(num_data)
    pred_std = random_state.uniform(size=num_data)
    Y_train = random_state.randn(10, 1)

    val_acq_log_ei = package_target.acq_fun(pred_mean, pred_std, Y_train)
    val_acq_ei = expected_improvement.acq_fun(pred_mean, pred_std, Y_train)

    count = 0
    for elem_1, elem_2 in zip(np.exp(val_acq_log_ei), val_acq_ei):
        if not np.allclose(elem_1, elem_2, rtol=1e-07, atol=0):
            print(elem_1, elem_2)
            count += 1

    print("")
    print(f"count {count}")
    np.testing.assert_allclose(np.exp(val_acq_log_ei), val_acq_ei)
