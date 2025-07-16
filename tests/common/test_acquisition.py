#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: June 21, 2024
#
"""test_acquisition"""

import typing
import pytest
import numpy as np

from bayeso import acquisition as package_target


def test_pi_typing():
    annos = package_target.pi.__annotations__

    assert annos["pred_mean"] == np.ndarray
    assert annos["pred_std"] == np.ndarray
    assert annos["Y_train"] == np.ndarray
    assert annos["jitter"] == float
    assert annos["return"] == np.ndarray


def test_pi():
    with pytest.raises(AssertionError) as error:
        package_target.pi("abc", np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(10), "abc", np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(10), np.ones(10), "abc")
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.pi(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

    val_acq = package_target.pi(np.arange(0, 10), np.ones(10), np.zeros((5, 1)))
    truth_val_acq = np.array(
        [
            5.00000000e-01,
            1.58655254e-01,
            2.27501319e-02,
            1.34989803e-03,
            3.16712418e-05,
            2.86651572e-07,
            9.86587645e-10,
            1.27981254e-12,
            6.22096057e-16,
            1.12858841e-19,
        ]
    )
    print(val_acq)

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_ei_typing():
    annos = package_target.ei.__annotations__

    assert annos["pred_mean"] == np.ndarray
    assert annos["pred_std"] == np.ndarray
    assert annos["Y_train"] == np.ndarray
    assert annos["jitter"] == float
    assert annos["return"] == np.ndarray


def test_ei():
    with pytest.raises(AssertionError) as error:
        package_target.ei("abc", np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(10), "abc", np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(10), np.ones(10), "abc")
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(10), np.ones(10), np.zeros((5, 1)), 1)
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(5), np.ones(10), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(10), np.ones(10), np.zeros(5))
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.ei(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)))

    val_acq = package_target.ei(np.arange(0, 10), np.ones(10), np.zeros((5, 1)))
    truth_val_acq = np.array(
        [
            3.98942280e-01,
            8.33154706e-02,
            8.49070262e-03,
            3.82154317e-04,
            7.14525843e-06,
            5.34616553e-08,
            1.56356980e-10,
            1.76032601e-13,
            7.55026241e-17,
            1.22477918e-20,
        ]
    )
    print(val_acq)

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_ucb_typing():
    annos = package_target.ucb.__annotations__

    assert annos["pred_mean"] == np.ndarray
    assert annos["pred_std"] == np.ndarray
    assert annos["Y_train"] == typing.Union[np.ndarray, type(None)]
    assert annos["kappa"] == float
    assert annos["increase_kappa"] == bool
    assert annos["return"] == np.ndarray


def test_ucb():
    with pytest.raises(AssertionError) as error:
        package_target.ucb("abc", np.ones(10))
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), "abc")
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), np.ones(10), kappa="abc")
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(5), np.ones(10))
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), np.ones(10), Y_train="abc")
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), np.ones(10), Y_train=np.zeros(5))
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), np.ones((10, 1)))
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones((10, 1)), np.ones(10))
    with pytest.raises(AssertionError) as error:
        package_target.ucb(np.ones(10), np.ones(10), increase_kappa="abc")

    val_acq = package_target.ucb(
        np.arange(0, 10), np.ones(10), Y_train=np.zeros((5, 1))
    )
    truth_val_acq = np.array(
        [
            3.58351894,
            2.58351894,
            1.58351894,
            0.58351894,
            -0.41648106,
            -1.41648106,
            -2.41648106,
            -3.41648106,
            -4.41648106,
            -5.41648106,
        ]
    )

    np.testing.assert_allclose(val_acq, truth_val_acq)

    val_acq = package_target.ucb(np.arange(0, 10), np.ones(10))
    truth_val_acq = np.array([2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_aei_typing():
    annos = package_target.aei.__annotations__

    assert annos["pred_mean"] == np.ndarray
    assert annos["pred_std"] == np.ndarray
    assert annos["Y_train"] == np.ndarray
    assert annos["jitter"] == float
    assert annos["return"] == np.ndarray


def test_aei():
    with pytest.raises(AssertionError) as error:
        package_target.aei("abc", np.ones(10), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), "abc", np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), np.ones(10), "abc", 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), np.ones(10), np.zeros((5, 1)), "abc")
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), np.ones(10), np.zeros((5, 1)), 1.0, jitter=1)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(5), np.ones(10), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), np.ones(10), np.zeros(5), 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones(10), np.ones((10, 1)), np.zeros((5, 1)), 1.0)
    with pytest.raises(AssertionError) as error:
        package_target.aei(np.ones((10, 1)), np.ones(10), np.zeros((5, 1)), 1.0)

    val_acq = package_target.aei(np.arange(0, 10), np.ones(10), np.zeros((5, 1)), 1.0)
    truth_val_acq = np.array(
        [
            1.16847489e-01,
            2.44025364e-02,
            2.48686922e-03,
            1.11930407e-04,
            2.09279771e-06,
            1.56585558e-08,
            4.57958958e-11,
            5.15587486e-14,
            2.21142019e-17,
            3.58729395e-21,
        ]
    )

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_pure_exploit_typing():
    annos = package_target.pure_exploit.__annotations__

    assert annos["pred_mean"] == np.ndarray
    assert annos["return"] == np.ndarray


def test_pure_exploit():
    with pytest.raises(AssertionError) as error:
        package_target.pure_exploit("abc")
    with pytest.raises(AssertionError) as error:
        package_target.pure_exploit(np.ones((10, 1)))

    val_acq = package_target.pure_exploit(np.arange(0, 10))
    truth_val_acq = -np.arange(0, 10)

    np.testing.assert_allclose(val_acq, truth_val_acq)


def test_pure_explore_typing():
    annos = package_target.pure_explore.__annotations__

    assert annos["pred_std"] == np.ndarray
    assert annos["return"] == np.ndarray


def test_pure_explore():
    with pytest.raises(AssertionError) as error:
        package_target.pure_explore("abc")
    with pytest.raises(AssertionError) as error:
        package_target.pure_explore(np.ones((10, 1)))

    val_acq = package_target.pure_explore(np.arange(0, 10))
    truth_val_acq = np.arange(0, 10)

    np.testing.assert_allclose(val_acq, truth_val_acq)
