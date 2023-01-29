#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: December 15, 2022
#
"""test_utils_common"""

import typing
import pytest
import numpy as np

from bayeso.utils import utils_common as package_target


TEST_EPSILON = 1e-5

def test_validate_types_typing():
    annos = package_target.validate_types.__annotations__

    assert annos['func'] == typing.Callable
    assert annos['return'] == typing.Callable

def test_validate_types():
    def fun_test_1(x) -> int:
        return int(x)

    with pytest.raises(AssertionError) as error:
        package_target.validate_types(fun_test_1)

    def fun_test_2(x: float):
        return int(x)

    with pytest.raises(AssertionError) as error:
        package_target.validate_types(fun_test_2)

    def fun_test_3(x: float) -> int:
        return int(x)

    fun_returned = package_target.validate_types(fun_test_3)

    fun_test_1(1.23)
    fun_test_2(1.23)
    fun_test_3(1.23)
    fun_returned(1.23)

def test_get_grids_typing():
    annos = package_target.get_grids.__annotations__

    assert annos['ranges'] == np.ndarray
    assert annos['num_grids'] == int
    assert annos['return'] == np.ndarray

def test_get_grids():
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    arr_range_2 = np.array([
        [0.0, 10.0],
        [2.0, 2.0],
        [5.0, 5.0],
    ])

    truth_arr_grid_1 = np.array([
        [0., -2., -5.],
        [0., -2., 0.],
        [0., -2., 5.],
        [5., -2., -5.],
        [5., -2., 0.],
        [5., -2., 5.],
        [10., -2., -5.],
        [10., -2., 0.],
        [10., -2., 5.],
        [0., 0., -5.],
        [0., 0., 0.],
        [0., 0., 5.],
        [5., 0., -5.],
        [5., 0., 0.],
        [5., 0., 5.],
        [10., 0., -5.],
        [10., 0., 0.],
        [10., 0., 5.],
        [0., 2., -5.],
        [0., 2., 0.],
        [0., 2., 5.],
        [5., 2., -5.],
        [5., 2., 0.],
        [5., 2., 5.],
        [10., 2., -5.],
        [10., 2., 0.],
        [10., 2., 5.],
    ])

    truth_arr_grid_2 = np.array([
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [0., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [5., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
        [10., 2., 5.],
    ])

    with pytest.raises(AssertionError) as error:
        package_target.get_grids('abc', 3)
    with pytest.raises(AssertionError) as error:
        package_target.get_grids(arr_range_1, 'abc')
    with pytest.raises(AssertionError) as error:
        package_target.get_grids(np.arange(0, 10), 3)
    with pytest.raises(AssertionError) as error:
        package_target.get_grids(np.ones((3, 3)), 3)
    with pytest.raises(AssertionError) as error:
        package_target.get_grids(np.array([[0.0, -2.0], [10.0, 20.0]]), 3)

    arr_grid_1 = package_target.get_grids(arr_range_1, 3)
    arr_grid_2 = package_target.get_grids(arr_range_2, 3)

    assert (arr_grid_1 == truth_arr_grid_1).all()
    assert (arr_grid_2 == truth_arr_grid_2).all()

def test_get_minimum_typing():
    annos = package_target.get_minimum.__annotations__

    assert annos['Y_all'] == np.ndarray
    assert annos['num_init'] == int
    assert annos['return'] == typing.Tuple[np.ndarray, np.ndarray, np.ndarray]

def test_get_minimum():
    with pytest.raises(AssertionError) as error:
        package_target.get_minimum(1.2, 2.1)
    with pytest.raises(AssertionError) as error:
        package_target.get_minimum(1.2, 3)

    num_init = 3
    num_exp = 3
    num_data = 10
    all_data = np.zeros((num_exp, num_init + num_data))
    with pytest.raises(AssertionError) as error:
        package_target.get_minimum(all_data, 2.1)
    cur_minimum, cur_mean, cur_std, cur_sem = package_target.get_minimum(all_data, num_init)
    assert len(cur_minimum.shape) == 2
    assert cur_minimum.shape == (num_exp, 1 + num_data)
    assert len(cur_mean.shape) == 1
    assert cur_mean.shape == (1 + num_data, )
    assert len(cur_std.shape) == 1
    assert cur_std.shape == (1 + num_data, )
    assert len(cur_sem.shape) == 1
    assert cur_sem.shape == (1 + num_data, )

    num_init = 5
    num_exp = 10
    num_data = -2
    all_data = np.zeros((num_exp, num_init + num_data))
    with pytest.raises(AssertionError) as error:
        package_target.get_minimum(all_data, num_init)

    num_init = 3
    all_data = np.array([
        [3.1, 2.1, 4.1, 2.0, 1.0, 4.1, 0.4],
        [2.3, 4.9, 2.9, 8.2, 3.2, 4.2, 4.9],
        [0.8, 2.4, 5.4, 4.5, 0.3, 1.5, 2.3],
    ])
    truth_all_data = np.array([
        [2.1, 2.0, 1.0, 1.0, 0.4],
        [2.3, 2.3, 2.3, 2.3, 2.3],
        [0.8, 0.8, 0.3, 0.3, 0.3],
    ])
    cur_minimum, cur_mean, cur_std, cur_sem = package_target.get_minimum(all_data, num_init)
    assert (cur_minimum == truth_all_data).all()
    assert (cur_mean == np.mean(truth_all_data, axis=0)).all()
    assert (cur_std == np.std(truth_all_data, axis=0, ddof=1)).all()
    assert (cur_sem == np.std(truth_all_data, axis=0, ddof=1) / np.sqrt(truth_all_data.shape[0])).all()

    num_init = 3
    all_data = np.array([
        [0.8, 2.4, 5.4, 4.5, 0.3, 1.5, 2.3],
    ])
    truth_all_data = np.array([
        [0.8, 0.8, 0.3, 0.3, 0.3],
    ])
    cur_minimum, cur_mean, cur_std, cur_sem = package_target.get_minimum(all_data, num_init)
    assert (cur_minimum == truth_all_data).all()
    assert (cur_mean == np.mean(truth_all_data, axis=0)).all()
    assert (cur_std == np.zeros(truth_all_data.shape[0])).all()
    assert (cur_sem == np.zeros(truth_all_data.shape[0]) / np.sqrt(truth_all_data.shape[0])).all()

def test_get_time_typing():
    annos = package_target.get_time.__annotations__

    assert annos['time_all'] == np.ndarray
    assert annos['num_init'] == int
    assert annos['include_init'] == bool
    assert annos['return'] == np.ndarray

def test_get_time():
    arr_time = np.array([
        [1.0, 0.5, 0.2, 0.7, 2.0],
        [2.0, 0.7, 1.2, 0.3, 0.7],
        [0.2, 0.1, 1.0, 0.2, 1.5],
    ])
    int_init = 2
    is_initial = True
    with pytest.raises(AssertionError) as error:
        package_target.get_time(arr_time, int_init, 1)
    with pytest.raises(AssertionError) as error:
        package_target.get_time(arr_time, 'abc', is_initial)
    with pytest.raises(AssertionError) as error:
        package_target.get_time('abc', int_init, is_initial)
    with pytest.raises(AssertionError) as error:
        package_target.get_time(np.arange(0, 10), int_init, is_initial)
    with pytest.raises(AssertionError) as error:
        package_target.get_time(arr_time, 10, is_initial)

    cur_time = package_target.get_time(arr_time, int_init, is_initial)
    truth_cur_time = np.array([0.0, 0.8, 1.2, 2.6])
    assert (np.abs(cur_time - truth_cur_time) < TEST_EPSILON).all()

    cur_time = package_target.get_time(arr_time, int_init, False)
    truth_cur_time = np.array([0.0, 1.06666667, 1.5, 2.3, 2.7, 4.1])
    assert (np.abs(cur_time - truth_cur_time) < TEST_EPSILON).all()
