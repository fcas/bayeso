#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: April 30, 2024
#
"""test_bo_bo_w_gp"""

import pytest
import numpy as np
import scipy

from bayeso.bo import bo_w_gp as package_target
from bayeso import covariance
from bayeso.utils import utils_bo
from bayeso.utils import utils_covariance


BO = package_target.BOwGP


def test_load_bo():
    # legitimate cases
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
    # wrong cases
    arr_range_3 = np.array([
        [20.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    arr_range_4 = np.array([
        [20.0, 10.0],
        [4.0, 2.0],
        [10.0, 5.0],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo = BO(1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(np.arange(0, 10))
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_3)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_4)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_cov=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_cov='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_acq=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_acq='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, use_ard='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, use_ard=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, prior_mu=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_gp=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_gp='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_optimizer_method_bo='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_modelselection_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_modelselection_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, str_exp=123)
    with pytest.raises(AssertionError) as error:
        model_bo = BO(arr_range_1, debug=1)

    model_bo = BO(arr_range_1)
    model_bo = BO(arr_range_2)

def test_get_samples():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    model_bo = BO(arr_range, debug=True)

    with pytest.raises(AssertionError) as error:
        model_bo.get_samples(1)
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('uniform', num_samples='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('uniform', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('gaussian', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('sobol', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('halton', seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_samples('abc')

    arr_initials = model_bo.get_samples('grid', num_samples=1)
    truth_arr_initials = np.array([
        [0.000, -2.000, -5.000],
    ])
    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials = model_bo.get_samples('grid', num_samples=3)
    truth_arr_initials = np.array([
        [0.000, -2.000, -5.000],
        [0.000, -2.000, 0.000],
        [0.000, -2.000, 5.000],
        [5.000, -2.000, -5.000],
        [5.000, -2.000, 0.000],
        [5.000, -2.000, 5.000],
        [10.000, -2.000, -5.000],
        [10.000, -2.000, 0.000],
        [10.000, -2.000, 5.000],
        [0.000, 0.000, -5.000],
        [0.000, 0.000, 0.000],
        [0.000, 0.000, 5.000],
        [5.000, 0.000, -5.000],
        [5.000, 0.000, 0.000],
        [5.000, 0.000, 5.000],
        [10.000, 0.000, -5.000],
        [10.000, 0.000, 0.000],
        [10.000, 0.000, 5.000],
        [0.000, 2.000, -5.000],
        [0.000, 2.000, 0.000],
        [0.000, 2.000, 5.000],
        [5.000, 2.000, -5.000],
        [5.000, 2.000, 0.000],
        [5.000, 2.000, 5.000],
        [10.000, 2.000, -5.000],
        [10.000, 2.000, 0.000],
        [10.000, 2.000, 5.000],
    ])
    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials_ = model_bo.get_samples('sobol', num_samples=4)
    arr_initials = model_bo.get_samples('sobol', num_samples=4, seed=42)

    print('sobol')
    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    truth_arr_initials = np.array([
        [
            4.31029474362731,
            1.257471889257431,
            3.06412766687572,
        ],
        [
            8.69813535362482,
            -0.2500223182141781,
            -0.012653125450015068,
        ],
        [
            5.779154505580664,
            0.04064444452524185,
            2.2647008765488863,
        ],
        [
            1.3686652854084969,
            -1.0451578684151173,
            -4.681709306314588,
        ],
    ])

    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials_ = model_bo.get_samples('halton', num_samples=3)
    arr_initials = model_bo.get_samples('halton', num_samples=3, seed=42)

    print('halton')
    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    if scipy.__version__ == '1.10.1':
        truth_arr_initials = np.array([
            [
                5.513058694915099,
                0.9508863268359247,
                4.394594269075903,
            ],
            [
                0.5130586949150984,
                -0.3824470064974086,
                0.39459426907590256,
            ],
            [
                8.013058694915099,
                -1.7157803398307416,
                2.3945942690759034,
            ],
        ])
    else:
        truth_arr_initials = np.array([
            [
                5.513058694915099,
                -1.3929280802587178,
                -3.572948073154651,
            ],
            [
                0.5130586949150984,
                1.2737385864079487,
                0.4270519268453521,
            ],
            [
                8.013058694915099,
                -0.059594746925384356,
                2.427051926845353,
            ],
        ])

    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials_ = model_bo.get_samples('uniform', num_samples=3)
    arr_initials = model_bo.get_samples('uniform', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [3.74540119, 1.80285723, 2.31993942],
        [5.98658484, -1.37592544, -3.4400548],
        [0.58083612, 1.46470458, 1.01115012],
    ])
    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials_ = model_bo.get_samples('gaussian', num_samples=3)
    arr_initials = model_bo.get_samples('gaussian', num_samples=3, seed=42)
    truth_arr_initials = np.array([
        [6.241785382528082, -0.13826430117118466, 1.6192213452517312],
        [8.807574641020064, -0.23415337472333597, -0.5853423923729514],
        [8.948032038768478, 0.7674347291529088, -1.1736859648373803],
    ])
    np.testing.assert_allclose(arr_initials, truth_arr_initials)

def test_get_initials():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    model_bo = BO(arr_range)

    with pytest.raises(AssertionError) as error:
        model_bo.get_initials(1, 10)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('grid', 'abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('grid', 10)
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('uniform', 10, seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.get_initials('abc', 10)

    arr_initials = model_bo.get_initials('sobol', 4)
    arr_initials = model_bo.get_initials('sobol', 4, seed=42)

    print('sobol')
    for elem_1 in arr_initials:
        for elem_2 in elem_1:
            print(elem_2)

    truth_arr_initials = np.array([
        [
            4.31029474362731,
            1.257471889257431,
            3.06412766687572,
        ],
        [
            8.69813535362482,
            -0.2500223182141781,
            -0.012653125450015068,
        ],
        [
            5.779154505580664,
            0.04064444452524185,
            2.2647008765488863,
        ],
        [
            1.3686652854084969,
            -1.0451578684151173,
            -4.681709306314588,
        ],
    ])

    np.testing.assert_allclose(arr_initials, truth_arr_initials)

    arr_initials = model_bo.get_initials('uniform', 3, seed=42)
    truth_arr_initials = np.array([
        [3.74540119, 1.80285723, 2.31993942],
        [5.98658484, -1.37592544, -3.4400548],
        [0.58083612, 1.46470458, 1.01115012],
    ])
    np.testing.assert_allclose(arr_initials, truth_arr_initials)

def test_optimize():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)
    model_bo = BO(arr_range_1)

    with pytest.raises(AssertionError) as error:
        model_bo.optimize(1, Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(np.random.randn(num_X), Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(np.random.randn(num_X, 1), Y)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(num_X))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(num_X, 2))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, np.random.randn(3, 1))
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_sampling_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_sampling_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_mlm_method=1)
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, str_mlm_method='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, num_samples='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, seed='abc')
    with pytest.raises(AssertionError) as error:
        model_bo.optimize(X, Y, seed=1.23)

    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_acq():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='ucb')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='aei')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_exploit')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1, str_acq='pure_explore')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_optimize_method_bo():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_optimizer_method_bo='L-BFGS-B')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    # TODO: add DIRECT test, now it causes an error.

    model_bo = BO(arr_range_1, str_optimizer_method_bo='CMA-ES')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_mlm_method():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1)
    next_point, dict_info = model_bo.optimize(X, Y, str_mlm_method='converged')
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

    model_bo = BO(arr_range_1)
    next_point, dict_info = model_bo.optimize(X, Y, str_mlm_method='combined')
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_str_modelselection_method():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_modelselection_method='loocv')
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]

def test_optimize_use_ard():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range, use_ard=False)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
    assert isinstance(hyps['lengthscales'], float)

    X = np.array([
        [3.0, 0.0, 1.0],
        [2.0, -1.0, 4.0],
        [9.0, 1.5, 3.0],
    ])
    Y = np.array([
        [100.0],
        [100.0],
        [100.0],
    ])

    model_bo = BO(arr_range, use_ard=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    next_points = dict_info['next_points']
    acquisitions = dict_info['acquisitions']
    cov_X_X = dict_info['cov_X_X']
    inv_cov_X_X = dict_info['inv_cov_X_X']
    hyps = dict_info['hyps']
    time_overall = dict_info['time_overall']
    time_surrogate = dict_info['time_surrogate']
    time_acq = dict_info['time_acq']

    assert isinstance(next_point, np.ndarray)
    assert isinstance(next_points, np.ndarray)
    assert isinstance(acquisitions, np.ndarray)
    assert isinstance(cov_X_X, np.ndarray)
    assert isinstance(inv_cov_X_X, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(time_overall, float)
    assert isinstance(time_surrogate, float)
    assert isinstance(time_acq, float)
    assert next_point.ndim == 1
    assert next_points.ndim == 2
    assert acquisitions.ndim == 1
    assert next_point.shape[0] == dim_X
    assert next_points.shape[1] == dim_X
    assert next_points.shape[0] == acquisitions.shape[0]
    assert isinstance(hyps['lengthscales'], np.ndarray)
    assert hyps['lengthscales'].ndim == 1
    assert hyps['lengthscales'].shape[0] == 3

def test_optimize_normalize_Y():
    np.random.seed(42)
    arr_range = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range, normalize_Y=True)
    next_point, dict_info = model_bo.optimize(X, Y)
    Y_original = dict_info['Y_original']
    Y_normalized = dict_info['Y_normalized']

    assert np.all(Y == Y_original)
    assert np.all(Y != Y_normalized)
    assert np.all(utils_bo.normalize_min_max(Y) == Y_normalized)

    model_bo = BO(arr_range, normalize_Y=False)
    next_point, dict_info = model_bo.optimize(X, Y)
    Y_original = dict_info['Y_original']
    Y_normalized = dict_info['Y_normalized']

    assert np.all(Y == Y_normalized)
    assert np.all(Y == Y_original)

def test_compute_posteriors():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='ei')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)

    X_test = model_bo.get_samples('sobol', num_samples=16, seed=111)

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1.0)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 'abc')

    pred_mean, pred_std = model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps)

    assert pred_mean.ndim == 1
    assert pred_std.ndim == 1
    assert pred_mean.shape[0] == pred_mean.shape[0] == X_test.shape[0]

def test_compute_posteriors_set():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    num_instances = 4
    X = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi', str_cov='set_se', str_exp=None)
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)
    
    X_test = np.array([
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, -1.0, 2.0, 1.0],
            [3.0, -2.0, 4.0, 1.0],
        ],
        [
            [4.0, 2.0, -3.0, 1.0],
            [5.0, 0.0, -2.0, 1.0],
            [6.0, -2.0, -1.0, 1.0],
        ],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(1, Y, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, 1, X_test, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 1.0)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, 'abc')

    with pytest.raises(AssertionError) as error:
        model_bo.compute_posteriors(X, Y, X_test, cov_X_X, inv_cov_X_X, hyps)

    pred_mean, pred_std = model_bo.compute_posteriors(X, Y, X_test[:, :, :dim_X], cov_X_X, inv_cov_X_X, hyps)

    assert pred_mean.ndim == 1
    assert pred_std.ndim == 1
    assert pred_mean.shape[0] == pred_mean.shape[0] == X_test.shape[0]

def test_compute_acquisitions():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    X = np.random.randn(num_X, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi', str_exp='test')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)

    X_test = model_bo.get_samples('sobol', num_samples=8, seed=111)

    truth_X_test = np.array([
        [
            3.359774835407734,
            -0.7351906783878803,
            -3.654018910601735,
        ],
        [
            5.6929760333150625,
            0.06583871319890022,
            4.514206442981958,
        ],
        [
            9.900951310992241,
            -1.9652910344302654,
            -2.275532428175211,
        ],
        [
            2.296332074329257,
            1.3591753654181957,
            0.795036694034934,
        ],
        [
            0.2561802417039871,
            -1.420634139329195,
            3.192977551370859,
        ],
        [
            7.546542389318347,
            1.7783369608223438,
            -4.986539306119084,
        ],
        [
            6.79605845361948,
            -0.12705285474658012,
            2.1275955345481634,
        ],
        [
            4.1511941608041525,
            0.5484802536666393,
            -0.9542650915682316,
        ],
    ])

    for elem_1 in X_test:
        for elem_2 in elem_1:
            print(elem_2)

    np.testing.assert_allclose(X_test, truth_X_test)

    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(1, X, Y, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, 1, Y, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, 1, cov_X_X, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, 1, inv_cov_X_X, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, 1, hyps)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, 1)
    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, 'abc')

    acqs = model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, hyps)

    print('acqs')
    for elem_1 in acqs:
        print(elem_1)

    truth_acqs = np.array([
        0.8049686141202866,
        0.7893967936446344,
        0.7893344548527176,
        0.6000390368336054,
        0.8103002453939229,
        0.7893313332382224,
        0.7894705208469283,
        0.7800155900459567,
    ])

    assert isinstance(acqs, np.ndarray)
    assert acqs.ndim == 1
    assert X_test.shape[0] == acqs.shape[0]
    np.testing.assert_allclose(acqs, truth_acqs)

def test_compute_acquisitions_set():
    np.random.seed(42)
    arr_range_1 = np.array([
        [0.0, 10.0],
        [-2.0, 2.0],
        [-5.0, 5.0],
    ])
    dim_X = arr_range_1.shape[0]
    num_X = 5
    num_instances = 4
    X = np.random.randn(num_X, num_instances, dim_X)
    Y = np.random.randn(num_X, 1)

    model_bo = BO(arr_range_1, str_acq='pi', str_cov='set_se', str_exp='test')
    hyps = utils_covariance.get_hyps(model_bo.str_cov, dim=dim_X, use_ard=model_bo.use_ard)

    cov_X_X, inv_cov_X_X, _ = covariance.get_kernel_inverse(X, hyps, model_bo.str_cov)
    
    X_test = np.array([
        [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, -1.0, 2.0, 1.0],
            [3.0, -2.0, 4.0, 1.0],
        ],
        [
            [4.0, 2.0, -3.0, 1.0],
            [5.0, 0.0, -2.0, 1.0],
            [6.0, -2.0, -1.0, 1.0],
        ],
    ])

    with pytest.raises(AssertionError) as error:
        model_bo.compute_acquisitions(X_test, X, Y, cov_X_X, inv_cov_X_X, hyps)
