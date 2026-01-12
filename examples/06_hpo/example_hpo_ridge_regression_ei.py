#
# author: Jungtaek Kim (jungtaek.kim.mail@gmail.com)
# last updated: November 20, 2024
#

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model

from bayeso import bo
from bayeso.wrappers import wrappers_bo_function


HOUSING = sklearn.datasets.fetch_california_housing()
HOUSING_DATA = HOUSING.data
HOUSING_LABELS = HOUSING.target
DATA_TRAIN, DATA_TEST, LABELS_TRAIN, LABELS_TEST = (
    sklearn.model_selection.train_test_split(
        HOUSING_DATA, HOUSING_LABELS, test_size=0.3
    )
)


def fun_target(X):
    print(X)
    ridge_model = sklearn.linear_model.Ridge(alpha=X[0])
    ridge_model.fit(DATA_TRAIN, LABELS_TRAIN)
    preds = ridge_model.predict(DATA_TEST)
    mse = sklearn.metrics.mean_squared_error(LABELS_TEST, preds)
    return mse


# (alpha, )
num_init = 1

model_bo = bo.BO(np.array([[0.1, 2]]), debug=True)

list_Y = []
list_time = []

for _ in range(0, 10):
    X_final, Y_final, time_final, _, _ = wrappers_bo_function.run_single_round(
        model_bo,
        fun_target,
        num_init,
        10,
        str_initial_method_bo="sobol",
        str_sampling_method_ao="sobol",
        num_samples_ao=100,
    )

    list_Y.append(Y_final)
    list_time.append(time_final)

arr_Y = np.array(list_Y)
arr_Y = np.expand_dims(np.squeeze(arr_Y), axis=0)
arr_time = np.array(list_time)
arr_time = np.expand_dims(arr_time, axis=0)
