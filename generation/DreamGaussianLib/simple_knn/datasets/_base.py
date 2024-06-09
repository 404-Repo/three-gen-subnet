"""
Base IO code for all datasets
"""

import os
import csv
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib
import numpy as np
import warnings
from urllib.request import urlretrieve

def check_pandas_support(caller_name):
    """Raise ImportError with detailed error message if pandsa is not
    installed.

    Plot utilities like :func:`fetch_openml` should lazily import
    pandas and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires pandas.
    """
    try:
        import pandas  # noqa
        return pandas
    except ImportError as e:
        raise ImportError(
            "{} requires pandas.".format(caller_name)
        ) from e


def _convert_data_dataframe(caller_name, data, target,
                            feature_names, target_names):
    pd = check_pandas_support('{} with as_frame=True'.format(caller_name))
    data_df = pd.DataFrame(data, columns=feature_names)
    target_df = pd.DataFrame(target, columns=target_names)
    combined_df = pd.concat([data_df, target_df], axis=1)
    X = combined_df[feature_names]
    y = combined_df[target_names]
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return combined_df, X, y


def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path : string
        The module path.

    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

    return data, target, target_names

def load_wine(*, as_frame=False):
    """Load and return the wine dataset (classification).

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
    Returns
    -------

    (data, target) : tuple

    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'wine_data.csv')

    with open(join(module_path, 'descr', 'wine_data.rst')) as rst_file:
        fdescr = rst_file.read()

    feature_names = ['alcohol',
                     'malic_acid',
                     'ash',
                     'alcalinity_of_ash',
                     'magnesium',
                     'total_phenols',
                     'flavanoids',
                     'nonflavanoid_phenols',
                     'proanthocyanins',
                     'color_intensity',
                     'hue',
                     'od280/od315_of_diluted_wines',
                     'proline']

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_wine",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    return data, target

def load_iris(*, as_frame=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.
    Returns
    -------

    (data, target) : tuple
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')
    iris_csv_filename = join(module_path, 'data', 'iris.csv')

    with open(join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    feature_names = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)']

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_iris",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    return data, target

def load_breast_cancer(*, as_frame=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <breast_cancer_dataset>`.

    Parameters
    ----------

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        
    Returns
    -------

    (data, target) : tuple

    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'breast_cancer.csv')
    csv_filename = join(module_path, 'data', 'breast_cancer.csv')

    with open(join(module_path, 'descr', 'breast_cancer.rst')) as rst_file:
        fdescr = rst_file.read()

    feature_names = np.array(['mean radius', 'mean texture',
                              'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness',
                              'mean concavity', 'mean concave points',
                              'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error',
                              'perimeter error', 'area error',
                              'smoothness error', 'compactness error',
                              'concavity error', 'concave points error',
                              'symmetry error', 'fractal dimension error',
                              'worst radius', 'worst texture',
                              'worst perimeter', 'worst area',
                              'worst smoothness', 'worst compactness',
                              'worst concavity', 'worst concave points',
                              'worst symmetry', 'worst fractal dimension'])

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, data, target = _convert_data_dataframe("load_breast_cancer",
                                                      data,
                                                      target,
                                                      feature_names,
                                                      target_columns)

    return data, target

def load_digits(*, n_class=10, as_frame=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    Read more in the :ref:`User Guide <digits_dataset>`.

    Parameters
    ----------
    n_class : int, default=10
        The number of classes to return. Between 0 and 10.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    -------
    (data, target) : tuple

    This is a copy of the test set of the UCI ML hand-written digits datasets
    https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    """
    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'data', 'digits.csv.gz'),
                      delimiter=',')
    with open(join(module_path, 'descr', 'digits.rst')) as f:
        descr = f.read()
    target = data[:, -1].astype(int, copy=False)
    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)

    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    feature_names = ['pixel_{}_{}'.format(row_idx, col_idx)
                     for row_idx in range(8)
                     for col_idx in range(8)]

    frame = None
    target_columns = ['target', ]
    if as_frame:
        frame, flat_data, target = _convert_data_dataframe("load_digits",
                                                           flat_data,
                                                           target,
                                                           feature_names,
                                                           target_columns)

    return flat_data, target