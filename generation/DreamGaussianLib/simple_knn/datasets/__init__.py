"""
The :mod:`simple_kNN.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
from ._base import load_breast_cancer
from ._base import load_digits
from ._base import load_iris
from ._base import load_wine


__all__ = ['load_digits',
           'load_iris',
           'load_breast_cancer',
           'load_wine'
        ]
