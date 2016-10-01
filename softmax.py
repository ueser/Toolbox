#! /usr/bin/env python
"""
Author: Umut Eser
Program: softmax.py
Date: Friday, September 30 2016
Description: Softmax applied over rows of a matrix
"""

import numpy as np


def softmax(X):
    """
    Calculates softmax of the rows of a matrix X.

    Parameters
    ----------
    X : 2D numpy array

    Return
    ------
    2D numpy array of positive numbers between 0 and 1

    Examples
    --------
    >>> softmax([[0.1, 0.2],[0.9, -10]])
    array([[ 0.47502081,  0.52497919],[  9.99981542e-01,   1.84578933e-05]])
    """
    e_X = np.exp(X - np.max(X,axis=1))
    return np.divide(e_X.T,e_X.sum(axis=1)).T


if __name__ == "__main__":
    import doctest
    doctest.testmod()
