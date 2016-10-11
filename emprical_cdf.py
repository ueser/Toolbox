#! /usr/bin/env python
"""
Author: Umut Eser
Program: emprical_cdf.py
Date: Friday, October 11, 2016
Description: Returns emprical cumulative distribution of a 1D sample 

"""

import numpy as np


def emprical_cdf(X):
    """
    Returns emprical cumulative distribution of a 1D sample 


    Parameters
    ----------
    X : 1D numpy array

    Return
    ------
    xvals: 1D numpy array of supports
    yvals: 1D numpy array of cumulative probability densities

    Examples
    --------
    >>> emprical_cdf([0.1, 0.2,0.1,0.1])
    """
  values, base = np.histogram(X, bins=100)
  #evaluate the cumulative
  cumulative = np.cumsum(values)
  cumulative = (cumulative+0.)/np.max(cumulative)
  return base[:-1], cumulative


if __name__ == "__main__":
    import doctest
    doctest.testmod()
