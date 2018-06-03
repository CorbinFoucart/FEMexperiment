# -*- coding: utf-8 -*-

"""@file isclose.py
Provides an implementation of the @c isclose() function

This function is found in the @c numpy library starting with version 1.7.0.
The MSEAS @c numpy library version is 1.5.1.

@author Chris Mirabito (mirabito@mit.edu)
"""

import numpy


def isclose(a, b, rtol=1e-05, atol=1e-08):
    r"""Implements @c numpy.isclose

    Returns a boolean array where two arrays are element-wise equal within
    a tolerance. The tolerance values are positive, typically very small
    numbers.  For finite values, two floating point values are considered
    equivalent if
    @f[
    |a-b|\leq T_\text{abs}+T_\text{rel}|b|
    @f]
    where @f$T_\text{abs}@f$ and @f$T_\text{rel}@f$ are the absolute and
    relative tolerances, respectively.  The above equation is not symmetric!

    @param a: Input array to compare
    @param b: Input array to compare
    @param rtol: Relative tolerance
    @param atol: Absolute tolerance

    @return A boolean array of where the arrays are equal within the given
     tolerance
    """

    return numpy.abs(a - b) <= (atol + rtol * numpy.abs(b))
