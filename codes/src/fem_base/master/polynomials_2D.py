#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import orthopy

def m2ij(p):
    """ returns a list A for which A[m] = (i,j) for orthopy polynomials psi_m"""
    return [(j,i) for i in range(p) for j in range(i+1)]

def ortho_triangle(bary, p):
    """ wraps orthopy, returns tree of orthonormal polys over bary coords
    @param bary  barycentric points on a triangle at which to eval polynomials
    @param p  the order of the basis -- how many polynomials we want
    @retval polys (npts, m) array of psi_m at the npts

    orthopy returns a list of arrays, where `out[k]` hosts the `2*k+1`
    values of the `k`th level of the tree
        (0, 0)
        (0, 1)   (1, 1)
        (0, 2)   (1, 2)   (2, 2)
          ...      ...      ...   (i, j)
    so we unpack the (i,j) into the index m via m2ij above.
    """
    ortho_output = orthopy.triangle.tree(bary, n=p, standardization='normal')
    npts = bary.shape[1]
    polys = np.zeros((npts, p))
    for m, (i,j) in enumerate(m2ij(p)):
        polys[:,m] = ortho_output[i][j,:]
    return polys

def P_tilde(r, N): pass

