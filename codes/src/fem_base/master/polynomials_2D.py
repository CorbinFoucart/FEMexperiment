#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import orthopy

import src.fem_base.master.barycentric_coord_tools as bct
from src.fem_base.master.master_2D import MASTER_ELEMENT_VERTICES

# define vertices on master elements as specified in master_2D.py
MASTER_TRI_VERTS = MASTER_ELEMENT_VERTICES['TRIANGLE']

def mk_m2ij(p):
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
    m2ij = mk_m2ij(p)
    ortho_output = orthopy.triangle.tree(bary, n=p, standardization='normal')
    npts, npolys = bary.shape[1], len(m2ij)
    polys = np.zeros((npts, npolys))
    for m, (i,j) in enumerate(m2ij):
        polys[:,m] = ortho_output[j][i,:]
    return polys

def P_tilde(pts, p, verts=MASTER_TRI_VERTS):
    """ generates the values of the orthonormal modal polynomials at pts r on the reference tri
    @param verts  tuple of tuples specifying the CCW vertices of the triangle in question
    @param pts  points defined on the triangle defd by verts (npts, 2)
    @param p  order of the orthonormal polynomial basis to be generated
    """
    bary_coords = bct.cart2bary(verts, pts.T)
    polys = ortho_triangle(bary_coords, p+1)
    return polys

def Vandermonde2D(pts, p, verts=MASTER_TRI_VERTS):
    return P_tilde(pts, p, verts)
