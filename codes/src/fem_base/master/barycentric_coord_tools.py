#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def cart2bary(verts, pts):
    """ retrns barycentric coords of pt on a tri w/ vertices verts
    @param verts  tuple of tuples specifying verts
    @param pt  numpy array of point on triangle shape (2, npts)
    """
    npts = pts.shape[1]
    _lambda = np.zeros((3, npts))
    (x1, y1), (x2, y2), (x3, y3) = verts
    T = np.array([[x1 - x3, x2 - x3],
                  [y1 - y3, y2 - y3]])
    r = np.copy(pts)
    r[0,:] -= x3
    r[1,:] -= y3

    _lambda[:2, :] = np.linalg.solve(T, r)
    _lambda[2, :] = 1 - _lambda[0,:] - _lambda[1,:]
    return _lambda

def bary2cart(verts, _lambda):
    """converts from barycentric to cartesian coordinates
    @param verts  tuple of tuples specifying triangle vertices
    @param _lambda array of barycentric points (3, npts)
    """
    npts = _lambda.shape[1]
    λ1, λ2, λ3 = _lambda[0,:], _lambda[1,:], _lambda[2,:]
    x, y = np.zeros(npts), np.zeros(npts)
    (x1, y1), (x2, y2), (x3, y3) = verts
    x = λ1*x1 + λ2*x2 + λ3*x3
    y = λ1*y1 + λ2*y2 + λ3*y3
    return x, y

def uniform_bary_coords(p):
    """ generate bary coords of uniform nodal pts for tri of order p"""
    N, Np = p, int((p+1)*(p+2)/2)
    node = 0
    bary = np.zeros((3, Np))
    for i in range(N+1):
        for j in range(N+1-i):
            λ1, λ2 = i/N, j/N
            λ3 = 1 - λ1 - λ2
            bary[:, node] = np.asarray([λ1, λ2, λ3])
            node += 1
    return bary
