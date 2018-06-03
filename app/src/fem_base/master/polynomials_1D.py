#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import orthopy
from scipy.interpolate import lagrange

def Jacobi_Poly(r, alpha, beta, N):
    """ wraps orthopy to return Jacobi polynomial """
    return orthopy.line_segment.tree_jacobi(r, n=N-1,
        alpha=alpha, beta=beta, standardization='normal')

def Jacobi_Poly_Derivative(r, alpha, beta, N):
    """ take a derivative of Jacobi Poly, more general than above
    copy the format of orthopy (list of arrays)
    """
    dp = [np.zeros_like(r)]
    Jacobi_P = Jacobi_Poly(r, alpha + 1, beta + 1, N)
    for n in range(1, N+1):
        gamma = np.sqrt(n * (n + alpha + beta + 1))
        dp.append(gamma * Jacobi_P[n-1])
    return dp

def P_tilde(r, N):
    P = np.zeros((len(r), N))
    polyvals = Jacobi_Poly(r, alpha=0, beta=0, N=N)
    for j in range(N):
        P[:, j] = polyvals[j]
    return P.T

def Vandermonde1D(N, x):
    """ initialize 1D vandermonde Matrix Vij = phi_j(x_i)"""
    V1D = np.zeros((len(x), N))
    JacobiP = Jacobi_Poly(x, alpha=0, beta=0, N=N)
    for j, polyvals in enumerate(JacobiP):
        V1D[:, j] = polyvals
    return V1D

def GradVandermonde1D(N, x):
    Vr = np.zeros((len(x), N))
    dJacobi_P = Jacobi_Poly_Derivative(x, alpha=0, beta=0, N=N-1)
    for j, polyder in enumerate(dJacobi_P):
        Vr[:,j] = polyder
    return Vr

def lagrange_polys(pts):
    lagrange_polys = []
    for i, pt in enumerate(pts):
        data = np.zeros_like(pts)
        data[i] = 1
        lagrange_polys.append(lagrange(pts, data))
    return lagrange_polys

def lagrange_basis_at_pts(lagrange_polys, eval_pts):
    """ evaluates lagrange polynomials at eval_pts"""
    result = np.zeros((len(lagrange_polys) ,len(eval_pts)))
    for i, poly in enumerate(lagrange_polys):
        result[i, :] = lagrange_polys[i](eval_pts)
    return result
