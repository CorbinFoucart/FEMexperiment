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

def xi_eta_to_ab(ξ, η):
    """ function to transform xi, eta coords to a, b
    see Hesthaven function 'rstoab'
    @param xi, eta vectors of xi, eta pts
    """
    a, b = np.zeros_like(ξ), np.zeros_like(η)
    singular = np.isclose(η, 1.0)
    nonsingular = np.logical_not(singular)
    a[nonsingular] = 2*(1. + ξ[nonsingular])/(1 - η[nonsingular]) - 1
    a[singular] = -1
    b = η
    return a, b

def poly_gradient_simplex(a, b, i, j):
    """ takes derivatives of modal basis polys w/r/t ξ, η
    transcribed from GradSimplex2D in Hesthaven
    """
    Pa, dP_da = Jacobi_Poly(a,0,    0,i)[-1], Jacobi_Poly_Derivative(a,0    ,0,i)[-1]
    Pb, dP_db = Jacobi_Poly(b,2*i+1,0,j)[-1], Jacobi_Poly_Derivative(b,2*i+1,0,j)[-1]

    # d/dξ = da/dξ * d/da + db/dξ * d/db = 2/(1-b) d/da
    dψ_dξ = dP_da * Pb
    if i > 0:
        dψ_dξ *= ( (0.5*(1-b))**(i-1) )

    # d/dη = ((1+a)/2)/((1-b)/2)d/da + d/db
    dψ_dη = dP_da * (Pb * 0.5*(1+a))
    if i > 0:
        dψ_dη *= ((0.5*(1-b))**(i-1))

    tmp = dP_db * (0.5*(1-b))**(i)
    if i > 0:
        tmp -= 0.5*i*Pb * (0.5*(1-b))**(i-1)
    dψ_dη += Pa * tmp

    # normalize both derivatives
    dψ_dξ *= 2**(i+0.5)
    dψ_dη *= 2**(i+0.5)
    return [dψ_dξ, dψ_dη]

def Vandermonde2D(pts, p, verts=MASTER_TRI_VERTS):
    """ evaluates the vandermonde matrix at the specified points """
    return P_tilde(pts, p, verts)

def GradVandermonde2D(p, ξ, η):
    """ compute the derivative vandermonde matrices in ξ, η directions """
    Np = int((p+1)*(p+2)/2)
    npts = len(ξ)
    a, b = xi_eta_to_ab(ξ, η)
    dVξ, dVη = np.zeros((npts, Np)), np.zeros((npts, Np))
    counter = 0
    for i in range(p+1):
        for j in range(p-i+1):
            dψ_dξ, dψ_dη = poly_gradient_simplex(a, b, i, j)
            dVξ[:,counter] = dψ_dξ
            dVη[:,counter] = dψ_dη
            counter += 1
    return [dVξ, dVη]
