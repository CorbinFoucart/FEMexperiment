#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import src.fem_base.master.barycentric_coord_tools as bct
import src.fem_base.master.polynomials_1D as p1d

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

def Simplex2DPoly(a, b, i, j):
    """ generates the orthonormal polynomial over the master simplex (mapped to a, b coords)
    transcriber from Hesthaven 2008 (Simplex2DP.m)
    """
    h1 = p1d.JacobiP(a, 0, 0, i)
    h2 = p1d.JacobiP(b, 2*i+1, 0, j)
    P = np.sqrt(2.)*h1*h2*(1-b)**i
    return P

def Vandermonde2D(N, ξ, η):
    a, b = xi_eta_to_ab(ξ, η)
    Np = int((N+1)*(N+2)/2)
    V2d = np.zeros((len(ξ), Np))
    counter = 0
    for i in range(N+1):
        for j in range(N-i+1):
            m = j + (N+1)*i + 1 - i/2.*(i-1)
            V2d[:,counter] = Simplex2DPoly(a, b, i,j)
            counter += 1
    return V2d

def Simplex2DPolyGradient(a, b, i, j):
    """ takes derivatives of modal basis polys w/r/t ξ, η
    transcribed from Hesthaven (GradSimplex2D.m)
    """
    Pa, dP_da = p1d.JacobiP(a,0,    0,i), p1d.GradJacobiP(a,0    ,0,i)
    Pb, dP_db = p1d.JacobiP(b,2*i+1,0,j), p1d.GradJacobiP(b,2*i+1,0,j)

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
