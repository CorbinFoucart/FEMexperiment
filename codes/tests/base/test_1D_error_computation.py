#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym
import pytest

import src.fem_base.master.master_1D as M1D
import src.handlers.dof_handler_1D as DOFH
import src.msh.mesh_1D as MSH
import src.fem_maps.fem_map as MAP
import src.fem_base.compute_errors as ERRORS

def analytical_L2_norm(expr, Ω):
    """ symbolically compute L2 norm of expr
    @param expr  sympy expression using x as symbol
    @param Ω  tuple describing domain
    """
    x = sym.Symbol('x')
    integrand = sym.Abs(expr)**2
    integral = float(sym.integrate(integrand, (x, *Ω)))
    return np.sqrt(integral)

def computational_domain(p, n_elm, Ω=(0, 3)):
    """ generate a 1D computational domain """
    mesh = MSH.Mesh1D(P=np.linspace(*Ω, n_elm+1))
    master = M1D.Master1D(p=p)
    dofh = DOFH.DG_DOFHandler1D(mesh, master)
    mapdgn = np.zeros((dofh.dgnodes.shape[0], 1, dofh.dgnodes.shape[1]))
    mapdgn[:,0,:] = dofh.dgnodes
    _map = MAP.Isoparametric_Mapping(master=[master], dgnodes=[mapdgn], map_nodes='NODAL')
    return mesh, master, dofh, _map

@pytest.mark.parametrize("p", [1, 3, 10])
def test_compute_domain_volume(p):
    """ compute volume of computational domain """
    Ω = (-2, 3)
    mesh, master, dofh, _map = computational_domain(p=p, n_elm=4, Ω=Ω)
    u, detJ = np.ones_like(dofh.dgnodes), _map._detJ[0]
    domain_volume = np.sum(np.dot(master.M, detJ*u))
    assert np.isclose(domain_volume, 5.0)

# expected output generated with analytical_L2_norm above
@pytest.mark.parametrize("p, expected", [
        (1, 3.0),
        (2, 6.971370023173351),
        (5, 126.90261119170373),
        (10, 22318.42416672186)
])
def test_compute_polynomial_L2_domain_integral(p, expected):
    """ compute the L2 norm of x**p integrated over Ω """
    Ω = (0, 3)
    mesh, master, dofh, _map = computational_domain(p=p, n_elm=4, Ω=Ω)
    u_h = dofh.dgnodes ** p
    dom_integral = ERRORS._1D_L2_domain_integral(u_h, dofh, _map, master)
    assert np.isclose(dom_integral, expected)
