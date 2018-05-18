#!/usr/bin/env python
"""
@package src.mms.MMS
Created May 2017
Contains the MMS classes

@author Corbin Foucart
"""
#===================================================================================================
# Standard python imports and calls
#===================================================================================================
import sys, os, warnings
import abc
import copy
import shutil
import pdb

# Local imports
from src.ms.ms_util import *

class MS(object):
    def __init__(self, *args):
        raise NotImplementedError

    def create_eval_fn(self, lambdified):
        """ creates a function which can accept vectorized spatial input.
        @retval evalfn  a lambda function which can be called on a numpy array of points, and a
          time. For example: TODO add example to documentation
        """
        if self.dim == 3:
            evalfn = lambda x, t: lambdified(x[:,0], x[:,1], x[:,2], t)
        elif self.dim == 2:
            evalfn = lambda x, t: lambdified(x[:,0], x[:,1], t)
        return evalfn

class ADR_MS(MS):
    def __init__(self, phi, dim, kappa=None, v=None,  *args, **kwargs):
        """
        phi, kappa, v are sympy expressions for each quantity; note that by default, there is no
        advection, and kappa is 1, reducing to a poisson equation.
        @param phi  sympy expression of the problem solution
        @param dim  problem dimension -- note that this can not be inferred
        @param kappa  list of sympy expressions indexed by dimension
        @param v  list of sympy expressions describing velocity field, indexed by dimension
        """
        self.dim = dim
        self.syms = list(sym.symbols('x y z')[0:self.dim]) + [Symbol('t')]
        self.phi = phi
        self.kappa = kappa if kappa is not None else [1]*self.dim
        self.v = v if v is not None else [0]*self.dim
        self.f = self.compute_analytical_forcing_function()
        self.lambdified_MS = self.create_lambdified_fns()

    def compute_analytical_forcing_function(self):
        """ computes analytical forcing function from exact solution
        $ f = -\nabla \cdot (\kappa \nabla\phi) + \nabla\cdot(\phi \vec{v}) $
        """
        phi = ScalarField(self.phi, self.dim)
        f = phi.time_derivative() - phi.scalar_laplacian(self.kappa) + phi.advection(self.v)
        return f

    def create_lambdified_fns(self):
        """ returns a dict containing the lambdified MS functions """
        fns, fields = dict(), ['phi', 'f']
        for field in fields:
            attr = getattr(self, field)
            lfn = sym.lambdify(self.syms, attr, 'numpy')
            fns[field] = self.create_eval_fn(lfn)
        return fns

class Poisson_MS(ADR_MS):
    def __init__(self, phi, dim, *args, **kwargs):
        ADR_MS.__init__(self, phi, dim, *args, v=None, **kwargs)

class INS_MS(MS):
    """
    MS object for the non-dimensionalized INS equations
    """
    def __init__(self, v, p, Re, *args, **kwargs):
        """
        v, p are analytical functions expressing the solution v, p.
        """
        self.dim = len(v)
        self.syms = list(sym.symbols('x y z')[0:self.dim]) + [Symbol('t')]
        self.v, self.p, self.Re = v, [p], Re
        self.check_div_free()
        self.f = self.compute_analytical_forcing_function()
        self.gradv = self.compute_gradv()
        self.lambdified_MS = self.create_lambdified_fns()

    def check_div_free(self):
        """ checks divergence free condition on the velocity field """
        if VectorField(self.v).divergence().expand() != 0:
            warnings.warn("WARNING: non-divergence-free velocity field")

    def compute_analytical_forcing_function(self):
        """ computes the forcing function symbolically analytically according to: [latex] """
        v, p, = VectorField(self.v), ScalarField(self.p[0], self.dim)
        time_derivative = v.time_derivative()
        advection = v.advection()
        gradp = p.gradient()
        diffusion = 1./self.Re * v.laplacian()
        f = time_derivative + advection + gradp - diffusion  # sympy Matrix
        return [_list[0] for _list in f.tolist()]

    def compute_gradv(self):
        """ computes the gradient of each scalar component of v
        This is useful for Neumann boundary conditions, when this should be dotted with normal.
        We store the gradients of each scalar component according to a list;
        e.g., gradv[i] = gradient of velocity component xi
        """
        v, gradv = VectorField(self.v), list()
        for idx, symbol in enumerate(self.syms[0:-1]):
            gradComponent = [expr for expr in v.component_gradient(idx)]
            gradv.append(gradComponent)
        return gradv

    def create_lambdified_fns(self):
        """ returns a dict containing the lambdified MS functions """
        fns, fields = dict(), ['p', 'v', 'f', 'gradv']
        for field in fields:
            attr = getattr(self, field)
            for expr in attr:
                lfn = sym.lambdify(self.syms, expr, 'numpy')
                fns[field] = self.create_eval_fn(lfn)
        return fns

