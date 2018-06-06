#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import quadpy

import src.fem_base.master.nodal_basis_2D as nb2d
import src.fem_base.master.barycentric_coord_tools as bct

class Master2D(object):
    def mk_shap_and_dshap_at_pts(self, pts):
        shap = self.basis.shape_functions_at_pts(pts)
        dshap = self.basis.shape_function_derivatives_at_pts(pts)
        return shap, dshap

class Master2DTriangle(Master2D):
    """ note vertex definitions in nodal_basis_2D.py """
    def __init__(self, p, nquad_pts=None, *args, **kwargs):
        self.p = p
        self.basis = nb2d.NodalBasis2DTriangle(self.p, **kwargs)
        self.nb, self.verts = self.basis.nb, self.basis.verts
        self.nodal_pts = self.basis.nodal_pts
        self.nq = 2*self.p + 2 if nquad_pts is None else nquad_pts
        self.quad_pts, self.wghts = triangle_quadrature(self.nq, self.verts)

        # shape functions at nodal and quadrature points
        self.shap_quad,  self.dshap_quad = self.mk_shap_and_dshap_at_pts(self.quad_pts)



class Master2DQuad(Master2D): pass

# adapter functions to interface with MPU cubature rules
def triangle_quadrature(n, verts):
    """ look up / compute quadrature rule over the triangle, order n
    @param n  the order of polynomial which should be integrated exactly
    @param verts  tuple of tuples defining the master element
    NOTE: leverages quadpy
    """
    if n > 50:
        raise NotImplementedError
    qr = quadpy.triangle.xiao_gimbutas.XiaoGimbutas(index=n)
    bary, weights = qr.bary, qr.weights
    xq, yq = bct.bary2cart(verts=verts, _lambda=bary.T)
    points = np.vstack((xq, yq)).T
    return points, weights
