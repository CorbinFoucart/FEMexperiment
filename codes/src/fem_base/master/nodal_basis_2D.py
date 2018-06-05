#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# vertex definitions for master Tri and Quad, over which to construct the bases
from src.fem_base.master.master_2D import MASTER_ELEMENT_VERTICES
import src.fem_base.master.barycentric_coord_tools as bct
import src.fem_base.master.polynomials_2D as p2d

class NodalBasis2D(object): pass

class NodalBasis2DTriangle(NodalBasis2D):
    def __init__(self, p, nodal_locations='UNIFORM'):
        """ creates a nodal basis over master element verts
        @param p  polynomial order of the nodal basis
        @param nodal_locations
            UNIFORM: uniformly spaced nodal points (in a barycentric sense)
            WARPED:  shifted points for better interpolation behavior (Hesthaven 2008)
        """
        self.p, self.nb = p, int((p+1)*(p+2)/2.)
        self.verts = MASTER_ELEMENT_VERTICES['TRIANGLE']
        if nodal_locations == 'UNIFORM':
            uniform_bary_coords = bct.uniform_bary_coords(p)
            xp, yp = bct.bary2cart(self.verts, uniform_bary_coords)
            self.nodal_pts = np.vstack((xp, yp)).T

    def shape_functions_at_pts(self, pts):
        """ computes values of shape functions at pts
        """
        V = p2d.Vandermonde2D(self.nodal_pts, self.p)
        VTi = np.linalg.inv(V.T)
        P_tilde = p2d.P_tilde(pts, self.p)[:, 0:self.nb]
        shap = np.dot(VTi, P_tilde.T)
        return shap

class NodalBasis2DQuad(NodalBasis2D):
    def __init__(self):
        self.basis_domain_verts = MASTER_ELEMENT_VERTICES['QUAD']


