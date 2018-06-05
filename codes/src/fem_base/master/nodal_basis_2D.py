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
            self.nodal_pts = self.mk_uniform_nodal_pts()

    def shape_functions_at_pts(self, pts):
        """ computes values of shape functions at pts (npts, 2) """
        V = p2d.Vandermonde2D(self.nodal_pts, self.p)
        VTi = np.linalg.inv(V.T)
        P_tilde = p2d.P_tilde(pts, self.p)[:, 0:self.nb]
        shap = np.dot(VTi, P_tilde.T)
        return shap

    def shape_function_derivatives_at_pts(self, pts):
        """ compute the derivatives of shape fns in ξ, η directions at pts
        returns list of derivatives of shape functions indexed by coord direction
        """
        V = p2d.Vandermonde2D(self.nodal_pts, self.p)
        dψ_dξ, dψ_dη = p2d.GradVandermonde2D(p=self.p, ξ=pts[:,0], η=pts[:,1])
        Vinv = np.linalg.inv(V)
        shap_der = [np.dot(dψ_dξ, Vinv), np.dot(dψ_dη, Vinv)]
        return shap_der

    def mk_uniform_nodal_pts(self):
        """ make uniformly spaced pts (barycentric sense) on the master element """
        uniform_bary_coords = bct.uniform_bary_coords(self.p)
        xp, yp = bct.bary2cart(self.verts, uniform_bary_coords)
        nodal_pts = np.vstack((xp, yp)).T
        return nodal_pts

    def mk_warped_nodal_pts(self):
        """ create uniform nodal points, then shift them """
        uniform_bary_coords = bct.uniform_bary_coords(self.p)
        shift = self.warpshift_pts(uniform_bary_coords)


    def warpshift_pts(self, bary_coords):
        """ shift the nodal points for better interpolation behavior
        @param bary_coords  the barycentric coordinates of the points to be shifted
        """
        pass

class NodalBasis2DQuad(NodalBasis2D):
    def __init__(self):
        self.basis_domain_verts = MASTER_ELEMENT_VERTICES['QUAD']


