#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# vertex definitions for master Tri and Quad, over which to construct the bases
import src.fem_base.master.barycentric_coord_tools as bct
import src.fem_base.master.polynomials_2D as p2d
import src.fem_base.master.nodal_basis_1D as NB1D

class NodalBasis2D(object): pass

class NodalBasis2DTriangle(NodalBasis2D):
    verts = ((-1, -1), (1, -1), (-1, 1))
    def __init__(self, p, nodal_locations='UNIFORM'):
        """ creates a nodal basis over master element verts
        @param p  polynomial order of the nodal basis
        @param nodal_locations
            UNIFORM: uniformly spaced nodal points (in a barycentric sense)
            WARPED:  shifted points for better interpolation behavior (Hesthaven 2008)
        """
        self.p, self.nb = p, int((p+1)*(p+2)/2.)
        if nodal_locations == 'UNIFORM':
            self.nodal_pts = self.mk_uniform_nodal_pts()
        elif nodal_locations == 'WARPED':
            self.nodal_pts = self.mk_warped_nodal_pts()
        else: raise ValueError('node_spacing {} not recognized'.format(node_spacing))

    def shape_functions_at_pts(self, pts):
        """ computes values of shape functions at pts (npts, 2) """
        V = p2d.Vandermonde2D(N=self.p, ξ=self.nodal_pts[:,0], η=self.nodal_pts[:,1])
        VTi = np.linalg.inv(V.T)
        P_tilde = p2d.Vandermonde2D(N=self.p, ξ=pts[:,0], η=pts[:,1])
        shap = np.dot(VTi, P_tilde.T)
        return shap.T

    def shape_function_derivatives_at_pts(self, pts):
        """ compute the derivatives of shape fns in ξ, η directions at pts
        returns list of derivatives of shape functions indexed by coord direction
        """
        V = p2d.Vandermonde2D(N=self.p, ξ=self.nodal_pts[:,0], η=self.nodal_pts[:,1])
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

    def mk_warped_nodal_pts(self, α=0.):
        """ compute warped nodal points on the order p master triangle
        @param α  blending constant, see Hesthaven p.179
        NOTE: Hesthaven uses scaled barycentric coords, so we transcribe
            Nodes2D.m, xytors.m directly rather than rescale things. This is done
            for readability in comparison to the Hesthaven text.
        """
        p = self.p
        λ1, λ2, λ3 = bct.uniform_bary_coords(p)
        x, y = -λ2 + λ3, (-λ2 - λ3 + 2*λ1) / np.sqrt(3)

        # warping and blending functions
        wt1, wt2, wt3 = self._w_t(p, λ3-λ2), 0.5*self._w_t(p, λ1-λ3), 0.5*self._w_t(p,λ2-λ1)
        b1, b2, b3 = 4*λ3*λ2, 4*λ3*λ1, 4*λ2*λ1
        w1, w2, w3 = wt1*b1*(1+(α*λ1)**2), wt2*b2*(1+(α*λ2)**2), wt3*b3*(1+(α*λ3)**2)

        # move the cartesian points on the equilateral triangle
        x += (1)*w1 + (-1)        *w2 + (-1)         *w3
        y += (0)*w1 + (np.sqrt(3))*w2 + (-np.sqrt(3))*w3

        # map back to master element from equilateral triangle
        λ1, λ2, λ3 = (np.sqrt(3)*y+1)/3, (-3*x-np.sqrt(3)*y+2)/6, (3*x-np.sqrt(3)*y+2)/6
        xp, yp = -λ2 + λ3 - λ1, -λ2 -λ3 + λ1
        nodal_pts = np.vstack((xp, yp)).T
        return nodal_pts

    def _w_t(self, p, pts):
        """ evaluate 1D warp factor w_tilde at order N at pts
        NOTE: see Hesthaven p.176
        """
        nb1d = NB1D.NodalBasis1D(p=p, node_spacing='EQUIDISTANT')
        shap = nb1d.shape_functions_at_pts(pts)

        r_eq = nb1d.nodal_pts
        r_LGL, _ = NB1D.LegendreGaussLobatto(nb1d.nb)

        numerator = np.dot(shap, r_LGL - r_eq)
        denom = 1 - pts**2
        denom[np.isclose(denom, 0.)] = 1  # don't divide by 0, numerator 0 here
        wr = numerator / denom
        return wr

class NodalBasis2DQuad(NodalBasis2D):
    def __init__(self): pass


