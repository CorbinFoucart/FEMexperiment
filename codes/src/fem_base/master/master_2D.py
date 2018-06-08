#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import quadpy

import src.fem_base.master.nodal_basis_2D as nb2d
import src.fem_base.master.barycentric_coord_tools as bct
import src.fem_base.master.master_1D as m1d

class Master2D(object):

    def mk_shap_and_dshap_at_pts(self, pts):
        shap = self.basis.shape_functions_at_pts(pts)
        dshap = self.basis.shape_function_derivatives_at_pts(pts)
        return shap, dshap

class Master2DTriangle(Master2D):
    """ note vertex definitions in nodal_basis_2D.py """
    def __init__(self, p, nquad_pts=None, *args, **kwargs):
        self.p, self.dim = p, 2
        self.basis = nb2d.NodalBasis2DTriangle(self.p, **kwargs)
        self.nb, self.verts, self.n_ed = self.basis.nb, self.basis.verts, 3
        self.nodal_pts = self.basis.nodal_pts
        self.nq = 2*self.p +2 if nquad_pts is None else nquad_pts
        self.quad_pts, self.wghts = triangle_quadrature(self.nq, self.verts)

        # shape functions at nodal and quadrature points
        self.shap_quad,  self.dshap_quad = self.mk_shap_and_dshap_at_pts(self.quad_pts)
        _, self.dshap_nodal = self.mk_shap_and_dshap_at_pts(self.nodal_pts)

        # mass, stiffness matrices
        self.M, self.S, self.K = self.mk_M(), self.mk_S(), self.mk_K()
        self.Minv = np.linalg.inv(self.M)

        # edge data structures, master edge, nodes on the edge, lifting matrix, normals
        self.master_edge = [m1d.Master1D(p=self.p)]
        self.ids_ed = self.find_nodes_on_edges()
        self.nr = sum([len(ids) for ids in self.ids_ed])
        self.L = self.mk_L()
        self.edge_normals = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [-1, 0], [0, -1]])

    def mk_M(self):
        """ the mass matrix, M_ij = (phi_i, phi_j) """
        shapw = np.dot(np.diag(self.wghts), self.shap_quad)
        M = np.dot(self.shap_quad.T, shapw)
        return M

    def mk_S(self):
        """ the stiffness matrix, S[k]_ij = (phi_i, \frac{d\phi_j}{dx_k})
        returns a list indexed by coordinate direction on the master element
        """
        S = [None, None]
        for i in range(self.dim):
            dshapw = np.dot(np.diag(self.wghts), self.dshap_quad[i])
            S[i] = np.dot(self.shap_quad.T, dshapw)
        return S

    def mk_K(self):
        """ the stiffness matrix, K_ij = (\frac{d\phi_i}{dx_k}, \frac{d\phi_j}{dx_k})
        returns a list indexed by coordinate direction on the master element
        """
        K = [None, None]
        for i in range(self.dim):
            dshapw = np.dot(np.diag(self.wghts), self.dshap_quad[i])
            K[i] = np.dot(self.dshap_quad[i].T, dshapw)
        return K

    def find_nodes_on_edges(self):
        """ computes the node numbers (ids) on each edge
        the i^th barycentric coord of a point on a tri edge will be 0, find these pts
        @retval ids_ed  list of vectors indexed by edge number
        NOTE: we manually flip edges 0 and 2 to ensure CCW ordering
            of ed dof around the element
        """
        ids_ed = [None, None, None]
        bary_coords = bct.cart2bary(self.verts, self.nodal_pts.T)
        ids_ed[0] = np.where( np.isclose(bary_coords[0, :], 0.) )[0][::-1]
        ids_ed[1] = np.where( np.isclose(bary_coords[1, :], 0.) )[0]
        ids_ed[2] = np.where( np.isclose(bary_coords[2, :], 0.) )[0][::-1]
        return ids_ed

    def mk_L(self):
        """ makes the elemental lifting matrix """
        L = np.zeros((self.nb, self.nr), dtype=int)
        for ed_dof, interior_dof in enumerate(np.hstack(self.ids_ed)):
            L[interior_dof, ed_dof] = 1
        return L

class Master2DQuad(Master2D): pass

def triangle_quadrature(n, verts):
    """ look up / compute quadrature rule over the triangle, order n
    @param n  the order of polynomial which should be integrated exactly
    @param verts  tuple of tuples defining the master element
    NOTE: leverages quadpy, 2*weights
    """
    if n > 50:
        raise NotImplementedError
    qr = quadpy.triangle.xiao_gimbutas.XiaoGimbutas(index=n)
    bary, weights = qr.bary, qr.weights
    xq, yq = bct.bary2cart(verts=verts, _lambda=bary.T)
    points = np.vstack((xq, yq)).T
    return points, 2*weights
