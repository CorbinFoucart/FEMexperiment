# -*- coding: utf-8 -*-
#

class Master1D(object):
    """ minimalist 1D master object, for use in purely 1D problems
    detailed explanation in tutorial/1D_modes_and_nodes.ipynb
    """
    def __init__(self, p, nquad_pts=None, *args, **kwargs):
        self.p, self.nb = p, p+1
        self.basis = NodalBasis1D(p=p, **kwargs)
        self.nodal_pts = self.basis.nodal_pts
        self.nq = 2*self.p + 2 if nquad_pts is None else nquad_pts
        self.quad_pts, self.wghts = GaussLegendre(self.nq)

        # shape functions at nodal and quadrature points
        self.shap_nodal, self.dshap_nodal = self.mk_shap_and_dshap_at_pts(self.nodal_pts)
        self.shap_quad,  self.dshap_quad = self.mk_shap_and_dshap_at_pts(self.quad_pts)

        # mass, stiffness matrices
        self.M, self.S, self.K = self.mk_M(), self.mk_S(), self.mk_K()
        self.Minv = np.linalg.inv(self.M)

        # lifting permuatation matrix L (0s, 1s)
        self.L = self.mk_L()

    def mk_shap_and_dshap_at_pts(self, pts):
        shap = self.basis.shape_functions_at_pts(pts)
        dshap = self.basis.shape_function_derivatives_at_pts(pts)
        return shap, dshap

    def mk_M(self):
        """ the mass matrix, M_ij = (phi_i, phi_j) """
        shapw = np.dot(np.diag(self.wghts), self.shap_quad)
        M = np.dot(self.shap_quad.T, shapw)
        return M

    def mk_S(self):
        """ the stiffness matrix, S_ij = (phi_i, \frac{d\phi_j}{dx}) """
        dshapw = np.dot(np.diag(self.wghts), self.dshap_quad)
        S = np.dot(self.shap_quad.T, dshapw)
        return S

    def mk_K(self):
        """ the stiffness matrix, K_ij = (\frac{d\phi_i}{dx}, \frac{d\phi_j}{dx}) """
        dshapw = np.dot(np.diag(self.wghts), self.dshap_quad)
        K = np.dot(self.dshap_quad.T, dshapw)
        return K

    def mk_L(self):
        L = np.zeros((self.nb, 2))
        L[0, 0]  = 1
        L[-1, 1] = 1
        return L

    @property
    def shap_der(self):
        """ return the shape derivatives for apps expecting 2, 3D"""
        return [self.dshap_quad]
