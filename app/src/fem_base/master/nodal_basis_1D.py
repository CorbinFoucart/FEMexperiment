# -*- coding: utf-8 -*-
#

class NodalBasis1D(object):
    """ minimalist nodal basis object: 
    efficiently computes shape functions and their derivatives
    detailed explanation in tutorial/1D_basis_and_master_element.ipynb 
    """
    def __init__(self, p, node_spacing='GAUSS_LOBATTO'):
        self.nb = p + 1
        if node_spacing == 'GAUSS_LOBATTO':
            self.nodal_pts, _ = LegendreGaussLobatto(self.nb)
        elif node_spacing == 'EQUIDISTANT':
            self.nodal_pts = np.linspace(-1, 1, self.nb)
        else: raise ValueError('node_spacing {} not recognized'.format(node_spacing))
    
    def shape_functions_at_pts(self, pts):
        """ computes shape functions evaluated at pts on [-1, 1]
        @retval shap (len(pts), nb) phi_j(pts[i])
        """
        V = Vandermonde1D(N=self.nb, x=self.nodal_pts)
        VTinv = np.linalg.inv(V.T)
        P = P_tilde(r=pts, N=self.nb)
        shap = np.dot(VTinv, P)
        return shap.T
    
    def shape_function_derivatives_at_pts(self, pts):
        """ computes shape function derivatives w/r/t x on [-1, 1]
        @retval shap_der, (Dr in Hesthaven), (len(pts), nb) d/dx phi_j(pts[i])
        """
        V  = Vandermonde1D(N=self.nb, x=self.nodal_pts)
        Vx = GradVandermonde1D(N=self.nb, x=pts)
        Vinv = np.linalg.inv(V)
        shap_der = np.dot(Vx, Vinv)
        return shap_der