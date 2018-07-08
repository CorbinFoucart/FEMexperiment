"""
Computes the finite element map data structures, jacobians, ed, nrmls etc
TODO improve description once code is finalized
@author foucartc
    @retval Jinv  inverse of Jacobian matrix of isoparametric transform at every
    dgnode Jinv[elmType][dof_per_elm, dim (phys coord), dim (master coord), elm]
    such that Jinv[et][0, i, j, k] jacobian inverse dr/dx at dof 0 on element k
"""
import numpy as np

def quad_Jinv_and_detJ(master, dgnodes):
    """ computes the inverse Jacobian (Jinv) and detJ transforms at quadrature
    points
    @param master  list of master elements
    @param dgnodes  list of dgnodes ndarrays arrays
    @retval detJ  determinants of the Jacobian matrices for each dgnode shape
        (dof_per_elm, nelm)
    """
    shap_der_list = [M.dshap_quad for M in master]
    Jinv, detJ = _Jinv_and_detJ(shap_der_list, dgnodes)
    return Jinv, detJ

def nodal_Jinv_and_detJ(master, dgnodes):
    """ computes the inverse Jacobian (Jinv) and detJ transforms at the dg nodal
    points
    @param master  list of master elements
    @param dgnodes  list of dgnodes ndarrays arrays
    @retval J List of arrays of Jacobians at each point. That is
        J[:, i, j, K] is J_ij of
        jacobian matrix J at the points on elm K, where the entry is
        \f$\frac{\partial x_i}{\partial \ xi_j}\f$
    @retval detJ  determinants of the Jacobian matrices for each dgnode shape
        (dof_per_elm, nelm)
    """
    shap_der_list = [M.dshap_nodal for M in master]
    Jinv, detJ = _Jinv_and_detJ(shap_der_list, dgnodes)
    return Jinv, detJ

class FEM_Mapping(object): pass
class Isoparametric_Mapping(FEM_Mapping):
    map_fns = {'QUAD':quad_Jinv_and_detJ, 'NODAL':nodal_Jinv_and_detJ}

    def __init__(self, master, dgnodes, map_nodes='QUAD'):
        self.Jinv, self._detJ = self.map_fns[map_nodes](master, dgnodes)

    def compute_normals(self):
        """ computes the element normal vectors """
        pass

    def elm_Jinv(self, elmType, elm): return self.Jinv[elmType][:,:,:,elm]
    def elm_detJ(self, elmType, elm): return self._detJ[elmType][:,elm]

class Affine_Mapping(Isoparametric_Mapping):
    """ this is only a mock class, it actually works like an isoparametric mapping,
    but returns a single detJ and Jinv over each element
    TODO: do the implementation of this in the normal cheap way.
    """
    def elm_Jinv(self, elmType, elm): return self.Jinv[elmType][0,:,:,elm]
    def elm_detJ(self, elmType, elm): return self._detJ[elmType][0,elm]

def _Jinv_and_detJ(shap_der_list, dgnodes):
    """ computes the inverse Jacobian (Jinv) and detJ transforms at quadrature points
    @param shap_der_list list of shape function derivates list by element type
        where shap_der_list[elmType][master elm coord][nodes, n_nbasis_fns] i.e.,
        each element of the top level list is a list of derivative matrices indexed
        by coordinate direction, with each entry an ndarray with shape (pts, nb)
        where pts denote the master element points at which the derivatives are
        evaluated at the corresponding physical space point.
    @param dgnodes  list of dgnodes ndarrays arrays
    @retval detJ  determinants of the Jacobian matrices for each dgnode shape
        (dof_per_elm, nelm)
    """
    dim = dgnodes[0].shape[1]
    Jinv, detJ = list(), list()
    J_inplace_inversion_fn = inv_J_fns[dim]
    for elmType, dgn in enumerate(dgnodes):
        J = _jacobian(shap_der_list[elmType], dgn)
        etype_detJ = _detJ(J, dim)
        detJ.append(etype_detJ)
        Jinv.append( J_inplace_inversion_fn(J, etype_detJ) )
    return Jinv, detJ

################################################################################
# JACOBIAN, JACOBIAN DETERMINANT, INVERSE JACOBIAN COMPUTATIONS
################################################################################
def _jacobian(shap_der_at_nodes, nodes):
    """ calculuates the jacobian matrices of an isoparametric transformation at
    the given nodes
    @param shap_der_at_nodes  list of derivative matrices indexed by coordinate
        direction, with each entry an ndarray with shape (pts, nb) where pts denote
        the master element points at which the derivatives are evaluated at the
        corresponding physical space point.
    @param nodes  ndarray of physical space nodal points that define each
        element shape (nb, dim, nElm) @retval J List of arrays of Jacobians at each
        point. That is J[:, i, j, K] is the array of jacobian matrices at the points
        on elm K, where the entry is \f$\frac{\partial x_i}{\partial \ xi_j}\f$
    """
    nb, dim, nelm = nodes.shape
    npts, _ = shap_der_at_nodes[0].shape
    J = np.zeros((npts, dim, dim, nelm))
    for x_i in range(dim):
        for xi_j, ddxi in enumerate(shap_der_at_nodes):
            J[:, x_i, xi_j, :] = np.dot(ddxi, nodes[:, x_i, :])
    return J

def _detJ(J, dim):
    """ computes the detJ for Jacobian arrays as computed in _jacobian
    @param J  List of arrays of Jacobians for  every nodal point. That is
        J[:, i, j, K] is the array of jacobian matrices at the points on elm K,
        where the entry is \f$\frac{\partial x_i}{\partial \ xi_j}\f$
    """
    detJ = det_J_fns[dim](J)
    return detJ

# set of functions that compute detJ or invert Jacobian arrays
def _detJ_1D(J):
    detJ = np.copy(J[:,0,0,:])
    return detJ

def _inv_Jacobian_1D(J, detJ):
    """ manually invert 1x1 jacobians J in place """
    J[:,0,0,:] = 1./detJ
    return J

def _detJ_2D(J):
    detJ = J[:,0,0,:] * J[:,1,1,:] - J[:,0,1,:] * J[:,1,0,:]
    return detJ

def _inv_Jacobian_2D(J, detJ):
    """ manually invert 2x2 jacobians J in place """
    tmp           =  J[:, 1, 1, :] / detJ
    J[:, 0, 1, :] = -J[:, 0, 1, :] / detJ
    J[:, 1, 0, :] = -J[:, 1, 0, :] / detJ
    J[:, 1, 1, :] =  J[:, 0, 0, :] / detJ
    J[:, 0, 0, :] = tmp
    return J

def _detJ_3D(J):
    """ manually compute determinant of 3x3 matrices in J format """
    detJ =   J[:,0,0,:] * (J[:,1,1,:] * J[:,2,2,:] - J[:,2,1,:] * J[:,1,2,:]) \
           - J[:,1,0,:] * (J[:,0,1,:] * J[:,2,2,:] - J[:,2,1,:] * J[:,0,2,:]) \
           + J[:,2,0,:] * (J[:,0,1,:] * J[:,1,2,:] - J[:,1,1,:] * J[:,0,2,:])
    return detJ

def _inv_Jacobian_3D(): raise(NotImplementedError)

# key is dim, value is function
inv_J_fns = {1:_inv_Jacobian_1D, 2:_inv_Jacobian_2D, 3:_inv_Jacobian_3D}
det_J_fns = {1:_detJ_1D, 2:_detJ_2D, 3:_detJ_3D}

################################################################################
# NORMAL VECTOR COMPUTATIONS
################################################################################

def _nrmls_1D(): raise(NotImplementedError)

def _nrmls_2D(shap_der, nodes_ed):
    """ computes 2D normals
    @param shap_der  list of derivative matrices indexed by coordinate
        direction, with each entry an ndarray with shape (pts, nb) where pts denote
        the master element points at which the derivatives are evaluated at the
        corresponding physical space point.
    @param nodes  ndarray of physical space nodal points that define each
        edge shape (ng, dim, nEdges) @retval J List of arrays of Jacobians at each
        point.
    NOTE: The derivates are all computed on the master edge, and there is only one
        differentiation direction.
    """
    ng, dim, n_edges = nodes_ed.shape
    normals = np.zeros_like(nodes_ed)
    for ed in range(n_edges):
        derivatives = np.dot(shap_der, nodes_ed[:, :, ed])
        dxdξ, dydξ  = derivatives[:,0], derivatives[:,1]
        I = np.sqrt(dxdξ**2 + dydξ**2)
        nrml = 1/np.abs(I)[:, np.newaxis] * np.vstack((dydξ, -dxdξ)).T
        normals[:, :, ed] = nrml
    return normals

def _nrmls_3D(): raise(NotImplementedError)

nrml_fns = {2:_nrmls_2D}

################################################################################
# graveyard MPU code
################################################################################
#elif dim == 3:

#    inv = [sp.zeros(jmt.shape) for jmt in JMT]
#    for k in range(len(dgnodes)):
#        inv[k][:, 0, 0, :] = \
#            (JMT[k][:, 1, 1, :] * JMT[k][:, 2, 2, :]\
#            - JMT[k][:, 2, 1, :] * JMT[k][:, 1, 2, :]) / J[k]
#        inv[k][:, 0, 1, :] = \
#            -(JMT[k][:, 0, 1, :] * JMT[k][:,2, 2, :]\
#            - JMT[k][:, 2, 1, :] * JMT[k][:, 0, 2, :]) / J[k]
#        inv[k][:, 0, 2, :] = \
#            (JMT[k][:, 0, 1, :] * JMT[k][:, 1 ,2, :]\
#            - JMT[k][:, 1, 1, :] * JMT[k][:, 0, 2, :]) / J[k]
#        inv[k][:, 1, 0, :] = \
#            -(JMT[k][:, 1, 0, :] * JMT[k][:, 2, 2, :]\
#            - JMT[k][:, 2, 0, :] * JMT[k][:, 1, 2, :]) / J[k]
#        inv[k][:, 1, 1, :] = \
#            (JMT[k][:, 0, 0, :] * JMT[k][:, 2, 2, :]\
#            - JMT[k][:, 2, 0, :] * JMT[k][:, 0, 2, :]) / J[k]
#        inv[k][:, 1, 2, :] = \
#            -(JMT[k][:,0, 0, :] * JMT[k][:,1, 2, :]\
#            - JMT[k][:, 1, 0, :] * JMT[k][:, 0, 2, :]) / J[k]
#        inv[k][:, 2, 0, :] = \
#            (JMT[k][:, 1, 0, :] * JMT[k][:, 2, 1, :]\
#            - JMT[k][:, 2, 0, :] * JMT[k][:, 1, 1, :]) / J[k]
#        inv[k][:, 2, 1, :] = \
#            -(JMT[k][:, 0, 0, :] * JMT[k][:, 2, 1, :]\
#            - JMT[k][:, 2, 0, :] * JMT[k][:, 0, 1, :]) / J[k]
#        inv[k][:, 2, 2, :] = \
#            (JMT[k][:, 0, 0, :] * JMT[k][:, 1, 1, :]\
#            - JMT[k][:, 1, 0, :] * JMT[k][:, 0, 1, :]) / J[k]

#    #Over-writing JMT...
#    factors = JMT
#    #Now multiply the mass-matrix inverse by the right-hand side
#    #(This is done explicitly)
#    factors = inv

#return J, factors

