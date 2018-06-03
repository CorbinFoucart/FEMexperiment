# -*- coding: utf-8 -*-
""" @package src.master.mk_master Makes the master class which contains all
operators for the master element, including index arrays for orienting faces
for face operations, and functions for calculating Jacobians [?].

Created on Thu Mar  3 22:51:54 2011

@author Matt Ueckermann
@author Corbin Foucart
"""
import src.fem_base.master.mk_basis as mkb
import src.fem_base.master.mk_basis_nodal as mkbn
from src.fem_base.master.mk_cubature import get_pts_weights
from numpy import zeros, array, dot, diag, arange, shape
import src.pyutil as pyutil

#===============================================================================
# Master class for NODAL bases
#===============================================================================

class Master_nodal:
    """@class Master_nodal
    @brief Class that creates and contains the master object for nodal bases
    __init__(self, n, dim, element, init_type='all', n_quad=None):

    """
    def __init__(self, order, dim, element, init_type='all', n_quad=None):
        """Class constructor
        @brief This is where all the action happens

        @param n (\c int) order of the basis

        @param dim (\c int) dimension of the problem

        @param element (\c int) the type of element (see int_el_pqr)

        @param init_type (\c string) The type of initialization
                        (not case sensitive):
                         'QF', 'FEMOP': Make the mass and stiffness matrices
                                        for quadrature-free implementations of
                                        linear operators
                         'QB', 'SHAP' : Make the gaussian quadrature weights,
                                        points, and evaluate the polynomial
                                        bases (and their first derivatives)
                                        at these points
                         'ALL', 'BOTH': Make both the QF and QB operators

        @param n_quad (\c int) The order of the quadrature for QB integration.
                      Usually taken as n_quad = 2 * n (DEFAULT) which is exact
                      for linear operators.

        @see src.master.mk_basis.int_el_pqr

        @author Matt Ueckermann
        """
        n = order

        ## Order of the basis
        self.n = n
        ## Dimension of the basis
        self.dim = dim
        ## Basis element type
        self.element=element

        #Make the polynomial basis
        basis = mkbn.Basis_nodal(n, dim, element)

        #Save it for later to be used to create the rest of the operators
        ## The polynomial basis used to create FEM operators.
        # The solution on every element is defined by
        # \f$f(t,\mathbf x) = \sum_{i=0}^{n_b} f_i(t) \theta(\mathbf x)\f$
        # where \f$\theta(\mathbf x)\f$ is defined by basis
        self.basis = basis
        self.nodal_pts = self.basis.nodal_pts

        #Get number of bases
        ## The number of bases
        self.nb = basis.nb
        ## The number of edges
        self.ne = basis.ne
        ## Number of vertices
        self.nv = len(basis.vol_verts)

        ##The number of edge element types.
        #For example, a prism has two element types on edges, 0-triangles and
        #1-quadrilaterals
        self.n_ed_type = len(pyutil.unique(basis.ed_type))

        if n_quad == None:
            if type(n) == list:
                self.n_quad = max(n) * 2
            else:
                self.n_quad = n * 2
        else:
            self.n_quad = n_quad

        self._mkelmSHAP(self.n_quad)

        if init_type.lower() in ['qf', 'femop', 'femops', 'all',' both']:
            self._mkelmFEM()

        #Finally, make the edge orientation matrix
        self._mk_ed_orient()

        # Make the Lifting reference matrix
        ## The lifting reference matrix is a useful re-indexing matrix for
        # relating edge unknowns to volume unknowns. For example, given a vector
        # e = [ed1_dofs; ed2_dofs; ... ; edn_dofs], L*e will re-reference this
        #numbering to the volume number of those same degrees of freedom (dofs)
        #and sum any overlapping elements. The reverse can also be done. To take
        #volume dofs and copy them to the edge unknowns, you can use L.T * f,
        #where f = [volume_dofs[0]; volume_dofs[1]; ... ; volume_dofs[m]]
        tot_nb_ed = 0
        for eds in self.basis.nodal_ed_ids:
            tot_nb_ed += len(eds)
        self.L = zeros((self.nb, tot_nb_ed))
        ns = 0
        for eor in self.ed_orient:
            ne = ns + len(eor[0])
            self.L[array(eor[0]), arange(ns, ne)] = 1.
            ns = ne

        # CF: save nb_e (# dof on edge space (red)) as an attribute, which we
        # can take from the columns of the L matrix (nb x nr)
        self.nr = shape(self.L)[1]

        # CF:  compute the derivatives of the nodal basis shape functions at the
        # nodal points directly via the basis
        self.nodal_shap_der = [basis.eval_at_pts(basis.nodal_pts, der_dim=i) for i in range(basis.dim)]

    def _mkelmFEM(self):
        """This creates the linear Finite element operators used on the element

        @author Matt Ueckermann
        """

        #Shorter variable names
        dim = self.dim
        shap = self.shap
        shapw = dot(diag(self.cube_wghts), shap)

        #helper variables (shorter names essentially)

        # Make the mass-matrix for the master element.
        ## The reference or master mass matrix
        # The mass matrix is defined by:
        # \f$M_{i\!j} = \int_K \theta_j(\mathbf x)\theta_i(\mathbf x)dK\f$
        #, where \f$K\f$ isfo the reference or master element.
        self.M = dot(shap.T, shapw)

        # Make the derivative matrices.
        ## The reference derivative (stiffness) matrix
        # The derivative matrix is defined by:
        # \f$\mathcal K_{ki\!j} = \int_K \theta_j(\mathbf x)
        # \frac{\partial}{\partial x_k} \theta_i(\mathbf x)dK\f$
        #, where \f$K\f$ is the reference or master element.
        self.K = [dot(self.shap_der[i].T, shapw) for i in range(dim)]

    def _mkelmSHAP(self, n_quad):
        '''Create the shape matrices for quadrature-based integration

        @param n_quad (\c int) The polynomial order which the quadrature should
                        integrate exactly
        '''
        ## Stores the coordinates of the quadrature points -- coordinate frame
        # of the master element. cube_pts.shape = (n_cube_pts, dim)
        self.cube_pts = None
        ## Stores the weights of the quadrature points.
        # cube_wghts.shape = (n_cube_pts, 1)
        self.cube_wghts = None
        #Because we used tensor products for the nodal bases, we need to use the
        #tensor-product quadrature rules to accurately integrate the polynomials
        #on them. This applies ONLY for element 1, since simplexes are not
        #affected, and the cubature rules for the prism were constructed using
        #a tensor product of the 2D triangle points -- consistent with the
        #nodal basis.
        if self.element == 1:
            force_mk = True
        else:
            force_mk = False

        self.cube_pts, self.cube_wghts = \
            get_pts_weights(n_quad, self.dim, self.element, force_mk)

        ## Matrix that transforms a set of vertices to cubature points.
        # This is needed to create the real-space cubature edge-points which is
        # used for initialization, visualization, and any function-evaluations
        # (from source terms, for example).
        # Cube_pts_in_real_space = dot(cube_pts2xy, verts)
        # where verts is something like shape(verts)=(n_verts, dim*n_elm), or
        #verts = [x1, y1, z1, x2, y2, z2 ... ], where x1 is a column
        #vector with the x-coordinates of the vertices of the first element
        self.cube_pts2xy = self.basis.mkmapcoords(self.cube_pts)

        ## The number of cubature points/weigths
        self.n_cube_pts = len(self.cube_wghts)

        ## The shape matrix evaluates each basis at the cubature points.
        # \f$shap_{i\!j} = \theta_j(\mathbf x_i)\f$
        # CF: note that the shape matrix here has (rows, cols) = (ncube_pts, nb)
        self.shap = self.basis.eval_at_pts(self.cube_pts)

        ## The shape derivative matrices evaluates the derivatives of each
        #  basis at the cubature points.
        # \f$shap_{der}[k]_{i\!j} = \partial_{x_k} \theta_j(\mathbf x_i) \f$
        self.shap_der = [self.basis.eval_at_pts(self.cube_pts, i) \
            for i in range(self.dim)]

    def _mk_ed_orient(self):
        """Creates the permutation arrays that allows one to match the points on
        faces that are rotated relative to each other"""
        ##The edge orientation or permutation matrix that re-orders edges such
        #that they can be aligned with edges from other elements.
        #The order numbers are defined by src.sol._get_orient_num
        self.ed_orient = mkb.mk_ed_orient(self.basis.nodal_ed_ids, \
            self.basis.ed_type, self.n, self.dim)

#===============================================================================
# Master class for Modal bases
#===============================================================================

class Master:
    """@class Master
    @brief Class that creates and contains the master object
    __init__(self, n, dim, element, init_type='all', n_quad=None):

    """
    def __init__(self, n, dim, element, init_type='all', n_quad=None):
        """Class constructor
        @brief This is where all the action happens

        @param n (\c int) order of the basis

        @param dim (\c int) dimension of the problem

        @param element (\c int) the type of element (see int_el_pqr)

        @param init_type (\c string) The type of initialization
                        (not case sensitive):
                         'QF', 'FEMOP': Make the mass and stiffness matrices
                                        for quadrature-free implementations of
                                        linear operators
                         'QB', 'SHAP' : Make the gaussian quadrature weights,
                                        points, and evaluate the polynomial
                                        bases (and their first derivatives)
                                        at these points
                         'ALL', 'BOTH': Make both the QF and QB operators

        @param n_quad (\c int) The order of the quadrature for QB integration.
                      Usually taken as n_quad = 2 * n (DEFAULT) which is exact
                      for linear operators.

        @see src.master.mk_basis.int_el_pqr

        @author Matt Ueckermann
        """

        if dim == 1:
            print("Master class currently only works for 2D or 3D meshes.")

        #Add doxypy tags to document member variables
        ## Order of the basis
        self.n = n
        ## Dimension of the basis
        self.dim = dim
        ## Basis element type
        self.element=element

        #Make the polynomial basis
        basis = mkb.Basis(n, dim, element)

        #Save it for later to be used to create the rest of the operators
        ## The polynomial basis used to create FEM operators.
        # The solution on every element is defined by
        # \f$f(t,\mathbf x) = \sum_{i=0}^{n_b} f_i(t) \theta(\mathbf x)\f$
        # where \f$\theta(\mathbf x)\f$ is defined by basis
        self.basis = basis

        #Get number of bases
        ## The number of bases
        self.nb = basis.nb
        ## The number of edges
        self.ne = basis.ne
        ## Number of vertices
        self.nv = len(basis.vol_verts)

        #Now do the same for the edges:
        basis_e = [mkb.Basis(n, dim - 1, e_type) \
            for e_type in pyutil.unique(basis.ed_type)]

        ##List containing the bases existing on the element edges
        self.basis_e = basis_e

        ##The number of bases for the edges of the elements
        self.nb_e = basis_e[0].nb

        ##The number of edge element types.
        #For example, a prism has two element types on edges, 0-triangles and
        #1-quadrilaterals
        self.n_ed_type = len(pyutil.unique(basis.ed_type))

        ##The number of vertices in each type of edge element
        self.nv_ed = [len(base.vol_verts) for base in basis_e]

        if init_type.lower() in ['qf', 'femop', 'femops', 'all',' both']:
            self._mkelmFEM()
            self._mkedFEM()

            self._mkFEMfloat()

        if init_type.lower() in ['qb', 'shap', 'all',' both']:
            if n_quad == None:
                self.n_quad = n * 2
            else:
                self.n_quad = n_quad

            self._mkelmSHAP(self.n_quad)
            self._mkedSHAP(self.n_quad)

    def _mkelmFEM(self):
        """This creates the linear Finite element operators used on the element

        @author Matt Ueckermann
        """

        #Shorter variable names
        n = self.n
        nb = self.nb
        dim = self.dim
        element = self.element
        basis = self.basis

        #helper variables (shorter names essentially)
        aij = basis.coeffs
        pqr = basis.monoms
        pqr_int = basis.elmint_monoms

        # Make the mass-matrix for the master element.
        ## The symbolic version of the reference or master mass matrix
        # The mass matrix is defined by:
        # \f$M_{i\!j} = \int_K \theta_j(\mathbf x)\theta_i(\mathbf x)dK\f$
        #, where \f$K\f$ isfo the reference or master element.
        self.M_sym = mkb.inprod_mat(aij, aij, pqr, pqr_int)

        # Make the derivative matrices.
        ## The symbolic version of the reference derivative (stiffness) matrix
        # The derivative matrix is defined by:
        # \f$\mathcal K_{ki\!j} = \int_K \theta_j(\mathbf x)
        # \frac{\partial}{\partial x_k} \theta_i(\mathbf x)dK\f$
        #, where \f$K\f$ is the reference or master element.
        self.K_sym = [None] * dim
        for i in range(dim):
            self.K_sym[i] = mkb.inprod_mat(aij, \
                                       mkb.polyder_mat(aij, pqr, i), \
                                       pqr, pqr_int)

    def _mkedFEM(self):
        '''This create the FEM operators used on the edges of elements
        @private

        @author Matt Ueckermann
        '''

        #shorter names
        n = self.n
        dim = self.dim
        nb = self.nb
        ne = self.ne
        nb_e = self.nb_e
        n_ed_type = self.n_ed_type
        basis = self.basis
        basis_e = self.basis_e

        #Make the ed-mass-matrix for theta-theta basis  (element-element basis)
        #on each of the edges
        ## The symbolic version of the reference or master mass matrix on each
        # of the edges of the element.
        # The edge mass matrices are defined by:
        # \f$M^{ed\theta\theta}_{ki\!j} = \int_{\partial_k K} \theta_j(\mathbf x)
        # \theta_i(\mathbf x)d \partial_k K\f$
        #, where \f$\partial_k K\f$ is edge 'k' of the master element.
        self.M_ed_TT_sym = [None] * ne
        for i in range(ne):
            if n_ed_type == 1:
                bnum = 0
            else:
                bnum = basis.ed_type[i]

            aij_e = mkb.elm2edge(basis, i, basis_e[bnum])

            self.M_ed_TT_sym[i] = mkb.inprod_mat(aij_e, aij_e, \
                            basis_e[bnum].monoms, basis_e[bnum].elmint_monoms)

        #Make the ed-mass-matrix for theta-phi basis  (element-edge basis)
        #on each of the edges
        ## The symbolic version of the reference or master mass matrix on each
        # of the edges of the element using the edge-space as the test-space.
        # The edge mass matrices using the edge-space test function
        # are defined by:
        # \f$M^{ed\theta\phi}_{ki\!j} = \int_{\partial_k K} \theta_j(\mathbf x)
        # \phi_i(\mathbf x)d \partial_k K\f$
        # , where \f$\partial_k K\f$ is edge 'k' of the master element.
        #
        # This mass matrix can be used to project the volume-space polynomials
        # unto the edge-space basis. This is simple, since, consider:
        # \f{align*}{ \int_{\partial_k K} b_j \phi_j(\mathbf x)
        # \phi_i(\mathbf x)d \partial_k K &= \int_{\partial_k K} a_j
        # \theta_j(\mathbf x) \phi_i(\mathbf x)d \partial_k K \\
        # b_j \mathbf I_{i\!j} &= \mathbf M^{\theta\phi}_{i\!j} a_j \\
        # b_j &= \mathbf M^{\theta\phi}_{i\!j} a_j
        # \f}
        # , where \f$b_j, a_j\f$ are the coefficients using the edge-space and
        # volume space, respectively.

        self.M_ed_TP_sym = [None] * ne
        for i in range(ne):
            if n_ed_type == 1:
                bnum = 0
            else:
                bnum = basis.ed_type[i]
            aij_e = mkb.elm2edge(basis, i, basis_e[bnum])

            self.M_ed_TP_sym[i] = mkb.inprod_mat(aij_e, basis_e[bnum].coeffs, \
                            basis_e[bnum].monoms, basis_e[bnum].elmint_monoms)

        #Make the ed-mass-matrix for phi-phi basis  (edge-edge basis)
        #on each of the edges
        ## The symbolic version of the reference or master mass matrix on each
        # of the edges of the element using the edge-space only.
        # The edge mass matrices using the edge-space only
        # are defined by:
        # \f$M^{ed\phi\phi}_{ki\!j} = \int_{\partial_k K} \phi_j(\mathbf x)
        # \phi_i(\mathbf x)d \partial_k K\f$
        #, where \f$\partial_k K\f$ is an edge of type 'k', that is, either a
        #triangle or a square for a prismatic element. If there is only one
        #type of element, then k only take 1 value, k==0
        self.M_ed_PP_sym = [None] * n_ed_type
        for i in range(n_ed_type):
            if n_ed_type == 1:
                bnum = 0
            else:
                bnum = basis.ed_type[i]

            self.M_ed_PP_sym[i] = mkb.inprod_mat(basis_e[bnum].coeffs, \
                basis_e[bnum].coeffs, basis_e[bnum].monoms, \
                basis_e[bnum].elmint_monoms)

        #Finally, make the ed-mass-matrix for theta-phi-theta basis
        #(element-edge-element basis) on each of the edges. This is to capture
        #the full quadratic non-linearities

    def _mkFEMfloat(self):
        '''This creates floating point copies of the symbolic matrices created
        previously.
        @author Matt Ueckermann
        '''
        nb = self.nb
        dim = self.dim
        nb_e = self.nb_e
        ne = self.ne
        n_ed_type = self.n_ed_type
        #FINALLY Make float copies of the matrices (this is what actually gets used)
        ## The floating point version of the mass-matrix
        self.M = zeros((nb, nb), dtype=float)

        ## The floating point version of the longer stiffness-matrix.
        #here the order of indices are re-arranged:
        #\f$\mathcal K_{i\!jk} = \int_K \theta_j(\mathbf x)
        # \frac{\partial}{\partial x_k} \theta_i(\mathbf x)dK\f$,
        # that is, the direction of the derivative is now last.
        self.K = zeros((nb, nb, dim), dtype=float)

        ## The floating point version of the elemental mass-matrix.
        # Note the order of indices are re-arrnaged:
        #\f$M^{ed\theta\theta}_{i\!jk} = \int_{\partial_k K} \theta_j(\mathbf x)
        # \theta_i(\mathbf x)d \partial_k K\f$
        # that is, the edge-number is now last.
        self.M_ed_TT = zeros((nb, nb, ne), dtype=float)

        ## The floating point version of the elemental mass-matrix using edge-
        # space test functions.
        # Note the order of indices are re-arrnaged:
        # \f$M^{ed\theta\phi}_{ki\!j} = \int_{\partial_k K} \theta_j(\mathbf x)
        # \phi_i(\mathbf x)d \partial_k K\f$
        # that is, the edge-number is now last.
        self.M_ed_TP = zeros((nb, nb_e, ne), dtype=float)

        ## The floating point version of the elemental mass-matrix using the
        #edge space only (on different edges).
        # Note the order of indices are re-arrnaged:
        # \f$M^{ed\phi\phi}_{ki\!j} = \int_{\partial_k K} \phi_j(\mathbf x)
        # \phi_i(\mathbf x)d \partial_k K\f$
        # that is, the edge-number is now last.
        self.M_ed_PP = zeros((nb_e, nb_e, n_ed_type), dtype=float)

        ## The floating point version of the long mass-matrix.
        # Note the order of indices are re-arrnaged:
        # \f$M^{ed\theta\phi\theta}_{mi\!jk} = \int_{\partial_k K}
        # \theta_j(\mathbf x) \phi_k(\mathbf x) \theta_i(\mathbf x)
        # d \partial_k K\f$
        # that is, the edge-number is now last.
        self.M_ed_TPT = zeros((nb * nb_e, nb, ne), dtype=float)

        #This is faster and does the same thing
        self.M = array(self.M_sym, dtype=float)
        for k in range(dim):
            self.K[:, :, k] = array(self.K_sym[k][:][:], dtype=float)

        for k in range(ne):
            self.M_ed_TT[:, :, k] = \
                array(self.M_ed_TT_sym[k][:][:], dtype=float)
            self.M_ed_TP[:, :, k] = \
                array(self.M_ed_TP_sym[k][:][:], dtype=float)

        for k in range(n_ed_type):
            self.M_ed_PP[:, :, k] = \
                array(self.M_ed_PP_sym[k][:][:], dtype=float)

    def _mkelmSHAP(self, n_quad):
        '''Create the shape matrices for quadrature-based integration

        @param n_quad (\c int) The polynomial order which the quadrature should
                        integrate exactly
        '''
        ## Stores the coordinates of the quadrature points -- coordinate frame
        # of the master element. cube_pts.shape = (n_cube_pts, dim)
        self.cube_pts = None
        ## Stores the weights of the quadrature points.
        # cube_wghts.shape = (n_cube_pts, 1)
        self.cube_wghts = None

        self.cube_pts, self.cube_wghts = \
            get_pts_weights(n_quad, self.dim, self.element)

        ## Matrix that transforms a set of vertices to cubature points.
        # This is needed to create the real-space cubature edge-points which is
        # used for initialization, visualization, and any function-evaluations
        # (from source terms, for example).
        # Cube_pts_in_real_space = dot(cube_pts2xy, verts)
        # where verts is something like shape(verts)=(n_verts, dim*n_elm), or
        #verts = [x1, y1, z1, x2, y2, z2 ... ], where x1 is a column
        #vector with the x-coordinates of the vertices of the first element
        self.cube_pts2xy = self.basis.mkmapcoords(self.cube_pts)

        ## The number of cubature points/weigths
        self.n_cube_pts = len(self.cube_wghts)

        ## The shape matrix evaluates each basis at the cubature points.
        # \f$shap_{i\!j} = \theta_j(\mathbf x_i)\f$
        self.shap = self.basis.eval_at_pts(self.cube_pts)

        ## The shape derivative matrices evaluates the derivatives of each
        #  basis at the cubature points.
        # \f$shap_{der}[k]_{i\!j} = \partial_{x_k} \theta_j(\mathbf x_i) \f$
        self.shap_der = [self.basis.eval_at_pts(self.cube_pts, i) \
            for i in range(self.dim)]

    def _mkedSHAP(self, n_quad):
        '''Create the shape matrices for quadrature-based integration on element
        edges.

        @param n_quad (\c int) The polynomial order which the quadrature should
                        integrate exactly
        '''
        ## Stores the coordinates of the quadrature points on the edge of
        # the type basis.ed_type -- coordinate frame that of the edge
        # master element. len(cube_pts_ed) = n_ed_type,
        # cube_pts_ed.shape[0] = (n_ed_cube_pts[0], dim)
        self.cube_pts_ed = [None] * self.n_ed_type

        ## Stores the weights of the edge quadrature points.
        # len(cube_wghts_ed) = n_ed_type,
        # cube_wghts_ed.shape[0] = (n_ed_cube_pts[0], 1)
        self.cube_wghts_ed = [None] * self.n_ed_type

        ## List of the number of cubature points for the current edge.
        # len(n_ed_cube_pts) = n_ed_type
        self.n_cube_pts_ed = [None] * self.n_ed_type

        ## The edge shape matrix evaluates each basis at the cubature points on
        # the edges of the elements.
        # \f$shap_{ed}[k]_{i\!j}= \theta_j(\mathbf x_i)\f$ on edge number \f$k\f$.
        self.shap_ed = [None] * self.ne

        ## The \f$\phi\f$ edge shape matrix evaluates each basis that lives on
        # the edge at the cubature points on the edge.
        # \f$shap_{ed,P}[k]_{i\!j} = \phi_j(\mathbf x_i)\f$ on edge type \f$k\f$.
        self.shap_ed_P = [None] * self.n_ed_type

        ## Matrix that transforms a set of vertices to cubature points.
        # This is needed to create the real-space cubature edge-points which is
        # used for initialization, visualization, and any function-evaluations
        # (from source terms, for example).
        # Cube_pts_in_real_space_on_edge = dot(cube_pts2xy_ed[i], verts)
        # where verts is something like shape(verts)=(n_verts, dim*n_elm), or
        #verts = [x1, y1, z1, x2, y2, z2 ... ], where x1 is a column
        #vector with the x-coordinates of the vertices of the first edge
        self.cube_pts2xy_ed = [None] * self.n_ed_type

        i = 0
        for ed_elm in pyutil.unique(self.basis.ed_type):
            #Get the cubature points and weights, and record how many we have
            self.cube_pts_ed[i], self.cube_wghts_ed[i] = \
                get_pts_weights(n_quad, (self.dim - 1), ed_elm)
            self.n_cube_pts_ed[i] = len(self.cube_wghts_ed[i])

            #Create the shape matrix for the basis that lives on
            #that type of element
            self.shap_ed_P[i] = \
                self.basis_e[i].eval_at_pts(self.cube_pts_ed[i])

            ##Create the edge version of the mapping matrix -- needed for
            #visualization and initalization
            self.cube_pts2xy_ed[i] = mkb.mk_mapcoords(self.cube_pts_ed[i], \
                self.basis_e[i].vol_verts, ed_elm, self.dim - 1)
            i+=1

        for i in range(self.ne):
            if self.n_ed_type == 1:
                bnum = 0
            else:
                bnum = self.basis.ed_type[i]

            aij_e = mkb.elm2edge(self.basis, i, self.basis_e[bnum])
            dumbase = self.basis_e[bnum].copy(aij_e)
            self.shap_ed[i] = dumbase.eval_at_pts(self.cube_pts_ed[bnum])
