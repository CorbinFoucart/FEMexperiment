"""
Set of DOF Handler classes which handle the enumeration of DOF for different types of finite
element problems.
@author foucartc
"""

import numpy as np
from collections import OrderedDict
from src import util

class base_dofh(object):
    def count_elm_dof(self):
        elm_dof = 0
        for master_elm, n_elm in zip(self.master, self.n_elm_type):
            n_basis_fns = master_elm.nb
            elm_dof += n_basis_fns * n_elm
        return elm_dof

    def mk_nodal_points(self, master):
        """ creates the physical space nodal points for a single element type
        @param master  the Master2D master element object
        retval dgnodes_arr  the dgnodes array for a single element type
        NOTE: if verts empty, return value is None
        """
        elm_T = self.mesh.connectivity_by_elm_type(master.name)
        n_elm = elm_T.shape[0]
        if n_elm > 0:
            dgnodes_arr = np.zeros((master.nb, master.dim, n_elm))
            verts = self.mesh.vert[:,:2][elm_T]
            for elm in range(n_elm):
                dgnodes_arr[:,:,elm] = master.map_to_physical_space(verts[elm,::])
            return dgnodes_arr

    def mk_dgnodes(self):
        """ makes the list of dgnodes arrays for each element type in the mesh
        @retval dgnodes  list of dgnodes_arrays by element type
        """
        dgnodes = list()
        for master in self.master:
            dgnodes_arr_elm = self.mk_nodal_points(master)
            if dgnodes_arr_elm is not None:
                dgnodes.append(dgnodes_arr_elm)
        return dgnodes

    def mk_nodal_points_ed(self, master):
        """ creates the physical space nodal points for a single edge type
        @param master  the master edge object
        retval dgnodes_arr  the dgnodes array for a single element type
        NOTE: if verts empty, return value is None
        """
        edge_verts = self.mesh.edge_vertices()
        T = master.map_to_physical_edge()
        shape = edge_verts.shape
        pts = np.dot(T, edge_verts.swapaxes(0, 1).swapaxes(1, 2).reshape(
            shape[1], shape[0]*shape[2]))
        return pts.reshape(master.nb, shape[2], shape[0])

    def mk_dgnodes_ed(self):
        """ makes dgnodes by embedding the master edge to phys space edge verts
        @TODO: this will have to change for 3D
        """
        dgnodes_ed = list()
        for master_edge in self.master_ed:
            dgnodes_arr_ed = self.mk_nodal_points_ed(master_edge)
            dgnodes_ed.append(dgnodes_arr_ed)
        return dgnodes_ed

    @staticmethod
    def _dof2xy(arr):
            """ takes a [dof, dim, elms] shape -> [ndof, dim] """
            elmidx_first = np.rollaxis(arr, -1)
            stack_by_elm = elmidx_first.reshape(-1, elmidx_first.shape[-1])
            return stack_by_elm

    @staticmethod
    def _xy2dof(arr, dof_elm, nelm):
        """ takes a [ndof, dim] shaped arr -> [dof_elm, dim, nelm] """
        reshaped = arr.reshape(nelm, dof_elm, -1)
        return np.transpose(np.rollaxis(reshaped, 1), axes=(0, 2, 1))

    def mk_cubature_points(self): pass

class CG_dofh(base_dofh):
    def __init__(self, n, mesh, master_elms, master_eds):
        """
        @param n  the polynomial order of the basis
        @param mesh  the mesh data-structure
        @param master_elms  a list of master element instances corresponding to
            mesh element types
        @param master_eds  a list of master element instances corresponding to
            mesh edge types
        """
        self.p = n
        self.mesh = mesh
        self.n_elm, self.n_ed, self.dim = mesh.n_elm, mesh.n_ed, mesh.dim
        self.n_elm_type  = mesh.n_elm_type
        self.master, self.master_ed = master_elms, master_eds
        self.n_elm_dof = self.count_elm_dof()
        self.dgnodes = self.mk_dgnodes()
        self.dgnodes_ed = self.mk_dgnodes_ed()
        self.bd_nodes, self.int_nodes, self.cgnodes, self.lg = self.assign_nodes()
        self.tot_dof = self.cgnodes.shape[0]

    def field2dgn(self, field):
        """ converts a field organized in cgnodes form to an array in dgnodes form
        WARNING: right now, as implemented, only works for scalar-valued data
        """
        dgfield = []
        for elmType, lg in enumerate(self.lg):
            dof, dim, nelm = self.dgnodes[elmType].shape
            arr = field[self.lg[elmType].T]
            dgfield.append(arr)
        return dgfield

    def update_cgnodes(self):
        """ updates the cgnodes, maybe after a dgnodes warping """
        self.bd_nodes, self.int_nodes, self.cgnodes, self.lg = self.assign_nodes()

    def assign_nodes(self):
        """ this is when node numbers are assigned """
        all_nodes = self.mk_node_set()
        bd_nodes  = self.mk_bd_set()
        interior_nodes = all_nodes - bd_nodes
        n_int, n_bd = len(interior_nodes), len(bd_nodes)
        nodes_dict = OrderedDict([(pt, idx) for idx, pt in enumerate(interior_nodes)] +
                                 [(pt, idx + n_int) for idx, pt in
                                 enumerate(bd_nodes)])

        bd_nodes = np.arange(n_int, len(nodes_dict))
        int_nodes = np.arange(n_int)
        cgnodes = np.asarray(nodes_dict.keys())
        lg = self.mk_lg(nodes_dict)
        return bd_nodes, int_nodes, cgnodes, lg

    def mk_bd_set(self):
        bd_nodes = set()
        for edType, ids in enumerate(self.mesh.ids_exterior_ed_by_type):
            bd_dgn = self.dgnodes_ed[edType][:,:,ids]
            bd_xy = np.around(self._dof2xy(bd_dgn), decimals=10)
            for pt in bd_xy:
                bd_nodes.add(tuple(pt))
        return bd_nodes

    def mk_node_set(self):
        """
        compiles a set containing all distinct nodal points as tuples dg -> cg
        nodes
        """
        node_set = set()
        for etype, dgnodes in enumerate(self.dgnodes):
            dof_elm, _, nelm = dgnodes.shape
            xy = np.around(self._dof2xy(dgnodes), decimals=10)
            node_set.update(set(tuple(pt) for pt in xy))
        return node_set

    def mk_lg(self, nodes_dict):
        """ makes the local to global array by element type
        @param nodes dict  ordered dict with [x1, ..., xn] tuples as keys, glob
            idx is value
        """
        lg = list()
        for etype, dgnodes in enumerate(self.dgnodes):
            dof_elm, _, nelm = dgnodes.shape
            xy = np.around(self._dof2xy(dgnodes), decimals=10)
            lgxy = np.asarray([nodes_dict[tuple(arr)] for arr in xy])
            lg_et = np.squeeze(self._xy2dof(lgxy, dof_elm, nelm), axis=1)
            lg.append(lg_et.T)
        return lg

class DG_dofh(base_dofh): pass

class HDG_dofh(base_dofh):
    def __init__(self, n, mesh, master_elms, master_eds):
        """
        @param n  the polynomial order of the basis
        @param mesh  the mesh data-structure
        @param master_elms  a list of master element instances corresponding to
            mesh element types @param master_eds  a list of master element instances
            corresponding to mesh edge types
        """
        self.p = n
        self.mesh = mesh
        self.n_elm, self.n_ed, self.dim = mesh.n_elm, mesh.n_ed, mesh.dim
        self.n_elm_type, self.n_ed_type = mesh.n_elm_type, mesh.n_ed_type
        self.master, self.master_ed = master_elms, master_eds
        self.n_elm_dof, self.n_dof_ed = self.count_elm_dof(), self.count_ed_dof()
        self.dgnodes = self.mk_dgnodes()
        self.dgnodes_ed = self.mk_dgnodes_ed()

    def count_ed_dof(self):
        """ returns a count of all the edge degrees of freedom for the problem
        """
        ed_dof = 0
        for master_ed, n_ed in zip(self.master_ed, self.n_elm_type):
            n_ed_basis_fns = master_ed.nb
            ed_dof += n_ed_basis_fns * n_ed
        return ed_dof

    def index_map_ed2elm(): pass


class Sol_Dofh(object):
    """This class collects the data-structures required for a solution
    __init__(self, n, mesh, n_trace=1, n_vec=0, n_p=0, init_type='all',
             n_quad=None):
    """

    def __init__(self, mesh, masters, master_edges):
        """ Class constructor
        @param n The order of the basis
        @param mesh The mesh data-structure
        @param n_quad (\c int) The order of the quadrature for QB integration.
                      Usually taken as n_quad = 2 * n (DEFAULT) which is exact
                      for linear operators.
        """
        ##The order of the basis
        self.n = masters[0].p

        ##A local copy of the mesh data-structure.
        self.mesh = mesh

        ##Number of elements
        self.n_elm = mesh.n_elm

        ##Number of elements separated into element type
        self.n_elm_type = mesh.n_elm_type

        ##Number of edges
        self.n_ed = mesh.n_ed

        ##Number of edges separated into type
        self.n_ed_type = mesh.n_ed_type

        #Create the master structures
        ##A list of the master data-structures needed. There is one master
        # structure for each element type in the mesh.
        self.master = masters

        ##Create the master element for the edge-space -- this results in some
        #overlap of bases (maybe we fix this later?
        #print n, mesh.dim-1, init_type, mesh.u_ed_type
        #CM: This produces a warning that Master_nodal works for 2D and 3D
        # in the case mesh.dim == 2
        self.master_ed = master_edges

        ##Number of bases per element - this will be the different for
        #different element types
        self.nb = [M.nb for M in self.master]

        ##Number of bases per edge - this will be the different for
        #different element types.
        self.nb_ed = [M.nb for M in self.master_ed]

        ##Dimension of the problem.
        self.dim = mesh.dim

        ###Total number of degrees of freedom for volume elements.
        #self.n_dofs_elm = 0
        #for n_base, n_elm in zip(self.nb, self.n_elm_type):
        #    self.n_dofs_elm += n_base * n_elm * (n_trace + n_p + self.dim * n_vec)

        ###Total number of degrees of freedom for edge elements.
        #self.n_dofs_ed = 0
        #for n_base, n_elm in zip(self.nb_ed, self.n_ed_type):
        #    self.n_dofs_ed += n_base * n_elm \
        #        * (n_trace + n_p + self.dim * n_vec)

        ##Number of degrees of freedom for a single field of volume elements.
        self.n_dofs_elm_1 = 0
        for n_base, n_elm in zip(self.nb, self.n_elm_type):
            self.n_dofs_elm_1 += n_base * n_elm

        ##Number of degrees of freedom for a single field of edge elements
        self.n_dofs_ed_1 = 0
        for n_base, n_elm in zip(self.nb_ed, self.n_ed_type):
            self.n_dofs_ed_1 += n_base * n_elm

        ##The unique edge types in the mesh
        self.u_ed_type = self.mesh.u_ed_type

        ##The unique element types in the mesh
        self. u_elm_type = self.mesh.u_elm_type

        ##Make oft-used FEM matrices
        ###Element Mass matrix inverse:
        #self.Minv = [np.linalg.inv(m.M) for m in self.master]
        ##Edge Mass matrix inverse:
        #Minv_ed = [np.linalg.inv(m.M) for m in self.master_ed]
        ###Element Derivative matrices (weak form)
        #self.Dw = [[np.dot(MI, D) for D in M.K] \
        #    for M, MI in zip(self.master, self.Minv)]
        ###Element Derivative matrices (strong form)
        #self.Ds = [[np.dot(MI, D.T) for D in M.K] \
        #    for M, MI in zip(self.master, self.Minv)]
        ###Edge Derivative matrices (strong form)
        #self.Ds_ed = [[np.dot(MI, D.T) for D in M.K] \
        #    for M, MI in zip(self.master_ed, Minv_ed)]

        ##Total number of edge degrees of freedom in an element
        #This includes the duplicates at 'vertices' (i.e. vertices in 2D)
        self.nb_ed_in_elm = [0] * len(self.u_elm_type)
        for i in range(len(self.u_elm_type)):
            for eds in self.master[i].basis.ed_type:
                j = (self.u_ed_type == eds).nonzero()[0][0]
                self.nb_ed_in_elm[i] += self.nb_ed[j]

        ##Edge to element 'lifting' matrices. That is,
        #lift \f$= \text{map}(<u, \theta>)\f$
        #    \f$ = \mathbf L \sum_{\partial K} \mathbf M_e u \f$
        self.lift = [m.L for m in self.master]
        self.lift = [np.dot(MI, L) for MI, L in zip(self.Minv, self.lift)]
        #Finally we need to multiply through by the block-diagonal edge
        #mass-matrices
        mass_ed = [np.zeros((nb, nb)) for nb in self.nb_ed_in_elm]
        for i in range(len(self.u_elm_type)):
            ns = 0
            for eds in self.master[i].basis.ed_type:
                j = (self.u_ed_type == eds).nonzero()[0][0]
                ne = ns + self.nb_ed[j]
                mass_ed[i][ns:ne, ns:ne] = self.master_ed[j].M
                ns = ne
        ##Edge mass-matrices arranged in the ed2elm space
        self.mass_ed = mass_ed
        self.lift = [np.dot(L, MED) for L, MED in zip(self.lift, mass_ed)]

        #Make the coordinates stored at nodal points -- for iso-parametric
        #mapping of domain
        self.dgnodes = None
        dgnodes = [dgn.swapaxes(0, 1) \
            for dgn in self.mk_cube_nodes(nodepts=True)]
        self.dgnodes_ed = None
        dgnodes_ed = [dgn.swapaxes(0, 1) \
            for dgn in self.mk_cube_nodes_ed(nodepts=True)]

        ##The physical spatial location of the element nodes in the
        #triangulation. dgnodes[i].shape = (nb[i], dim, n_elm_type[i])
        self.dgnodes = dgnodes
        ##The physical spatial location of the edge nodes in the
        #triangulation. dgnodes_ed[i].shape = (nb_ed[i], dim, n_ed_type[i])
        self.dgnodes_ed = dgnodes_ed

        #Make the jacobians for the mesh
        ##The element jacobian calculated at every nodal point.
        self.jac = None

        ##The element derivative factors at every nodal point. That is,
        #factors[k][i][j] is the factors for element types k, where the factor
        #is \f$\frac{\partial r_i}{\partial x_j}\f$ -- basically the jacobian
        #matrix. Also, factors[k][i][j].shape = [nb[i], n_elm_type[i]]
        self.jac_factor = None
        ##The edge jacobian calculated at every edge nodal point.
        self.jac_ed = None
        ##The edge normal calculated at every nodal points.
        self.nrml = None

        ##The tvtk object that contains information needed for plotting
        self.ug_src = None

        #Next we have to build a rather annoying structure -- the elements have
        #global numbers -- however the data is stored/organized according to the
        #element type. So within the element type, the element will have a
        #different number/location. The next structure figures out what that
        #element number is. The same goes for the edge types
        ## The "global element number" to "element number within a type"
        # conversion array. For example, the data for global element number i
        # is stored in field[elm_type[i]][:, :, glob2type_elm[i]].
        self.glob2type_elm = np.zeros(self.n_elm, dtype=int)
        sumtype = [0] * len(self.u_elm_type)
        for i in xrange(self.n_elm):
            elm_type = self.mesh.elm_type[i]
            self.glob2type_elm[i] = sumtype[elm_type]
            sumtype[elm_type] += 1

        ## The "global edge number" to "edge number within a type"
        # conversion array. For example, the data for global edge number i
        # is stored in field_ed[ed_type[i]][:, :, glob2type_ed[i]].
        self.glob2type_ed = np.zeros(self.n_ed, dtype=int)
        sumtype = [0] * len(self.u_ed_type)
        for i in xrange(self.n_ed):
            ed_type = self.mesh.ed_type[i]
            self.glob2type_ed[i] = sumtype[ed_type]
            sumtype[ed_type] += 1

        ##Mapping array needed to map element degrees of freedom to edge degrees
        #of freedom. elm2ed_ids[ed_type][left_right].shape = [nb_ed, n_ed_type]
        #where left_right is [0,1] to represent the left and right elements,
        #respectively. (Left and right are defined on the location of the
        #element in the mesh.ed2ed array). To select the edge dofs from elements
        #of a certain type is a little tricky... @see _get_elm2ed_array_1
        self.elm2ed_ids = None

        ##Mapping array needed to map edge degrees of freedom to element degrees
        # of freedom. ed2elm_ids[elm_type].shape = [totalEdgeDofPerElm, n_elm_type],
        # where totalEdgeDofPerElm is the total number of edge degrees of freedom for an
        #element (with overlaps included in the total, e.g. the vertex of a 2D
        #triangle is counted twice). Conveniently, this array can also be used
        #in reverse, to find out which edge element i of type j's data goes, you
        #can use the lifting matrix (master.L.T) along with this array.
        #@see sol.hdg.Diffusion.mk_A for an example.
        self.ed2elm_ids = None

        ##Mapping array needed to negate the edge degrees of freedom to element
        #degrees of freeom on the RIGHT of an edge
        self.ed2elm_ids_right = None

        self.elm2ed_ids, self.ed2elm_ids, self.ed2elm_ids_right \
            = self._get_elm2ed2elm_ids()

        #Boundary condition information
        tot_bc_ids = -min(self.mesh.ed2ed[self.mesh.ids_exterior_ed, 2])
        ##Total number of different bc_ids
        self.n_bc_id = tot_bc_ids

        ## Array that assigns a boundary condition type to each unique boundary
        #id from the mesh for tracers.
        #bcid2type_trace.shape = (max_bc_id + 1, n_trace),
        #where max_bc_id + 1 = -min(mesh.ed2ed[2, :])
        #By default, the boundary type is 0, which should be
        #Dirichlet for most solvers. The boundary conditions types are solver-
        #dependent.
        self.bcid2type_trace = np.zeros((tot_bc_ids, n_trace), dtype=int)

        ## Array that assigns a boundary condition type to each unique boundary
        #id from the mesh for vectors.
        #bcid2type_vecs.shape = (max_bc_id + 1, dim, n_vec),
        #where max_bc_id + 1 = -min(mesh.ed2ed[2, :])
        #By default, the boundary type is 0, which should be
        #Dirichlet for most solvers. The boundary conditions types are solver-
        #dependent.
        self.bcid2type_vecs = np.zeros((tot_bc_ids, self.dim, n_vec), dtype=int)

        ## Array that assigns a boundary condition type to each unique boundary
        #id from the mesh for patricular fields (i.e. Pressure).
        #bcid2type_p.shape = (max_bc_id + 1, n_p),
        #where max_bc_id + 1 = -min(mesh.ed2ed[2, :])
        #By default, the boundary type is 0, which should be
        #Dirichlet for most solvers. The boundary conditions types are solver-
        #dependent.
        self.bcid2type_p = np.zeros((tot_bc_ids, n_p), dtype=int)

        ## Just a copy of the boundary id for different element types
        # CF: I think he means edge types here
        self.bcid = -self.mesh.ed2ed[:, 2] - 1
        self.bcid = [self.bcid[(self.mesh.ids_exterior_ed) & (self.mesh.ed_type == i)] \
            for i in range(len(self.mesh.n_ed_type))]

        ##A list of boolian arrays used to select the interior edges only
        self.ids_interior_ed = \
            [self.mesh.ids_interior_ed[self.mesh.ed_type == i] \
            for i in range(len(self.mesh.n_ed_type))]

        ##A list of boolian arrays used to select the exterior edges only
        self.ids_exterior_ed = \
            [self.mesh.ids_exterior_ed[self.mesh.ed_type == i] \
            for i in range(len(self.mesh.n_ed_type))]

        ##The number of boundary edges for a particular edge type
        self.n_ed_bc_type = [idsum.sum() for idsum in self.ids_exterior_ed]

        ##Index mapping array from ed_type edge id number to ed_bc_type id
        #number. Basically, in the solver we will refer to, for e.g. the
        #data field_ed[i][:, :, j], where j refers to a boundary edge, numbered
        #according to the element-type local id number. The boundary condition
        #data is stored in an array smaller that field_ed, that is, field_ed_bc
        #contains ONLY the boundary condition information, so calling
        #field_ed_bc[i][:, :, j] will exceed the array bounds. Instead we call
        #field_ed_bc[i][:, :, in2ex_bcid[j]].
        #TODO: Determine if this array is actually still needed
        #   (Indexing has been improved since the below was implemented)
        self.in2ex_bcid = [ex.cumsum()-1 for ex in self.ids_exterior_ed]

        ##Tracer field boundary conditions value edge storage array.
        #Numpy array of floats storing the different vector components on the
        # boundary edges only. Used to set the value of boundary conditions
        #vecs_ed_bc[i].shape = (nb_ed, dim, n_vec, n_ed_bc_type)
        self.trace_ed_bc = None
        if n_trace > 0:
            self.trace_ed_bc = [np.zeros((nb, n_trace, n_ed)) \
                for nb, n_ed in zip(self.nb_ed, self.n_ed_bc_type)]

        ##Vector field boundary conditions value edge storage array.
        #Numpy array of floats storing the different vector components on the
        # boundary edges only. Used to set the value of boundary conditions
        #vecs_ed_bc[i].shape = (nb_ed, dim, n_vec, n_ed_bc_type)
        self.vecs_ed_bc = None
        if n_vec > 0:
            self.vecs_ed_bc = [np.zeros((nb, self.dim, n_vec, n_ed)) \
                for nb, n_ed in zip(self.nb_ed, self.n_ed_bc_type)]

        ##Particular field boundary conditions value edge storage array.
        #Numpy array of floats storing the different vector components on the
        # boundary edges only. Used to set the value of boundary conditions
        #p_ed_bc[i].shape = (nb_ed, n_p, n_ed_bc_type)
        #Normally this field is used to store the pressure.
        self.p_ed_bc = None
        if n_p > 0:
            self.p_ed_bc = [np.zeros((nb, n_p, n_ed)) \
                for nb, n_ed in zip(self.nb_ed, self.n_ed_bc_type)]

        #CM DEBUGGING
#        print "From Sol_nodal.__init__, args to update_jacs:"
#        print dgnodes[0][:,:,0]

        #Calculate the Jacobians:
        self.update_jacs(dgnodes, dgnodes_ed)

        #CM DEBUGGING
        #print "From Sol_nodal.__init__, result of update_jacs:"
        #print dgnodes[0][:,:,0]

        if init_type.lower() in ['all', 'both', 'qb', 'shap']:
            #print "Calculating Jacobians at cubature points for: Volume...",
            ##Jacobians evaluated at the cubature points
            self.cube_jac = None
            #print" edges... ",
            ##Jacobians evaluated at the edge cubature points
            self.cube_jac_ed = None
            self.update_cube_jacs()
            #print "Done."

        #Time and time-stepping information
        ##The time at which the fields are currently given. Only used for
        #applications with time.
        self.time = None
        ##The integration timestep size. Only used for applications with time.
        self.dt = None
        ##Function to set the timestep size. Only used for applications with
        #variable timesteps during integration.
        self.setdt = None

        ##Set a maximum timestep size -- this useful for when the velocity goes
        #to zero everywhere
        self.maxdt = 1.

        # CF set ppeBool to be false by default
        self.ppeBool = False


    ###########################################################################
    def update_jacs(self, dgnodes=None, dgnodes_ed=None, discmesh=False):
        """Updates the jacobians, jacobian factors, edge jacobians, and normals

        @param dgnodes (\c float) List of spatial points for the spatial
            location of each node for each type of element. For example,
            dgnodes[i].shape = (master[i].nb, dim, n_elm_type[i])
            If None, self.dgnodes will be used.
        @param dgnodes_ed (\c float) As above but for the edge space. For
            example,
            dgnodes_ed[i].shape = (master_ed[i].nb, dim-1, n_ed_type[i])
            If None, self.dgnodes will be used.
        @param discmesh (\c bool) Flag to indicate whether or not mesh is
            discontinuous.

        @note Function returns nothing, but modifies:
            sol.jac, sol.jac_factor, sol.jac_ed, sol.nrml
        """
        if dgnodes == None:
            dgnodes = self.dgnodes
        if dgnodes_ed == None:
            dgnodes_ed = self.dgnodes_ed

        self.jac_ed, self.nrml = \
            mkj.jacobian_ed(self.Ds_ed, self.dim, dgnodes_ed)
        self.jac, self.jac_factor = \
            mkj.jacobian(self, discmesh)

    def update_cube_jacs(self):
        """Updates the jacobians and edge jacobians at the cubature points.
        This is used for initialization only.

        @note Function returns nothing, but modifies:
            sol.cube_jac, sol.cube_jac_ed

        @note Jacobians are 'calculated' by interpolating the jacobians stored
            at nodal points.
        """
        #Jacobians evaluated at the cubature points
        self.cube_jac = self.mk_cube_jacobians()
        #Jacobians evaluated at the edge cubature points
        self.cube_jac_ed = self.mk_cube_jacobians_ed()

    ###########################################################################

    def trace_init(self, funcfld, fldcoords=None):
        """Initialize the tracer variables.

        @param funcfld Either a list of lambda functions that takes "points" as
                       and input (see below), or a list of numpy arrays of the
                       function evaluated at the points fldcoords.
                       len(funcfld) < = n_trace

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of value, fldcoords is an
                         array of floats, with
                         fldcoords.shape = (n_points, dim), if irreg = True
                         fldcoords = ogrid[xrange, yrange, zrange]

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note The intialization evaluate the input field/function at cubature
              points, and then projects the function onto the basis functions.
              \f$t_i = \mathbf M^{-1} \int_\Omega f\theta d\Omega\f$
        """
        self.trace = self._init(self.trace, funcfld, fldcoords)
        self.trace_ed = self._init_ed(self.trace_ed, funcfld, fldcoords)

    ###########################################################################

    def vecs_init(self, funcfld, fldcoords=None):
        """Initialize the vector variables.

        @param funcfld Either a list of lists of lambda functions that takes
                       "points" as an input (see below), or a list of lists of
                       numpy arrays of the function evaluated at the points
                       fldcoords. len(funcfld) < = n_vec

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of value, fldcoords is an
                         array of floats, with
                         fldcoords.shape = (n_points, dim), if irreg = True
                         fldcoords = ogrid[xrange, yrange, zrange]

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note The intialization evaluate the input field/function at cubature
              points, and then projects the function onto the basis functions.
              \f$t_i = \mathbf M^{-1} \int_\Omega f\theta d\Omega\f$

        @code
        >>> #The form of funcfld is a little confusing, so here is an example
        >>> x1_init = lambda x: x[:, 0]
        >>> y1_init = lambda x: x[:, 1]
        >>> z1_init = lambda x: x[:, 2]
        >>> x2_init = lambda x: x[:, 0]**2
        >>> y2_init = lambda x: x[:, 1]**2
        >>> z2_init = lambda x: x[:, 2]**2
        >>> funcfld =[[x1_init, y1_init, z1_init], [x1_init, y1_init, z1_init]]
        >>> Sol.vec_init(funcfld)
        @endcode
        """
        for i in xrange(len(funcfld)):
            tmp = [v[:, :, i, :] for v in self.vecs]
            tmp2 = self._init(tmp, funcfld[i], fldcoords)
            for j in range(len(self.vecs)):
                self.vecs[j][:, :, i, :] = tmp2[j]
            tmp = [v[:, :, i, :] for v in self.vecs_ed]
            tmp2 = self._init_ed(tmp, funcfld[i], fldcoords)
            for j in range(len(self.vecs_ed)):
                self.vecs_ed[j][:, :, i, :] = tmp2[j]

    ###########################################################################

    def p_init(self, funcfld, fldcoords=None):
        """Initialize the particular variables.

        @param funcfld Either a list of lambda functions that takes "points" as
                       and input (see below), or a list of numpy arrays of the
                       function evaluated at the points fldcoords.
                       len(funcfld) < = n_p

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of value, fldcoords is an
                         array of floats, with
                         fldcoords.shape = (n_points, dim), if irreg = True
                         fldcoords = ogrid[xrange, yrange, zrange]

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note The intialization evaluate the input field/function at cubature
              points, and then projects the function onto the basis functions.
              \f$t_i = \mathbf M^{-1} \int_\Omega f\theta d\Omega\f$
        """
        self.p = self._init(self.p, funcfld, fldcoords)
        self.p_ed = self._init_ed(self.p_ed, funcfld, fldcoords)

    ###########################################################################

    def _init(self, field, funcfld, fldcoords=None, irreg=False, init_type='QB'):
        """Initialize a scalar variables on the element space.

        @param field The field that will be initialized

        @param funcfld Either a lambda function that takes inputs points (see
                       below), or
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of values, fldcoords is an
                         array of floats, with
                         if irreg = True: fldcoords.shape = (n_points, dim),
                         if irreg = False: fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @param init_type (\c string): 'QF' or 'QB' for quadrature free
            (interpolating) or quadrature-based (projection) initialization.

        @retval field The initialized field

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note The intialization evaluate the input field/function at cubature
              points, and then projects the function onto the basis functions.
              \f$t_i = \mathbf M^{-1} \int_\Omega f\theta d\Omega\f$
        """
        if type(funcfld) is not list:
            funcfld = [funcfld]

        if not callable(funcfld[0]):
            #Some minor error checking (sensible error message)
            if fldcoords == None:
                print("ERROR: fldcoords needs to be set if funcfld is a list", \
                    "of numpy arrays instead of lambda functions.")
            if irreg:
                pass
                #funcfld = [interpnd.LinearNDInterpolator(fldcoords, vals) \
                #    for vals in funcfld]
            else:
                #For regularly-spaced data, it is faster to use
                #ndimage.map_coordinates, implemented in pyutil
                funcfld = [pyutil.Bi_linear_interp(fldcoords, vals, self.dim) \
                    for vals in funcfld]

        if init_type == "QF":
            nfld = len(funcfld)
            field = [np.zeros((dgn.shape[0], nfld, dgn.shape[2])) \
                for dgn in self.dgnodes]
            for i in range(len(self.dgnodes)):
                for j in range(nfld):
                    field[i][:, j, :] = funcfld[j](self.dgnodes[i])

        elif init_type == "QB":
            #Calculate where the cubature points are in real space
            cube_nodes = self.mk_cube_nodes()

            #For each element type in the mesh
            for j in range(len(self.u_elm_type)):
                #Get the shape of this matrix
                shape = cube_nodes[j].shape

                #Need to reformat this to match the inputs to the function
                #i.e. to match 'points'
                cnds = cube_nodes[j].reshape(shape[0], shape[1] * shape[2]).T

                #Shorter names
                wghts = self.master[j].cube_wghts
                shapw = np.dot(self.master[j].shap.T, np.diag(wghts))

                #For each tracer field (for each initialization function)
                for i in xrange(len(funcfld)):
                    field[j][:, i, :] = np.dot(shapw, \
                        funcfld[i](cnds).reshape(shape[1], shape[2]) \
                            * self.cube_jac[j])

            # Now multiply with the inverse mass-matrix
            # We have to keep count of how many elements of each type has been used
            # in order to select the correct jacobian

            # THERE SHOULD BE A NICER WAY TO DO THE STUFF BELOW GIVEN THE NEW DATA-
            # STRUCTURE!
            for j in range(len(self.u_elm_type)):
                shap = self.master[j].shap
                wghts = self.master[j].cube_wghts
                shapw = np.dot(np.diag(wghts), shap)
                for i in xrange(self.n_elm_type[j]): #How to vectorize this?
                    jacs = self.cube_jac[j][:, i].flatten()
                    M = np.dot(np.dot(shap.T, np.diag(jacs)), shapw)
                    field[j][:, :, i] = np.linalg.solve(M, field[j][:, :, i])

        return field

    ###########################################################################

    def _init_ed(self, field, funcfld, fldcoords=None, irreg=False,\
        init_type='QB'):
        """Initialize a scalar variables on the edge-space.

        @param field The field that will be initialized

        @param funcfld Either a lambda function that takes inputs points (see
                       below), or
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of value, fldcoords is an
                         array of floats, with
                         fldcoords.shape = (n_points, dim), if irreg = True
                         fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @param init_type (\c string): 'QF' or 'QB' for quadrature free
            (interpolating) or quadrature-based (projection) initialization.

        @retval field The initialized field

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note The intialization evaluates the input field/function at cubature
          points, and then projects the function onto the basis functions.
          \f$t_i = \mathbf M^{-1} \int_{\partial \Omega} f\phi d\partial\Omega\f$
        """
        if type(funcfld) is not list:
            funcfld = [funcfld]

        if not callable(funcfld[0]):
            #Some minor error checking (sensible error message)
            if fldcoords == None:
                print("ERROR: fldcoords needs to be set if funcfld is a list", \
                    "of numpy arrays instead of lambda functions.")
            if irreg:
                pass
                #funcfld = [interpnd.LinearNDInterpolator(fldcoords, vals) \
                #    for vals in funcfld]
            else:
                #For regularly-spaced data, it is faster to use
                #ndimage.map_coordinates, implemented in pyutil
                funcfld = [pyutil.Bi_linear_interp(fldcoords, vals, self.dim) \
                    for vals in funcfld]


        if init_type == 'QF':
            nfld = len(funcfld)
            field = [np.zeros((dgn.shape[0], nfld, dgn.shape[2])) \
                for dgn in self.dgnodes_ed]
            for i in range(len(self.dgnodes_ed)):
                for j in range(nfld):
                    field[i][:, j, :] = funcfld[j](self.dgnodes_ed[i])

        elif init_type == 'QB':
            #Calculate where the cubature points are in real space
            cube_nodes_ed = self.mk_cube_nodes_ed()

            #For each edge-element type in the mesh
            for j in range(len(self.n_ed_type)):
                #Get the shape of this matrix
                shape = cube_nodes_ed[j].shape

                #Need to reformat this to match the inputs to the function
                #i.e. to match 'points'
                cnds = cube_nodes_ed[j].reshape(shape[0], shape[1] * shape[2]).T
                #print "In Sol_nodal._init_ed():"
                #print "cnds.shape = ", cnds.shape

                #Shorter names
                shap = self.master_ed[j].shap
                wghts = self.master_ed[j].cube_wghts
                shapw = np.dot(shap.T, np.diag(wghts))

                #For each tracer field (for each initialization function)
                for i in xrange(len(funcfld)):
                        field[j][:, i, :] = np.dot(shapw, \
                            funcfld[i](cnds).reshape(shape[1], shape[2]) \
                            * self.cube_jac_ed[j])

            #NOw multiply with the inverse mass-matrix
            #We have to count the number of each edge type used already in order
            #to select the correct jacobian
            for j in xrange(len(self.n_ed_type)):
                shap = self.master_ed[j].shap
                wghts = self.master_ed[j].cube_wghts
                for i in xrange(self.n_ed_type[j]):
                    jacs = self.cube_jac_ed[j][:, i].flatten()
                    M = np.dot(np.dot(shap.T, np.diag(wghts * jacs)), shap)
                    field[j][:, :, i] = np.linalg.solve(M, field[j][:, :, i])

        return field

    ###########################################################################

    def mk_cube_nodes(self, elmids=None, nodepts=False):
        """ This creates the cubature nodes in real-space so that a function
        defined in real space can be evaluated at the cubature points

        @param elmids (\c int) List or single integer of element numbers for
                        which the cubature node location will be calculated

        @param nodepts (\c bool) Make the real space points for points defining
                        the nodal basis instead of the cubature points.

        @retval cube_nodes (\c float) List of numpy array which give the
                           cubature nodes for each type of element

        @see sol.plot.mk_nodes_vol
        """
        #We can just use the more general function in sol.plot, and supply the
        #cubature matrices
        if nodepts:
            vol_pts = [M.basis.nodal_pts for M in self.master]
        else:
            vol_pts = [M.cube_pts for M in self.master]
        if self.dgnodes == None:
            cube_nodes = mk_nodes_vol(self, vol_pts, elmids)
        else:
            cube_nodes = [dgn.swapaxes(0, 1) \
                for dgn in mk_nodes_vol(self, vol_pts, elmids, \
                dgnodes=self.dgnodes)]
        return cube_nodes

    ###########################################################################
    #Following function deprecated by adding master_ed in the sol structure
    #def _get_unique_ed_elm_and_coords(self):
    #    """Just a small function to calculate the unique edge elements, and
    #    how to get at their bases, cubature points, etc. The edge info for
    #    each element is contained in the master structure for that element, so
    #    there tends to be some overlap. This just finds one route to get to
    #    the necessary information. The current system works very well for
    #    element operation, but when doing operations on the edge, it's a little
    #    tricky"""
    #    n_ed_type = 0
    #    elm = []
    #    elm_coords = []
    #    for i in range(len(self.master)):
    #        for j in range(len(self.master[i].basis_e)):
    #            elm.append(self.master[i].basis_e[j].element)
    #            elm_coords.append([i,j])
    #
    #    Only interested in the unique edge-element types. Save the index to
    #    apply the same operation to the elm_coords
    #    elm, I = np.unique(elm, return_index=True)
    #    elm_coords = np.array(elm_coords)[I]
    #    n_ed_type = len(elm)
    #    return elm, elm_coords, n_ed_type

    ###########################################################################

    def mk_cube_nodes_ed(self, edids=None, nodepts=False):
        """ This creates the cubature nodes in real-space so that a function
        defined in real space can be evaluated at the cubature points

        @param edids (\c int) List or single integer of edge numbers  for
                        which the cubature node locations will be calculated

        @param nodepts (\c bool) Make the real space points for points defining
                        the nodal basis instead of the cubature points.

        @retval cube_nodes_ed (\c float) List of numpy array which give the
                           cubature nodes for each type of element

        @see sol.plot.mk_nodes_ed
        """

        if nodepts:
            ed_pts = [M.basis.nodal_pts for M in self.master_ed]
        else:
            ed_pts = [self.master_ed[i].cube_pts \
                for i in range(len(self.n_ed_type))]
        if self.dgnodes_ed == None:
            cube_nodes_ed = mk_nodes_ed(self, ed_pts, edids)
        else:
            cube_nodes_ed = [dgn.swapaxes(0, 1) \
                for dgn in mk_nodes_ed(self, ed_pts, edids, \
                dgnodes_ed=self.dgnodes_ed)]

        return cube_nodes_ed #This checks out

    ###########################################################################

    def mk_cube_jacobians(self, elmids=None):
        """ This evaluates the jacobians at the cubature nodes in real-space so
        that a function can be integrated on the master element

        @param elmids (\c int) List or single integer of element numbers  for
                        which the cubature node locations will be calculated

        @retval cube_jac (\c float) List of numpy array which give the
                           jacobian values for each type of element
        """

        cube_jac = [ np.dot(M.shap, jac) \
            for (M, jac)  in zip(self.master, self.jac)]

        return cube_jac

    ###########################################################################

    def mk_cube_jacobians_ed(self, edids=None):
        """ This evaluates the jacobians at the cubature nodes in real-space so
        that a function can be integrated on the master element

        @param edids (\c int) List or single integer of edge numbers  for
                        which the cubature node locations will be calculated

        @retval cube_jac_ed (\c float) List of numpy array which give the
                           jacobian values for each type of element
        """
        cube_jac_ed = [ np.dot(M.shap, jac) \
            for (M, jac)  in zip(self.master_ed, self.jac_ed)]

        return cube_jac_ed

    ###########################################################################

    def _get_orient_num(self, orient, one_num=False, reverse=False, ed_type=0):
        """Method that defines the number for a particular orientation of an
        edge.

        @param orient (\c int) List containing at least the first 2 points of
                               the node ordering in the edge

        @param one_num (\c bool) Flag to indicate that a single number should
                               be given instead

        @param reverse (\c bool) Flag to indicate that the orient numbers
                                are known and the ordering is requested.

        @param ed_type (\c int) The edge element type -- needs to be specified
                               for the reverse lookup.
                               0 -- Triangle
                               1 -- Quadrilateral

        @retval orient_num (\c int) The number (or coordinates) of where to
                               find the re-ordering for that orientation

        @verbatim
        Dimension = 1: Edge is defined by 2 points, [0, 1]
            Other edge: [0, 1], orient number [0, 0], 0 if one_num
            Other edge: [1, 0], orient number [0, 1], 1 if one_num
        Dimension = 2:
         Triangle: Edge is defined by 3 points, [0, 1, 2]
            Other edge [0, 1, ...], orient number [0, 0], 0 if one_num
            Other edge [1, 2, ...], orient number [1, 0], 1 if one_num
            Other edge [2, 0, ...], orient number [2, 0], 2 if one_num
            Other edge [0, 2, ...], orient number [1, 0], 4 if one_num
            Other edge [1, 0, ...], orient number [1, 1], 5 if one_num
            Other edge [2, 1, ...], orient number [1, 2], 6 if one_num
         Rectangle: Edge is defined by 4 points, [0, 1, 2, 4]
            Other edge [0, 1, ...], orient number [0, 0], 0 if one_num
            Other edge [1, 2, ...], orient number [1, 0], 1 if one_num
            Other edge [2, 3, ...], orient number [2, 0], 2 if one_num
            Other edge [3, 0, ...], orient number [3, 0], 3 if one_num
            Other edge [0, 3, ...], orient number [1, 0], 4 if one_num
            Other edge [1, 0, ...], orient number [1, 1], 5 if one_num
            Other edge [2, 1, ...], orient number [1, 2], 6 if one_num
            Other edge [3, 2, ...], orient number [1, 3], 7 if one_num
        @endverbatim

        @see src.master.mk_basis_nodal.get_orient_num
        """
        return mkb.get_orient_num(orient, self.dim, one_num, reverse, ed_type)

    ###########################################################################

    def _get_elm2ed2elm_ids(self):
        """Calculates a mapping from elements to edges and vice versa.

        @retval elm2ed_ids (\c int) List of numpy arrays that gives the element
                            degree of freedom id of a particular element type
                            that corresponds to a degree of freedom id on the
                            edge.

        @retval ed2elm_ids (\c int) List of numpy arrays that gives the edge
                            degree of freedom id of a particular edge type
                            that corresponds to a degree of freedom id on the
                            element.

        @retval ed2elm_ids_right (\c int) List of numpy arrays that gives the
                            RIGHT edge degrees of freeom id of a particular
                            edge type that corresponds to a degree of freedom id
                            on the element. This is a subset of the ids of
                            ed2elm_ids, but is needed to negate the right edges
                            when a normal is involved in the calculation. (The
                            outward normal points in different directions for
                            the left, and right edges).

        @note To get the left edges  you use elm2ed_ids[0], and the right edges
        are elm2ed_ids[1]. The edge type 'j' is elm2ed_ids[i][j].

        The get the edge information for element type 'j', you would use
        ed2elm_ids[j].

        Care needs to be taken for the element types when doing the edges, and
        the edge types when doing the elements. See example code.

        @note These arrays should really be used directly through the
        get_elm2ed_array and get_ed2elm_array functions, and probably shouldn't
        be used directly.

        @see get_elm2ed_array
        @see get_ed2elm_array
        """
        #Shorter names
        master = self.master
        mesh = self.mesh

        #Initialize the edge ID arrays
        # CF: the 2 is because we are making 2 index maps; one for the left, one
        # for the right
        elm2ed_ids = [[-np.ones((nnbb, nnee), dtype=int)
            for i in range(2)]
            for nnbb, nnee in zip(self.nb_ed, self.n_ed_type)]

        #MPU: Figure out the number of IDS we need for the edge-to-volume mapping
        #For each element type
        numUniqueElmTypes = len(self.u_elm_type)

        # CF: totalEdgeDofPerElm will hold the total number of edge DOF for each
        # unique element type e.g., for a 2D mixed mesh of triangles and
        # squares, p = 2: [9, 12]
        totalEdgeDofPerElm = [0] * numUniqueElmTypes

        # CF: for each unique element type, edgeStartDofIdx keeps track of where
        # each edge starts in terms of edge degrees of freedom. This list will
        # hold lists corresponding to each element type.
        #
        # for example, in a 2D mixed mesh with triangles and quads with p = 2,
        # the lists in this list would be:  [[0, 3, 6], [0, 3, 6, 9]] since,
        # moving around the triangle, the three edges start at edge dof 0, 3,
        # and 6.  Likewise for a square, the edges start at dof 0, 3, 6, 9.
        edgeStartDofIdx = [[]  for i in range(numUniqueElmTypes)]
        for elmType in range(numUniqueElmTypes):
            for edgeType in master[elmType].basis.ed_type:
                # CF: get the index of the unique edge type corresponding to
                # this edge
                j = (self.u_ed_type == edgeType).nonzero()[0][0]
                # CF: append the current count of "total" edge dof per elm to
                # start a new edge
                edgeStartDofIdx[elmType].append(totalEdgeDofPerElm[elmType])
                # CF: update the total edge dof count by adding the number of
                # dof per that edge
                totalEdgeDofPerElm[elmType] += self.nb_ed[j]

        # CF: For each element type, create the index map array, which will has
        # number of edge dof rows and number of elements by column, same for the
        # right side counterpart array
        ed2elm_ids = [-np.ones((nnbb, nnee), dtype=int)
            for nnbb, nnee in zip(totalEdgeDofPerElm, self.n_elm_type)]

        ed2elm_ids_right = [np.zeros((nnbb, nnee), dtype=bool)
            for nnbb, nnee in zip(totalEdgeDofPerElm, self.n_elm_type)]

        # CF: for every edge in the mesh
        totNumMeshEdges = mesh.n_ed
        for edge in xrange(totNumMeshEdges):

            # MPU: First, get the global vertices and edge type
            globalEdgeVerts = mesh.get_global_vertex_numbers(globEdNum=edge)
            # CF: get the edge type corresponding to the current edge
            edgeType = mesh.ed_type[edge]
            # CF: glob2type_ed is the global edge number -> edge number within a
            # type conversion array, so we retrieve the typedEdgeNumber from the
            # (global) edge
            typedEdgeNumber = self.glob2type_ed[edge]

            # MPU: iterate over left and right edges (CF: does he mean elements
            # here?)
            for lrIdx, side in enumerate(['LEFT', 'RIGHT']):

                # First figure out which elements are on either side of the edge
                # CF: mesh.get_adjacent_elm() returns -1 as a sentinel value if
                # the edge is a boundary edge and has no adjacent element on
                # that side. Therefore we manually check to see that adjElm is
                # non-negative.
                #
                # Aside: Better for readability would be to return None, making
                # the check if adjElm is not None: do something
                adjElm = mesh.get_adjacent_elm(edge, side)
                if adjElm >= 0:

                    # CF: glob2type_elm is the global element number -> elm
                    # number within a type typedElmNum =
                    # self.glob2type_elm[globElmNum]
                    typedElmNum = self.glob2type_elm[adjElm]
                    adjElmLocEdge = mesh.get_adjacent_elm_local_edge(edge, side)
                    elmType = mesh.elm_type[adjElm]

                    #MPU: Then we have to find the orientation of these edges
                    # CF: recall that ids_ed give the vertex numbers on the
                    # master element which correspond to that local edge. These
                    # are supposed to be in CCW order to give the outward
                    # pointing normal on the master element in 3D
                    loc_ed_ids = master[elmType].basis.ids_ed[adjElmLocEdge]

                    # CF: Regardless of problem dimension, mesh.elm[...] is
                    # indexing into the global connectivity array. So here we
                    # are getting the global vertex numbers which make up the
                    # adjacent element. (This could include -1s)
                    adjElmGlobVerts = mesh.elm[adjElm, loc_ed_ids].ravel()

                    #MPU: For periodic meshes, we have to be careful, since
                    these #vertex number may not match the edge numbers.
                    if mesh.vertmap != None:
                        if not any(adjElmGlobVerts[0] == globalEdgeVerts):
                            adjElmGlobVerts = mesh.vertmap[adjElmGlobVerts]

                    # MPU: Here we compare the global element numbering in the
                    # volume to that of the edge. This returns a orientation
                    # integer value, which can be used in conjunction with
                    # master[elmType].ed_orient
                    #
                    # CF: The idea is that the order of vertices on the edge may
                    # not agree with the order of the vertices of the element
                    # side corresponding to the edge. In order to number the dof
                    # correctly, the relative orientations must be computed,
                    # which is done in self._get_orient_num, which itself wraps
                    # mkb.get_orient_num, which really does the heavy lifting.
                    ed_orient = self._get_orient_num(
                        [(adjElmGlobVerts[0] == globalEdgeVerts).nonzero()[0][0],
                        + (adjElmGlobVerts[1] == globalEdgeVerts).nonzero()[0][0]],
                        one_num=True)

                    #MPU: Finally, we fill the ed_left_ids matrix for this edge:
                    #
                    # CF: The added quantity gives the relative volume dof
                    # numbers on the edge
                    tofill = typedElmNum * self.nb[elmType] + master[elmType].ed_orient[adjElmLocEdge][ed_orient]
                    elm2ed_ids[edgeType][lrIdx][:, typedEdgeNumber] = tofill

                    #Lastly, we have to try and do the reverse operation. Where
                    #we have the solution on the edge, and we want to line that
                    #up with the elements

                    #Start by creating a dictionary that will map volume edge
                    #numbers in the ed_orient array straight to locations in
                    #ed_to_elm_ids
                    startid = typedEdgeNumber * self.nb_ed[edgeType]
                    ids = range(startid, startid + self.nb_ed[edgeType])
                    a = dict(zip(master[elmType].ed_orient[adjElmLocEdge][0], ids))

                    #Now with the dictionary in hand, find the orientation of the edge
                    #relative to the volume
                    ed2elm_orient = self._get_orient_num(
                          [(globalEdgeVerts[0] == adjElmGlobVerts).nonzero()[0][0],
                           (globalEdgeVerts[1] == adjElmGlobVerts).nonzero()[0][0]],
                          one_num=True)

                    #Now extract the ordering as numbered by the master element
                    ed2elm_orient = master[elmType].ed_orient[adjElmLocEdge][ed2elm_orient]

                    #Grab the ids that will be updated in the ed_to_elm_ids array
                    startid = edgeStartDofIdx[elmType][adjElmLocEdge]
                    ids = np.arange(startid, startid + self.nb_ed[edgeType])

                    #Finally, add the new ids into the array, using the dictionary to
                    #convert from the master element local numbering to the
                    #glob2type_edge numbering.
                    ed2elm_ids[elmType][ids, typedElmNum] = [a[num] for num in ed2elm_orient]

                    #And if this is a right edge, we set those entries in
                    #ed2elm_ids_right equal to True, so that we can select them
                    #later
                    if side == 'RIGHT':
                        ed2elm_ids_right[elmType][ids, typedElmNum] = True

        #Return the id arrays
        return elm2ed_ids, ed2elm_ids, ed2elm_ids_right

    def _get_elm2ed_array_1(self, field):
        """Takes data from the elements and returns that data on the edges
        formatted to correspond to edge-local coordinates.
        """
        # Initialize the output array
        field_ed_lr = [[np.empty((nnbb, nnee), dtype=float) for i in range(2)]
                        for nnbb, nnee in zip(self.nb_ed, self.n_ed_type)]

        for lr in [0, 1]:
            #We have to grab the data from the correct element types, so
            #store the element type on the one side of the edge in the
            #following array
            elm_num = copy.copy(self.mesh.ed2ed[:, lr * 2])

            #for boundary elements, elm_num < 0, so the wrong elm_type will be
            #selected -- we need to fix this
            elm_num[elm_num < 0] = self.mesh.ed2ed[elm_num < 0, 0]
            elm_types_at_ed = self.mesh.elm_type[elm_num].ravel()

            #Now, each edge type is stored in a different array, so we
            #have to loop over the arrays of different edge types
            for et in range(len(self.u_ed_type)):
                #Now we have to store the type of the element, for edges of type 'et'
                elm_types_at_ed_loc = elm_types_at_ed[self.mesh.ed_type == et]

                #Since each element type is stored in a different array, the
                #data sources come from different places, so we have to loop
                #over the element types
                for emt in range(len(self.u_elm_type)):
                    #The element data on the 'lr' side of the edges of type 'et'
                    #are stored in the following places
                    field_ed_lr[et][lr][:, elm_types_at_ed_loc == emt] = \
                        field[emt].ravel('F')\
                        [self.elm2ed_ids[et][lr][:, elm_types_at_ed_loc == emt]]

        return field_ed_lr

    def get_elm2ed_array(self, field):
        """Takes data from the volumes and returns that data on the elements
        formatted to correspond to element-local coordinates.

        @param field (\c float) List of numpy arrays that hold the value of a
                                function on each type of element.

        @retval field_ed_lr (\c float) List of list of numpy arrays holding the
                                    left and right function values from the
                                    elements on a particular element type.
                                    field_ed_lr[i][lr] is the element value on
                                    the edge of type i, coming from the
                                    left (lr = 0) or right (lr = 1)
        """
        #I SWEAR that there has to be a more efficient way of doing this.
        #Maybe I should just make everything a numpy array... that is a
        #possibility, instead of lists of numpy arrays. Anyway, this will
        #suffice for now... perhaps the empty arrays will be efficient.
        fshp = field[0].shape
        if len(fshp) == 2: #single scalar
            field_ed_lr = self._get_elm2ed_array_1(field)

        elif len(fshp) == 3: #Mutiple scalars of one vector
            field_ed_lr = [[np.empty((nnbb, fshp[1], nnee), dtype=float) \
                for i in range(2)] \
                for nnbb, nnee in zip(self.nb_ed, self.n_ed_type) ]

            for i in range(fshp[1]):
                tmpfield = [fld[:, i, :] for fld in field]
                tmp = self._get_elm2ed_array_1(tmpfield)
                for j in range(2):
                    for k in range(len(self.u_ed_type)):
                        field_ed_lr[k][j][:, i, :] = tmp[k][j]

        elif len(fshp) == 4: #Multiple vectors
            field_ed_lr = \
                [[np.empty((nnbb, fshp[1], fshp[2], nnee), dtype=float) \
                for i in range(2)] \
                for nnbb, nnee in zip(self.nb_ed, self.n_ed_type)]

            for i in range(fshp[1]):
                for ii in range(fshp[2]):
                    tmpfield = [fld[:, i, ii, :] for fld in field]
                    tmp = self._get_elm2ed_array_1(tmpfield)
                    for j in range(2):
                        for k in range(len(self.u_ed_type)):
                            field_ed_lr[k][j][:, i, ii, :] = tmp[k][j]
        return field_ed_lr

    def _get_ed2elm_array_1(self, field_ed, negate_right=False):
        """Takes data from the edges and returns that data on the elements
        formatted to correspond to element-local coordinates.
        """
        #Find total number of edge degrees of freedom for that element type
        totalEdgeDofPerElm = [0] * len(self.u_elm_type)
        for i in range(len(self.u_elm_type)):
            for eds in self.master[i].basis.ed_type:
                j = (self.u_ed_type == eds).nonzero()[0][0]
                totalEdgeDofPerElm[i] += self.nb_ed[j]

        #initialize the element array
        field_elm = [np.empty((nnbb, nnee), dtype=float) \
            for nnbb, nnee in zip(totalEdgeDofPerElm, self.n_elm_type)]

        #Loop over the different types of elements
        for emt in range(len(self.u_elm_type)): #Element type
            #WE have to fill each of the edge dof on an edge-by edge basis.
            #luckily they are in order, but we keep track of the order with
            #the ns (number-start) variable
            ns = 0
            for i in range(self.master[emt].ne):
                #Determine the edge type, so we can select the correct array
                #from where we get our data
                et = (self.master[emt].basis.ed_type[i]\
                    == self.u_ed_type).nonzero()[0][0]

                #advance the ids, that is, get the end id, ne (number-end)
                ne = ns + self.nb_ed[et]

                #Fill the field
                field_elm[emt][ns:ne, :] = \
                    field_ed[et].ravel('F')[self.ed2elm_ids[emt][ns:ne, :]]

                #advance the ids
                ns = ne
            #Now negate the elements on the right, if needed
            if negate_right:
                field_elm[emt][self.ed2elm_ids_right[emt]] =\
                    -field_elm[emt][self.ed2elm_ids_right[emt]]

        return field_elm

    def get_ed2elm_array(self, field_ed, negate_right=False):
        """Takes data from the edges and returns that data on the elements
        formatted to correspond to element-local coordinates.

        @param field_ed (\c float) List of numpy arrays that hold the value of a
                                function on each type of edge.

        @param negate_right (\c bool) Flag that, if true, will return the
                                negative edge values on the elements to the
                                right of the edge. This is because the normal
                                will point in opposite directions for left, and
                                right elements. So if a normal is present in
                                the calculations for the edges, this negation
                                is required.

        @retval field_elm (\c float) List of list of numpy arrays holding the
                                    function values from the edges on a
                                    particular element type on the element.
        """
        #I SWEAR that there has to be a more efficient way of doing this.
        #Maybe I should just make everything a numpy array... that is a
        #possibility, instead of lists of numpy arrays. Anyway, this will
        #suffice for now... perhaps the empty arrays will be efficient.
        totalEdgeDofPerElm = [0] * len(self.u_elm_type)
        for i in range(len(self.u_elm_type)):
            for eds in self.master[i].basis.ed_type:
                j = (self.u_ed_type == eds).nonzero()[0][0]
                totalEdgeDofPerElm[i] += self.nb_ed[j]

        fshp = field_ed[0].shape
        if len(fshp) == 2: #single scalar
            field_elm = self._get_ed2elm_array_1(field_ed, negate_right)

        elif len(fshp) == 3: #Mutiple scalars of one vector
            field_elm = [np.empty((nnbb, fshp[1], nnee), dtype=float) \
                for nnbb, nnee in zip(totalEdgeDofPerElm, self.n_elm_type) ]

            for i in range(fshp[1]):
                tmpfield = [fld[:, i, :] for fld in field_ed]
                tmp = self._get_ed2elm_array_1(tmpfield, negate_right)
                for k in range(len(self.u_elm_type)):
                    field_elm[k][:, i, :] = tmp[k]

        elif len(fshp) == 4: #Multiple vectors
            field_elm = \
                [np.empty((nnbb, fshp[1], fshp[2], nnee), dtype=float) \
                for nnbb, nnee in zip(totalEdgeDofPerElm, self.n_elm_type) ]

            for i in range(fshp[1]):
                for ii in range(fshp[2]):
                    tmpfield = [fld[:, i, ii, :] for fld in field_ed]
                    tmp = self._get_ed2elm_array_1(tmpfield, negate_right)
                    for k in range(len(self.u_elm_type)):
                        field_elm[k][:, i, ii, :] = tmp[k]
        return field_elm

    #@profile
    def _set_field_bc(self, field, funcfld, fldcoords=None, irreg=False):
        """Set boundary conditions for a general field

        @param field (\c float) Array of edge boundary condition degrees of
                        freedom

        @param funcfld Either a lambda function that takes the
                       inputs funcfld(points, time, bcid), OR
                       if fldcoords = None and funcfld[i] = array: it is assumed
                           that the supplied array is at the nodal points
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of values, fldcoords is an
                         array of floats, with
                         if irreg = True: fldcoords.shape = (n_points, dim),
                         if irreg = False: fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note This function interpolates the data/function unto the nodal points.
        """
        if type(funcfld) is not list:
            funcfld = [funcfld]

        if not callable(funcfld[0]):
            if fldcoords == None:
                #CM changed this branch 10/28/14
                print("ERROR: fldcoords needs to be set if funcfld is a list", \
                    "of numpy arrays instead of lambda functions.")
                #field = copy.copy(funcfld)
                #return field
            if irreg:
                pass
                #from scipy.interpolate import interpnd
                #funcfld = [interpnd.LinearNDInterpolator(fldcoords, vals) \
                #    for vals in funcfld]
            else:
                #For regularly-spaced data, it is faster to use
                #ndimage.map_coordinates, implemented in pyutil
                funcfld = [pyutil.Bi_linear_interp(fldcoords, vals, self.dim) \
                    for vals in funcfld]

        #CM added branch 10/28/2014
        #The first branch is "QF-type" BC enforcement
        qb_bcs = True
        #qb_bcs = False
        if not qb_bcs:
            #For each edge type in the mesh
            for j in range(len(self.u_ed_type)):
                #For each field (for each initialization function)
                for i in xrange(len(funcfld)):
                    field[j][:, i, :] = funcfld[i]\
                        (self.dgnodes_ed[j][:, :, self.ids_exterior_ed[j]], \
                        self.time, self.bcid[j])
        else:
            #CM ADDED THIS BRANCH 10/28/2014
            #NEEDS TO BE FIXED

            #Calculate where the cubature points are in real space
            cube_nodes_ed = self.mk_cube_nodes_ed()

            #For each edge-element type in the mesh
            for j in range(len(self.n_ed_type)):
                #Get the shape of this matrix
                #shape = cube_nodes_ed[j][:,:,self.ids_exterior_ed[j]].shape
                #print "edge cubature nodes shape = ", shape
                #This is d x dof/ed x ed
                tmp = cube_nodes_ed[j][:,:,self.ids_exterior_ed[j]]
                #print tmp[:,:,0]
                cnds = np.swapaxes(tmp,0,1)

                #Shorter names
                shap = self.master_ed[j].shap
                wghts = self.master_ed[j].cube_wghts
                shapw = np.dot(shap.T, np.diag(wghts))

                #For each tracer field (for each initialization function)
                #CM: Apply Gaussian quadrature to compute the integral
                #CM: This is the RHS of the system
                for i in xrange(len(funcfld)):
                    field[j][:, i, :] = np.dot(shapw, \
                        funcfld[i](cnds, self.time, self.bcid[j]) \
                            * self.cube_jac_ed[j][:,self.ids_exterior_ed[j]])

            #Now multiply with the inverse mass-matrix
            #We have to count the number of each edge type used already in order
            #to select the correct jacobian
            for j in xrange(len(self.n_ed_type)):
                shap = self.master_ed[j].shap
                wghts = self.master_ed[j].cube_wghts
                for i in xrange(field[j].shape[2]):
                    jacs = (self.cube_jac_ed[j][:, self.ids_exterior_ed[j]])[:,i].flatten()
                    M = np.dot(np.dot(shap.T, np.diag(wghts * jacs)), shap)
                    #if i == 1:
                    #    print "LHS M = "
                    #    print M
                    #    STOP
                    field[j][:, :, i] = np.linalg.solve(M, field[j][:, :, i])

#            if len(funcfld) == 3:
#                print "Result of linear solve = "
#                print field[0][:,:,1]
#                #STOP

        return field

    def set_trace_bc(self, funcfld, fldcoords=None, irreg=False):
        """Set the trace boundary conditions

        @param funcfld Either a lambda function that takes the
                       inputs funcfld(points, time, bcid), OR
                       if fldcoords = None and funcfld[i] = array: it is assumed
                           that the supplied array is at the nodal points
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of values, fldcoords is an
                         array of floats, with
                         if irreg = True: fldcoords.shape = (n_points, dim),
                         if irreg = False: fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns
        @note time: will use sol.time

        @note bcid: will use sol.bcid

        @note This function interpolates the data/function onto the nodal points.
        """
        self.trace_ed_bc = self._set_field_bc(\
            self.trace_ed_bc, funcfld, fldcoords, irreg)

    def set_vecs_bc(self, funcfld, fldcoords=None, irreg=False):
        """Set the trace boundary conditions

        @param funcfld Either a lambda function that takes the
                       inputs funcfld(points, time, bcid), OR
                       if fldcoords = None and funcfld[i] = array: it is assumed
                           that the supplied array is at the nodal points
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of values, fldcoords is an
                         array of floats, with
                         if irreg = True: fldcoords.shape = (n_points, dim),
                         if irreg = False: fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note time: will use sol.time

        @note bcid: will use sol.bcid

        @note This function interpolates the data/function unto the nodal points.

        @code
        >>> #The form of funcfld is a little confusing, so here is an example
        >>> x1_init = lambda x: x[:, 0]
        >>> y1_init = lambda x: x[:, 1]
        >>> z1_init = lambda x: x[:, 2]
        >>> x2_init = lambda x: x[:, 0]**2
        >>> y2_init = lambda x: x[:, 1]**2
        >>> z2_init = lambda x: x[:, 2]**2
        >>> funcfld =[[x1_init, y1_init, z1_init], [x1_init, y1_init, z1_init]]
        >>> Sol.set_vecs_bc(funcfld)
        @endcode
        """
        for i in xrange(len(funcfld)):
            tmp = [v[:, :, i, :] for v in self.vecs_ed_bc]
            tmp2 = self._set_field_bc(tmp, funcfld[i], fldcoords, irreg)
            for j in range(len(self.vecs_ed_bc)):
                self.vecs_ed_bc[j][:, :, i, :] = tmp2[j]

    def set_p_bc(self, funcfld, fldcoords=None, irreg=False):
        """Set the particular boundary conditions

        @param funcfld Either a lambda function that takes the
                       inputs funcfld(points, time, bcid), OR
                       if fldcoords = None and funcfld[i] = array: it is assumed
                           that the supplied array is at the nodal points
                       if irreg == True: a numpy array of the function
                                         evaluated at the points fldcoords
                       if irreg == False: a multi-dimensional numpy array of
                                          the function evaluated at
                                          equally-spaced points

        @param fldcoords When funcfld is a lambda function, this is not needed.
                         When funcfld is an array of values, fldcoords is an
                         array of floats, with
                         if irreg = True: fldcoords.shape = (n_points, dim),
                         if irreg = False: fldcoords = ogrid[xrange, yrange, zrange]

        @param irreg (\c bool) Flag to indicate whether provided data is
                      irregularly gridded. Default = False

        @note points: numpy array of floats. points.shape = (n_points, dim).
                      Each row contains one point with x,y,z coordinates in the
                      columns

        @note time: will use sol.time

        @note bcid: will use sol.bcid

        @note This function interpolates the data/function unto the nodal points.
        """
        self.p_ed_bc = self._set_field_bc(\
            self.p_ed_bc, funcfld, fldcoords, irreg)

    def save(self, filename, flags='wb'):
        """
        Saves the fields p, vecs, and trace into a file.
        NOTE, this does NOT save the entire object.

        @param filename (\c string) Name and path of file to save (.pkl will be
                                    appended)

        @param flags (\c string) The open flags. Default is 'wb' or write binary.
        @see file
        """
        #@return error (\c int) Return an error code. If successful, error=0.
        #"""
        savefile = open(filename  + '.pkl', flags)

        if self.n_p > 0:
            cPickle.dump(self.p, savefile)
            cPickle.dump(self.p_ed, savefile)
            cPickle.dump(self.p_ed_bc, savefile)

            #CM DEBUG Added to save pressure correction term
            #if self.deltap != None:
            #    cPickle.dump(self.deltap, savefile)
            #    cPickle.dump(self.deltap_ed, savefile)
            #    cPickle.dump(self.deltap_grad, savefile)

        if self.n_vec > 0:
            cPickle.dump(self.vecs, savefile)
            cPickle.dump(self.vecs_ed, savefile)
            cPickle.dump(self.vecs_ed_bc, savefile)

            #CM Added to save the first and second predictor velocities
#            if self.vecsp1 != None:
#                cPickle.dump(self.vecsp1, savefile)
#                cPickle.dump(self.vecsp1_ed, savefile)
#                cPickle.dump(self.vecsp2, savefile)
#                cPickle.dump(self.vecsp2_ed, savefile)

        if self.n_trace > 0:
            cPickle.dump(self.trace, savefile)
            cPickle.dump(self.trace_ed, savefile)
            cPickle.dump(self.trace_ed_bc, savefile)

        if self.time != None:
            cPickle.dump([self.time], savefile)

        savefile.close()

        #Any errors will be thrown by cPickle or open, so if it gets this far,
        #there should be no errors... could change behaviour in the future
        #return 0

    def load(self, filename, flags='rb'):
        """
        Loads the fields p, vecs, and trace from a file.
        NOTE, this does NOT load the entire object.

        @param filename (\c string) Name and path of file from which to load
                                    data. (.pkl will be appended)

        @param flags (\c string) The open flags. Default is 'rb' or read binary.
        @see file
        """
        #@return error (\c int) Return an error code. If successful, error=0.
        #"""
        loadfile = open(filename + '.pkl', flags)

        load_intermediates = True

        if self.n_p > 0:
            self.p = cPickle.load(loadfile)
            self.p_ed = cPickle.load(loadfile)
            self.p_ed_bc = cPickle.load(loadfile)

            #CM Added to load pressure correction term
#            if load_intermediates:
#                self.deltap = cPickle.load(loadfile)
#                self.deltap_ed = cPickle.load(loadfile)
#                self.deltap_grad = cPickle.load(loadfile)

        if self.n_vec > 0:
            self.vecs = cPickle.load(loadfile)
            self.vecs_ed = cPickle.load(loadfile)
            self.vecs_ed_bc = cPickle.load(loadfile)

            #CM Added to load first and second predictor velocities
#            if load_intermediates:
#                self.vecsp1 = cPickle.load(loadfile)
#                self.vecsp1_ed = cPickle.load(loadfile)
#                self.vecsp2 = cPickle.load(loadfile)
#                self.vecsp2_ed = cPickle.load(loadfile)

        if self.n_trace > 0:
            self.trace = cPickle.load(loadfile)
            self.trace_ed = cPickle.load(loadfile)
            self.trace_ed_bc = cPickle.load(loadfile)

        if self.time != None:
            self.time = cPickle.load(loadfile)
            if type(self.time) == list:
                self.time = self.time[0]

        loadfile.close()

        #Any errors will be thrown by cPickle or open, so if it gets this far,
        #there should be no errors... could change behaviour in the future
        #return 0
