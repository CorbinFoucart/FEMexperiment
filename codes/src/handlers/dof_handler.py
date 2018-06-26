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

    def mk_nodal_ed_points(self):
        master_ed_nodal_pts = [Med.basis.nodal_pts for Med in self.master_ed]
        nodes = util.mk_nodes_ed(self, master_ed_nodal_pts)
        return nodes

    def mk_dgnodes_ed(self):
        dgnodes = [dgn.swapaxes(0, 1) for dgn in self.mk_nodal_ed_points()]
        return dgnodes

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
        #self.dgnodes_ed = self.mk_dgnodes_ed()

    def count_ed_dof(self):
        ed_dof = 0
        for master_ed, n_ed in zip(self.master_ed, self.n_elm_type):
            n_ed_basis_fns = master_ed.nb
            ed_dof += n_ed_basis_fns * n_ed
        return ed_dof

    def index_map_ed2elm(): pass

