#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DOFHandler1D(object): pass

class DG_DOFHandler1D(DOFHandler1D):
    def __init__(self, mesh, master):
        self.mesh, self.master = mesh, master
        self.n_dof = self.master.nb * self.mesh.nElm
        self.dgnodes = self.mk_dgnodes()
        self.lg = self.mk_lg()
        self.lg_PM = self.mk_minus_plus_lg()
        self.nb, self.nElm = self.master.nb, self.mesh.nElm
        self.ed2elm = self.mk_ed2elm()

    def mk_dgnodes(self):
        """ map master nodal pts to element vertices def'd in self.mesh """
        dgn = np.zeros((self.master.nb, self.mesh.nElm))
        master_nodal_pts = np.squeeze(self.master.nodal_pts)
        for elm, elm_verts in enumerate(self.mesh.connectivity):
            elm_vert_pts = self.mesh.verts[elm_verts]
            elm_width = elm_vert_pts[1] - elm_vert_pts[0]
            mapped_pts = elm_vert_pts[0] + (1+master_nodal_pts)/2.*(elm_width)
            dgn[:, elm] = mapped_pts
        return dgn

    def mk_lg(self):
        """ number all dof sequentially by dgnodes """
        node_numbers = np.arange(np.size(self.dgnodes))
        lg = node_numbers.reshape(self.dgnodes.shape, order='F')
        return lg

    def mk_minus_plus_lg(self):
        """ (-) denotes element interior, (+) denotes exterior"""
        lg_PM = dict()
        lg_PM['-'] = self.lg[[0, -1], :].ravel(order='F')
        lgP = self.lg[[0, -1],:]
        lgP[0, 1: ] -= 1 # shift nodes to left of first
        lgP[1, :-1] += 1 # shift nodes to right of last
        lg_PM['+'] = lgP.ravel(order='F')
        return lg_PM

    def mk_ed2elm(self):
        """ internal map holding the indicies to reshape vector of values on faces to
        element edge space (2, nElm), duplicating the values on either side of interior faces
        """
        f2elm = np.zeros((2, self.nElm))
        faces = np.arange(self.mesh.nEdges)
        # numpy magic is doing the following:
        # [[0, 1, 2, 3]
        #  [0, 1, 2, 3]]  - ravel('F') -> [0, 0, 1, 1, 2, 2, 3, 3]
        #  this is close, but ends duplicated. => trim the ends and reshape to f2elm shape
        # [[0, 1, 2]
        #  [1, 2, 3]]
        f2elm = np.vstack((faces, faces)).ravel( order='F')[1:-1].reshape(f2elm.shape, order='F')
        return f2elm

    def edge2elm_ed(self, arr):
        """ internal method to move edge values (defined on the interfaces)
        to values on the "element edge space", the edge dof interior to each element
        @param arr  array formatted on edge space (nFaces,)
        @retval elmEdArr  array formatted on "element edge space" (2, nElm)
        """
        return arr[self.ed2elm]
