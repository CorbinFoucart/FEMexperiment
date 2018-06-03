#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Mesh1D(object):
    def __init__(self, P):
        """ @param P  vertex points, sorted by x position """
        self.verts = P
        self.nElm, self.nVerts, self.nEdges = len(self.verts) - 1, len(self.verts), len(self.verts)
        self.connectivity = self.build_T()
        connected_one_side = np.bincount(self.connectivity.ravel()) == 1
        self.boundary_verts = np.where(connected_one_side)[0]

    def build_T(self):
        """ element connectivity array from 1D vertex list """
        T = np.zeros((self.nElm, 2), dtype=int)
        T[:,0] = np.arange(self.nElm)
        T[:,1] = np.arange(self.nElm) + 1
        return T
