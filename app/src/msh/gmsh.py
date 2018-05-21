# -*- coding: utf-8 -*-
# pylint: disable=E1101,C0103

r"""@package gmsh
Contains utilities for reading data from a GMSH-generated mesh file

@author Chris Mirabito (mirabito@mit.edu)
"""

import numpy
import struct

class GmshElement(object):
    """Enumeration of GMSH element types"""
    TRIANGLE = 2
    QUAD = 3


class GmshFileReader(object):
    """Reads and processes a GMSH-generated file"""

    __DIM = 3
    __MAX_VERT = 4
    __VERSION = '2.2'
    __FILE_TYPE = '1'
    __DATA_SIZE = '8'

    def __init__(self, name):
        """Constructs a new reader object from the GMSH file name
        @param name [@c string]: Name of the GMSH file
        """
        self.name = name

    def read_ascii(self):
        """Reads a GMSH .msh file in ASCII format
        @retval elts [@c numpy.ndarray]: Element connectivity matrix
        @retval verts [@c numpy.ndarray]: Vertex list
        """
        with open(self.name, 'r') as f:
            for line in f:
                if '$Nodes' in line:
                    num_verts = int(f.next())
                    verts = numpy.empty((num_verts, GmshFileReader.__DIM))
                    for i in xrange(num_verts):
                        verts[i, :] = f.next().split()[1:]
                elif '$Elements' in line:
                    num_elts = int(f.next())
                    elts = numpy.zeros((num_elts, GmshFileReader.__MAX_VERT),
                                       dtype=numpy.int)
                    for i in xrange(num_elts):
                        words = f.next().split()
                        elt_type, num_tags = int(words[1]), int(words[2])
                        start_col = 3 + num_tags
                        if elt_type == GmshElement.TRIANGLE:
                            elts[i, :-1] = words[start_col:]
                        elif elt_type == GmshElement.QUAD:
                            elts[i, :] = words[start_col:]
                    if all(elts[:, -1] == 0):
                        elts = elts[:, :-1]
                    elts = elts[~numpy.all(elts == 0, axis=1)]
                    elts -= 1
                    break
        return elts, verts

    def read_binary(self):
        """Reads a GMSH .msh file in binary format
        @retval elts [@c numpy.ndarray]: Element connectivity matrix
        @retval verts [@c numpy.ndarray]: Vertex list
        """
        with open(self.name, 'rb') as f:
            if f.readline() != '$MeshFormat\n':
                raise ValueError('Missing or bad GMSH file section header')
            version, file_type, data_size = f.readline().split()
            if version != GmshFileReader.__VERSION:
                raise ValueError('Missing or bad GMSH file version number')
            if file_type != GmshFileReader.__FILE_TYPE:
                raise ValueError('Missing or bad GMSH file type')
            if data_size != GmshFileReader.__DATA_SIZE:
                raise ValueError('Missing or bad GMSH data size')

            one = struct.unpack('i', f.read(4))[0]
            if one & (1 << (8*4-1)):
                raise ValueError('Endianness of GMSH file and system differ')
            elif one != 1:
                raise ValueError('Could not read GMSH endianness identifier')
            f.read(1)
            if f.readline() != '$EndMeshFormat\n':
                raise ValueError('Missing or bad GMSH file section footer')
            if f.readline() != '$Nodes\n':
                raise ValueError('Missing or bad GMSH file section header')
            num_verts = int(f.readline())
            verts = numpy.empty((num_verts, GmshFileReader.__DIM))
            for i in xrange(num_verts):
                struct.unpack('i', f.read(4))
                verts[i, :] = struct.unpack('%dd' % GmshFileReader.__DIM,
                    f.read(int(GmshFileReader.__DATA_SIZE) \
                        *GmshFileReader.__DIM))
            f.read(1)
            if f.readline() != '$EndNodes\n':
                raise ValueError('Missing or bad GMSH file section footer')
            if f.readline() != '$Elements\n':
                raise ValueError('Missing or bad GMSH file section header')
            num_elts = int(f.readline())
            elts = numpy.zeros((num_elts, GmshFileReader.__MAX_VERT),
                               dtype=numpy.int)
            elts_read = 0
            while elts_read < num_elts:
                elt_type, num_elts_following, num_tags = \
                    struct.unpack('3i', f.read(3*4))
                if elt_type == GmshElement.TRIANGLE:
                    elt_verts = 3
                elif elt_type == GmshElement.QUAD:
                    elt_verts = 4
                else:
                    raise ValueError('Invalid element type')
                for j in xrange(num_elts_following):
                    struct.unpack('%di' % (num_tags + 1),
                                  f.read(4*(num_tags + 1)))
                    for k in xrange(elt_verts):
                        elts[elts_read, k] = struct.unpack('i', f.read(4))[0]
                    elts_read += 1
            if all(elts[:, -1] == 0):
                elts = elts[:, :-1]
            elts = elts[~numpy.all(elts == 0, axis=1)]
            elts -= 1
        return elts, verts

