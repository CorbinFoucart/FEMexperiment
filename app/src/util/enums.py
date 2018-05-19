# -*- coding: utf-8 -*-

"""@file enums.py
Provides a set of enumerated types for the problem setup scripts

@author Chris Mirabito (mirabito@mit.edu)
"""


class Boundary(object):
    """Defines the boundary enumeration"""
    WEST, EAST, SOUTH, NORTH, SURFACE, BOTTOM = range(6)


class BCType(object):
    """Defines the boundary condition type enumeration"""
    DIRICHLET, NEUMANN, ZERO_NEUMANN = range(3)


class Velocity(object):
    """Defines the velocity component enumeration"""
    u, v, w = range(3)


class ElementType(object):
    """Defines the element type enumeration used for domain masking"""
    TRI_LEFT = -1
    BLANK = 0
    TRI_RIGHT = 1
    QUAD = 2
