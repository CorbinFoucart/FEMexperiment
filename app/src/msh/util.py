'''
Created on Jul 11, 2010

@author: Matt Ueckermann
'''

import pdb
import numpy as np
import src.pyutil as util

def simpvol2D(t, p):
    """Determine the volume of the 2D simplexes.
    @param t (\c int) Triangulation matrix
    @param p (\c float) Vertex matrix
    @see msh.mesh.Mesh2D
    @retval CCW Boolean array. True if numbered CCW, false otherwise.

    @author Matt Ueckermann
    """
    m, n = t.shape

    #Test for the 3 element types: 1D line, 2D triangle, or 2D square
    ids1D = np.nonzero(t[:, 2] < 0)
    if n > 3:
        ids2Dtri = np.nonzero((t[:, 3] < 0) & (t[:, 2] >= 0))
        ids2Dquad = np.nonzero(t[:, 3] >= 0)
    else:
        ids2Dtri = np.nonzero(t[:, 2] >= 0)

    vol = np.zeros(m)
    vol[ids1D] = p[t[ids1D, 1], 0] - p[t[ids1D, 0], 0]

    d12 = p[t[ids2Dtri, 1], :] - p[t[ids2Dtri, 0], :]
    d13 = p[t[ids2Dtri, 2], :] - p[t[ids2Dtri, 0], :]
    vol[ids2Dtri] = (d12[0][:, 0] * d13[0][:, 1] \
                   - d12[0][:, 1] * d13[0][:, 0]) / 2

    if n > 3:
        d12 = p[t[ids2Dquad, 1], :] - p[t[ids2Dquad, 0], :]
        d13 = p[t[ids2Dquad, 2], :] - p[t[ids2Dquad, 0], :]
        d34 = p[t[ids2Dquad, 3], :] - p[t[ids2Dquad, 2], :]
        d41 = p[t[ids2Dquad, 0], :] - p[t[ids2Dquad, 3], :]
        vol[ids2Dquad] = (d12[0][:, 0] * d13[0][:, 1]
                        - d12[0][:, 1] * d13[0][:, 0]) / 2 \
                        + (d34[0][:, 0] * d41[0][:, 1]
                        - d34[0][:, 1] * d41[0][:, 0]) / 2
    return vol

####################################################################################################
# fix T, P arrays -- remove duplicate elements, vertices CCW
####################################################################################################
def fix(t, p):
    """Fixes a faulty connectivity matrix and removes duplicate nodes
    1) remove duplicate elements from T
    2) change the connectivity array such that all vertices are ordered CCW

    @param t [@c numpy.ndarray]: Element connectivity matrix
    @param p [@c numpy.ndarray]: Vertex matrix
    @retval t Corrected connectivity matrix
    @retval p Corrected vertex matrix

    @author Matt Ueckermann
    @author: Corbin Foucart -- re-write
    """
    t = eliminate_duplicate_elements(t)
    _, _, ids2Dquad = get_element_type_indices(t)
    fix_CrissCross_quad_nodes(t, p, ids2Dquad)
    flip_CW_elements(t, p, ids2Dquad)
    return t, p

def eliminate_duplicate_elements(T):
    """ removes non-unique rows from connectivity array T
    NOTE: result sorted by first node number
    """
    return np.unique(T, axis=0)

def get_element_type_indices(t):
    """ returns 3 bool arrays denoting 1D, TRI, QUAD element types
    MPU connectivity array T design is such that:
        1D:  will have -1 in third column
        Tri: will have -1 in fourth column AND won't have -1 in third
        Quad: will have an entry in fourth column
    """
    ids1D = t[:, 2] < 0
    ids2Dtri = (t[:, 3] < 0) & (t[:, 2] >= 0)
    ids2Dquad = (t[:, 3] >= 0)
    return ids1D, ids2Dtri, ids2Dquad

def fix_CrissCross_quad_nodes(t, p, ids2Dquad):
    """ untangles quad nodes are ordered in a non CW or CCW (criss-cross) manner
    @author Matt Ueckermann -- factored out
    NOTE: after, all quads should either be CW or CCW, but not criss-crossed
    """
    #For quads (or rectangles) we have to make sure there is no cross-over
    #compare the volume calculated on the two triangles making up the quad to
    #how simpvol calculates the area. If there is a descrepancy, there is a
    #cross-over
    flip = abs(simpvol2D(t[np.ix_(ids2Dquad, [0, 1, 2])], p)) \
         + abs(simpvol2D(t[np.ix_(ids2Dquad, [2, 3, 0])], p)) \
         > abs(simpvol2D(t[np.ix_(ids2Dquad, [0, 1, 2, 3])], p)) + 1e-12

    #There are two types of cross-overs, correct the first type
    t[np.ix_(np.nonzero(ids2Dquad)[0][flip], [0, 1, 2, 3])] = \
                        t[np.ix_(np.nonzero(ids2Dquad)[0][flip], [0, 2, 1, 3])]

    #Check if there are still cross-overs (of the second type)
    flip = abs(simpvol2D(t[np.ix_(ids2Dquad, [0, 1, 2])], p)) \
         + abs(simpvol2D(t[np.ix_(ids2Dquad, [2, 3, 0])], p)) \
         > abs(simpvol2D(t[np.ix_(ids2Dquad, [0, 1, 2, 3])], p)) + 1e-12

    #Correct type 2 cross-overs
    t[np.ix_(np.nonzero(ids2Dquad)[0][flip], [0, 1, 2, 3])] = \
                        t[np.ix_(np.nonzero(ids2Dquad)[0][flip], [0, 1, 3, 2])]

def flip_CW_elements(t, p, ids2Dquad):
    """ flips CW tris and quads to CCW ordering
    Computes the signed volumne of every element in the T connectivity array.
    - If the signed volume is positive, the element is CCW already
    - If the signed volume is 0,        the element is a 1D element
    - If the signed volume is negative, the element is CW
    Permuting two nodes fixes a CW tri, permuting an additional nodes fixes a quad
    """
    flip = simpvol2D(t, p) < 0
    t[np.ix_(flip, [0, 1])] = t[np.ix_(flip, [1, 0])]
    #Quads have one additional flip
    t[np.ix_((flip & (ids2Dquad)), [2, 3])] =  t[np.ix_((flip & (ids2Dquad)), [3, 2])]

def sort_by_element_type(T):
    """ sorts the connectivity array so that triangles are listed first, then quads
    fourth column is -1 for tri, positive for quad
    """
    return sortrows_by_column(T, col=3)

def sort_by_vertex_number(T):
    """ first vertex is in column 0 """
    return sortrows_by_column(T, col=0)

def sortrows_by_column(arr, col):
    """ sorts rows according to val in col """
    return arr[arr[:,col].argsort()]

####################################################################################################
# build mesh connectivity data structures
####################################################################################################
def connect_elm(elm, elm_type, dim, n_types=None):
    ''' Function to compute element to element connectivity
    @param  elm     The triangulation matrix that defines elements
    @param  elm_type     The master types of elements (a list of length n_elm), p.134 MPU thesis
    @param  dim     The dimension of the problem (1, 2, or 3)
    @param  n_types List containing the unique element types. If not specified
                    it is calculated as unique(elm_type)
    @retval elm2elm The face connectivity matrix (same size as elm)
    @retval ed2ed   The edge connectivity matrix (used to determine left and right elements)

    @verbatim
    For example, elm2elm[0,:] = [10, 4, 5, -1] means that element 0 is
    connected to elements 10, 4, and 5 through the first, second and third
    edges of element 0. The '-1' indicates that this mesh also has elements with
    four edges, but the current element only has three.

    For example, ed2ed[0,:] = [ 12.,   0.,   6.,   2.,  18.,  19.] means that
    edge 0 is bordered by the edge 0 of element 12 on one side, and the
    edge 2 of element 6 on the other. Also, this face is made up of the two
    global vertices 18 and 19.
    @endverbatim

    '''
    from src.fem_base.master.mk_basis import int_el_pqr
    if n_types == None:
        n_types = util.unique(elm_type)

    ids_ed = [None] * len(n_types)
    # Need to know both the maximum edge length and how many vertices in the edge with the most
    # vertices

    max_ed_len = 0

    #Number of edges (CF: where edges means "faces")
    n_ed = 0
    for i in range(len(n_types)):
        #Get the face definitions from the master scripts
        #CF: Note that n_types refer to master element types
        #    gets the vertices and edge numbers on the master elm,
        #    iterate over edge i, where "edge" means "face"
        verts, ids_ed[i] = int_el_pqr(dim=dim, element=n_types[i])
        #Increase number of edges by edgeType * n elements with that master elm type
        n_ed = n_ed + len(ids_ed[i]) * np.sum(elm_type == n_types[i])

        # CF: if we have encountered a larger "edge" length, set it as the max
        if len(ids_ed[i][0]) > max_ed_len:
            max_ed_len = len(ids_ed[i][0])

    #Create a list of faces. For faces that have fewer points than the max
    #number of points, the 'imaginary' vertex number of '-1' is used. For
    #example. In a 3D mesh that have both triangular and quadrilateral faces,
    #the fourth vertex making up a triangular face will be '-1'. This was it
    #never matches a quadrilateral face, and two triangular faces will match

    ed = -np.ones((n_ed, max_ed_len), dtype=int)
    elm_num = np.zeros((n_ed, 2), dtype=int)
    ns_ed = 0 #Needed to get the indexing on ed correct
    for i in range(len(n_types)):
        #Make sure we only select the elements of the current type
        ids = (elm_type == n_types[i]) # find all ids in elm which match this master type
        n_elm = np.sum(ids) # number of elements matching this master type

        # for each face of the master element type
        for j in range(len(ids_ed[i])):
            #Get indexing for ed correct
            ne_ed = n_elm + ns_ed
            #Number of vertices in edge definition
            n_ved = len(ids_ed[i][j])
            #Each edge is defined by the global vertex numbers
            ed[ns_ed:ne_ed, 0:n_ved] = elm[np.ix_(ids, ids_ed[i][j])]
            #We also need an index matrix to record to which element each
            #edge belongs, and what the local edge number is for the element
            elm_num[ns_ed:ne_ed, 0] = ids.nonzero()[0]
            elm_num[ns_ed:ne_ed, 1] = j
            #move ed index for next loop iteration
            ns_ed = ne_ed

    #Sort the edges so that the edge numbering will be consistent. Ie. the
    #lowest numbered vertex is always in column 1
    ed = np.sort(ed, 1)

    #Now the fun part, we find the unique edges
    #CM: Why would they be non-unique? B/c each interior face listed twice
    ed2edtmp, I1, J = util.unique(ed, 'rows', True)

    #Initialize ed2ed matrix
    # CF: why 4 + max_ed_len?
    ed2ed = -np.ones((len(ed2edtmp), 4 + max_ed_len), dtype=int)
    ed2ed[:, 0:2] = elm_num[I1]

    I2 = np.argsort(J)
    J = J[I2]
    elm_num = elm_num[I2, :]
    I3 = J[1:] == J[0:-1]
    I3 = np.append(I3, [False])
    elm_num1 = elm_num[I3, :]
    elm_num2 = elm_num[I3.nonzero()[0] + 1, :]

    #Finish creating ed2ed matrix
    ed2ed[J[I3.nonzero()[0] + 1], 0:2] = elm_num1
    ed2ed[J[I3.nonzero()[0] + 1], 2:4] = elm_num2

    #We also have to define the edge vertices according to the numbering used
    #by one of the elements
    for i in range(len(ed2ed)):
        #If there are multiple types of elements, ids_ed will have more than
        #one elements, and 2D quads will have index 1. If there is only one type
        #of element, ids_ed will only have one index, and 2D quads will have
        #index 0 instead of elm_type[ed2ed[i, 0]]
        if len(n_types) > 1:
            cur_elm_type = elm_type[ed2ed[i, 0]]
        else:
            cur_elm_type = 0
        cur_elm = ed2ed[i, 0]
        loc_ed = ed2ed[i, 1]
        n_ved = len(ids_ed[cur_elm_type][loc_ed])
        ed2ed[i, 4:n_ved + 4] = elm[cur_elm, ids_ed[cur_elm_type][loc_ed]]

    #Sort boundaries at the bottom
    ed2ed = np.flipud(util.sortrows(ed2ed, 2))

    #Sort by element
    ids = ed2ed[:, 2] >= 0
    if any(ids):
        ed2ed[ids, :] = util.sortrows(ed2ed[ids, :], 0)
    ids = ids == False
    if any(ids):
        ed2ed[ids, :] = util.sortrows(ed2ed[ids, :], 0)

    #Build elm to elm matrix
    n_elm = len(elm) #get number of elements
    n_ed_in_elm = len(elm[0])
    elm2elm = -np.ones(elm.shape, dtype=int).ravel()
    elm2elm[n_ed_in_elm * elm_num1[:, 0] + elm_num1[:, 1]] = elm_num2[:, 0]
    elm2elm[n_ed_in_elm * elm_num2[:, 0] + elm_num2[:, 1]] = elm_num1[:, 0]
    elm2elm = elm2elm.reshape(elm.shape)
    return elm2elm, ed2ed

def connect_elm2ed(elm2elm, ed2ed):
    """Create the elm2ed connectivity matrix"""
    elm2ed = -np.ones_like(elm2elm)
    ednum = 0
    for ed in ed2ed[:, 0:4]:
        elm2ed[ed[0], ed[1]] = ednum
        if ed[2]>=0:
            elm2ed[ed[2], ed[3]] = ednum
        ednum += 1

    if (elm2ed[:, :5] < 0).any():
        print("something failed miserably :(")

    return elm2ed
