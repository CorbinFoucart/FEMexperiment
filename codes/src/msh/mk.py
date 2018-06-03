'''
Created on Jul 8, 2010

@author: Matt Ueckermann
'''

import numpy as np
from src.msh.util import fix
import copy
from src.pyutil import sortrows

def struct2D(EMASK=None, dims=None, ydims=None):
    """ Function that takes a masking matrix, and makes a structured 2D grid of
    triangles.

    @param EMASK The masking matrix. EMASK == 0 indicates a masked point,
                  whereas EMASK == 1 indicates a square CV that will be divided
                  into two triangular elements with the diagonal from bottom-
                  left to top right, EMASK == -1 also triangles but with
                  diagonal from top left to bottom right, and EMASK == 2
                  indicates a square element.
    @param dims  The dimensions of the mesh. dims = [x0, y0, dx, dy], where x0
                 and y0 are the smallest x and y coordinate values respectively,
                 and dx, dy are the sizes of the steps in the x, and y
                 directions respectively.
                 BY DEFAULT: the domain will be [0,0][1,0],[1,1],[0,1] if dims
                             is unspecified.
                 IF ydims != None: dims takes on a different value. See next.

    @param ydims The dimensions of the mesh can also be specified by the x and y
                 limites. If ydims exists, ydims = [ymin, yxmin] gives the min
                 and max y-dimensions. In contrast, dims = [xmin, xmax] gives
                 the min and max x-dimensions

    @retval t    The 'triangulation' matrix. Returns the three vertices that
                  make up an element.

    @retval p    The 'points' matrix. Returns the coordinates (in 3D) of the
                 (x,y,z) locations of the vertex points in the triangulation.
                 Since this function only creates 2D meshes, z=0 by default.


    @code              Example: [1 1 1;
                                1 1 1;
                                1 0 1];
                        Defines a 3x3 structured mesh with a bump on the
                        bottom of the domain.
    @endcode
    """

    #Assign some default values if the user inputs nothing: used for testing
    if EMASK is None:
        #Debugging assignments
        EMASK = np.ones((4, 5),dtype=np.float)
        EMASK[1:4,0] = -1
        EMASK[1:4,3] = 2
        EMASK[np.array([[2,3,2]]),np.array([[1,1,2]])] = 0

    #Get the size of the 'grid'
    m, n = EMASK.shape

    #Assign some default values if the user inputs nothing: used for testing
    if dims == None:
        dims = np.array([0., 0., 1./n, 1./m])
        x0 = dims[0]
        y0 = dims[1]
        dx = dims[2]
        dy = dims[3]
    elif ydims != None:
        x0 = dims[0]
        y0 = ydims[0]
        dx = float(dims[1] - dims[0]) / float(n)
        dy = float(ydims[1] - ydims[0]) / float(m)
    else:
        x0 = dims[0]
        y0 = dims[1]
        dx = dims[2]
        dy = dims[3]

    #The orientation as specified by the user in up-down opposite to what's
    #convenient computationally
    EMASK = np.flipud(EMASK)


    """
    #bndexpr = {sprintf('all(abs(p(:,2)-%g)<%g)',y0,  abs(dy*1e-1)), \
    #        sprintf('all(abs(p(:,1)-%g)<%g)',x0+dx*(n-1),abs(dx*1e-1)), ...
    #        sprintf('all(abs(p(:,2)-%g)<%g)',y0+dy*(m-1),abs(dy*1e-1)), ...
    #        sprintf('all(abs(p(:,1)-%g)<%g)',x0         ,abs(dx*1e-1))};
    """

    # Create Reference matrix for ease of coding
    E = np.arange(n * m)
    E = E.reshape(n, m).transpose()

    # Triangulation matrix, contains indices to nodes which make up an element
    # They start at 1 because when we remove the masked elements, we test
    # to see if the first element is zero (multiplication by zero).
    t = np.zeros((EMASK.size*2,4),dtype=int)
    for i in xrange (n):
        for ii in xrange(m):
            #Triangles of parity -1
            if EMASK[ii, i] == -1:
                t[E[ii, i] * 2, :3] = np.array([i * (m + 1) + ii + 1, \
                                               i * (m + 1) + ii + 2, \
                                         (i + 1) * (m + 1) + ii + 1])

                t[E[ii, i] * 2 + 1, :3] = np.array([(i + 1) * (m + 1) + ii + 1,\
                                               i * (m + 1) + ii + 2, \
                                         (i + 1) * (m + 1) + ii + 2])

            #Triangles of parity 1
            elif EMASK[ii, i] == 1:
                t[E[ii, i] * 2, :3] = np.array([(i) * (m + 1) + ii + 2, \
                                               (i + 1) * (m + 1) + ii + 2, \
                                               (i) * (m + 1) + ii + 1])

                t[E[ii, i] * 2 + 1, :3] = np.array([(i + 1) * (m + 1) + ii + 1, \
                                               (i) * (m + 1) + ii + 1, \
                                               (i + 1) * (m + 1) + ii + 2])
            #Squares
            elif abs(EMASK[ii, i]) == 2:
                t[E[ii, i] * 2, :4] = np.array([(i) * (m + 1) + ii + 1, \
                                                (i) * (m + 1) + ii + 2, \
                                                (i + 1) * (m + 1) + ii + 2, \
                                                (i + 1) * (m + 1) + ii + 1])


    # Remove masked elements, and subtract one so that the indices in t
    # correspond to the rows in p
    t = t[t[:, 1].nonzero()[0], :] - 1

    # This array contains the values of the coordinates. The rows of this array
    # correspond to the indices used in t
    p = np.zeros(((EMASK.shape[0] + 1) * (EMASK.shape[1] + 1), 3))
    for i in xrange(n + 1):
        p[range(i * (m + 1), (m + 1) * (i + 1)) ,0] = x0 + dx * i
        p[range(i * (m + 1), (m + 1) * (i + 1)) ,1] = y0 + dy * \
                                    np.arange(0, m + 1)

    #### Remove unwanted/masked vertices
    # Variable used to keep accurate indexing
    vremoved = 0

    for i in xrange((n + 1) * (m + 1)):
        #If no element contains the current node
        ids = np.transpose((t == i - vremoved).nonzero())
        if ids.size == 0:
            t[t > i - vremoved] -= 1
            p = np.delete(p, i - vremoved, 0)
            vremoved += 1

    #Fix mesh to conform to certain standards
    t, p = fix(t, p)

    return t, p

###############################################################################
def delta_z(vert, levels, bathys, levfrac=None):
    """ This function creates a list of increments in z (\f$\Delta z\f$) for
    each vertex based on any number of sigma-coordinates.

    @param vert      (\c float) Numpy array of of x-y coordinates
    @param levels    (\c int) List of number of levels in each sigma-layer slice
    @param bathys    (\c float) List of numpy arrays of size [len(vert) x 1].
                     The i'th numpy array in the list gives the depth of the
                     i'th sigma-layer. Note bathymetry should be negative!
    @param levfrac   (\c float) List of numpy arrays of size [len(levels) x 1]
                     giving the fraction that each level takes of the total
                     number of levels in that sigma layer.
                     Note: sum(levfrac[i]) = 1 should be true, for all layers
                     i = 0:len(levfrac), but this is automatically enforced.

    @author Matt Ueckermann
    """

    #initialize the dz matrix
    dz = np.zeros((len(vert), np.sum(levels)), dtype=float)



    #get the number of sigma layers
    if type(levels)==list:
        n_sigma = len(levels)
    else:
        n_sigma = 1
        #put everything in lists for consistency
        levels = [levels]
        bathys = [bathys]
        levfrac = [levfrac]

    #For ease of coding, we modify the bathys list a little. The zero level is
    #put at the end, so that when bathys[-1] is referenced (when doing the first
    #level) the correct total zdepth for that layer is calculated
    bathys = bathys + [0]
    #keep track of the current total level depth
    levnum = 0
    for i in range(n_sigma):
        #If levfrac is None, use uniform spacing
        if levfrac[i] == None:
            levfrac[i] = np.array([1. / levels[i]] * levels[i])
            #for numerical reasons, ensure the levels sum to 1
            levfrac[i][0] = 1 - np.sum(levfrac[i][1:])
        elif np.sum(levfrac[i]) != 1:
            levfrac[i] = levfrac[i] / np.sum(levfrac[i])
            levfrac[i][0] = 1 - np.sum(levfrac[i][1:])

        zdepth = bathys[i - 1] - bathys[i]

        #some error checking
        if any(zdepth < 0):
            print '''WARNING: Sigma levels intersect, or islands are present
            that have not been properly masked. Sigma level''', i-1, ''' is
            intersecting sigma level''', i, '.'
        for j in range(levels[i]):
            dz[:, levnum] = zdepth * levfrac[i][j]
            levnum += 1

    return dz

def periodic(mesh, match, bcids=None, TOL=1e-12):
    """Turns a non-periodic mesh into a periodic one by connecting boundary
    edges.

    @param mesh (\c object) An instance of a mesh object

    @param match (\c lambda) A function that takes spatial two sets of spatial
        points as an input and returns true if these points are periodic, or
        false if not periodic.
        For example:
        match = lambda x1, x2: (x1[:, 0] == 1) & (x2[:, 0] == -1)
            & (x2[:, 1] == x2[:, 1])

    @param bcids (\c list) A list of boundary ids to include in the search for
        the periodic boundaries. If left blank, all boundaries are searched

    @param TOL (\c float) The tolerance to which the parameterized curves ids
        should match to be considered joined. That is, if |s_e1 - s_e2| < TOL
        we consider these two edges to be periodic.

    @retval mesh (\c object) The updated mesh data-structure with the periodic
        connectivity

    """
    if bcids == None:
        ids = mesh.ids_exterior_ed
    else:
        ids = np.zeros((mesh.ed2ed.shape[0]), dtype=bool)
        for bid in bcids:
            ids[-mesh.ed2ed[:, 2] - 1 == bid] = True

    #Find edge centroids
    vts = mesh.ed2ed[ids, 4:]
    x = np.array([coord[vts].mean(1) for coord in mesh.vert[:].T]).T

    #Keep track of the element ids
    elmids = ids.nonzero()[0]
    edids = np.arange(x.shape[0])

    #Keep track of the matches
    I = -np.ones((x.shape[0]), dtype=int)

    #Do a brute-force search to find matching edges
    activeids = np.ones((x.shape[0]), dtype=bool)
    i = 0
    while any(activeids):
        activeids[i] = False
        for xcomp, j in zip(x[activeids, :], edids[activeids]):
            if match(xcomp, x[i, :]) or match(x[i, :], xcomp):
                I[i] = elmids[j]
                I[j] = -elmids[i]-1 #This is mostly for debugging
                #I[i] = ids[j]
                #I[j] = ids[i]
                activeids[j] = False
                #print xcomp, x[i, :]
                break
        while i < activeids.shape[0] and not activeids[i]:
            i = i + 1

    #Find the ids of edges that will be connected now
    ids_ex = copy.copy(ids)
    ids_ex[ids] = (I >= 0)
    II = I[I>=0]

    #First, create a vertex map, which maps the vertex id's of the first edge
    #to that of it's matching edge
    vertmap = np.arange(mesh.vert.shape[0])
    v_keep = vertmap[mesh.ed2ed[ids_ex, 4:]]
    v_elim = vertmap[mesh.ed2ed[II, 4:]]
    #But we don't know the orientation, luckily, for that we can use the match
    #condition
    for vk in v_keep.T:
        for ve in v_elim.T:
            xk = mesh.vert[vk, :]
            xe = mesh.vert[ve, :]
            ids = match(xk.T, xe.T) | match(xe.T, xk.T)
            vertmap[ve[ids]] = vk[ids]

    mesh.vertmap = vertmap

    #Now modify the mesh arrays
    mesh.ed2ed[ids_ex, 2] = mesh.ed2ed[II, 0]
    mesh.ed2ed[ids_ex, 3] = mesh.ed2ed[II, 1]
    mesh.ed2ed = np.delete(mesh.ed2ed, II, axis=0)
    [tmp, I] = sortrows(mesh.ed2ed, 2, True)
    I = np.flipud(I)
    mesh.ed2ed = mesh.ed2ed[I, :]
    mesh.ed_type = np.delete(mesh.ed_type, II, axis=0)
    mesh.ed_type = mesh.ed_type[I]
    mesh.elm2elm[mesh.ed2ed[ids_ex, 0], mesh.ed2ed[ids_ex, 1]] \
        = mesh.ed2ed[ids_ex, 2]
    mesh.elm2elm[mesh.ed2ed[ids_ex, 2], mesh.ed2ed[ids_ex, 3]] \
        = mesh.ed2ed[ids_ex, 0]

    #Re-create the interior/exterior ids
    mesh.ids_interior_ed = (mesh.ed2ed[:, 2] >= 0).ravel()
    mesh.ids_exterior_ed = (mesh.ed2ed[:, 2] < 0).ravel()

    #Re-create the integer values
    mesh.n_ed = mesh.ed2ed.shape[0]
    mesh.n_ed_type = [np.sum(mesh.ed_type == tp) \
        for tp in np.unique(mesh.ed_type)]
    mesh.u_ed_type = \
        mesh.u_ed_type[np.array(np.unique(mesh.ed_type), dtype=int)]

    #Finally, return the newly periodic mesh
    return mesh

def distort(sol, amt=0.1):
    """Mesh distortion routine -- useful for creating curved meshes to test
    solver

    @param sol (\c object) Solution datastructure

    @param amt (\c float) Amount of distortion (default = 0.1)

    @retval sol (\c float) the updated sol data-structure

    @note ONLY distorts the first 2 dimension of a mesh
    """
    if sol.dim != 2:
        print "WARNING: msh.mk.distort only works for the first 2 dimensions" +\
            " of a mesh"

    TPI = np.pi

    sol.mesh.vert[:, 0] += -amt * \
        np.sin(TPI * sol.mesh.vert[:, 0]) * np.cos(TPI * sol.mesh.vert[:, 1])
    sol.mesh.vert[:, 1] += amt * \
        np.cos(TPI * sol.mesh.vert[:, 0]) * np.sin(TPI * sol.mesh.vert[:, 1])

    for i in range(len(sol.dgnodes)):
        sol.dgnodes[i][:, 0, :] += -amt * \
            np.sin(TPI * sol.dgnodes[i][:, 0, :]) *\
            np.cos(TPI * sol.dgnodes[i][:, 1, :])
        sol.dgnodes[i][:, 1, :] += amt * \
            np.cos(TPI * sol.dgnodes[i][:, 0, :]) *\
            np.sin(TPI * sol.dgnodes[i][:, 1, :])

    for i in range(len(sol.dgnodes_ed)):
        sol.dgnodes_ed[i][:, 0, :] += -amt * \
            np.sin(TPI * sol.dgnodes_ed[i][:, 0, :]) *\
            np.cos(TPI * sol.dgnodes_ed[i][:, 1, :])
        sol.dgnodes_ed[i][:, 1, :] += amt * \
            np.cos(TPI * sol.dgnodes_ed[i][:, 0, :]) *\
            np.sin(TPI * sol.dgnodes_ed[i][:, 1, :])

    #Now update the jacobians etc. in the sol data-structure
    sol.update_jacs()
    sol.update_cube_jacs()

    return sol
