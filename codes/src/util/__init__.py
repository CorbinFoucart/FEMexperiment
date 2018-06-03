""" @package src.util
A collection of utility functions for various tasks related to FEM and oceans.


@author: Matt Ueckermann

Created on Sat Feb 26 18:30:11 2011
"""
import numpy as np
import copy
from src.fem_base.master.mk_basis import mk_mapcoords


def in_prod(sol, field1, field2):
    """Take the inner product
    \f$\langle \phi_i \phi_j \rangle_\Omega = \int_\Omega \phi_i \phi_j d\Omega\f$ of
    two fields \f$\phi_i\f$ and \f$\phi_j\f$

    @param sol (\c object) The solution class for the fields

    @param field1 (\c float) List of numpy arrays giving the row fields.
                    For example field1 = sol.trace[:, 0:2, :]

    @param field2 (\c float) List of numpy arrays giving the column fields.
                    For example field1 = sol.trace[:, 1:3, :]

    @retval in_prod (\c float) Numpy array such that
                    in_prod(i, j) = \f$\langle \phi_i \phi_j \rangle_\Omega\f$
    """
    nfld_i = field1[0].shape[1]
    nfld_j = field2[0].shape[1]
    in_prod = np.zeros((nfld_i, nfld_j))
    for i in xrange(nfld_i): #Loop over the fields in field1
        for j in xrange(nfld_j): #Loop over the fields in field1
            #Loop over the different element types
            for k in xrange(len(sol.u_elm_type)):
                #Do the inner product using quadrature-free integration
                in_prod[i, j] += np.sum(np.dot(sol.master[k].M, \
                    sol.jac[k] * field1[k][:, i, :] * field2[k][:, j, :]))
            #in_prod[i, j] = in_prod[i, j] ** (0.5)
    return in_prod

#CM ADDED THIS FUNCTION FOR DEBUGGING 09/26/2013

def in_prod_bdry(sol, f, g, bcid2type):
    """
    Computes the integral of the field on the boundary
    Returns this value in F_bdry

    NOTE: Shape of f is f[nt][ndof, nc, ned]
    and shape of bcid2type has shape 1 x d=2 x nc
    """
    nfld_i = f[0].shape[1]
    nfld_j = g[0].shape[1]
    in_prod = np.zeros((nfld_i, nfld_j))
    for i in xrange(nfld_i):
        for j in xrange(nfld_j):
            #Loop over the different edge types
            for k in xrange(len(sol.u_ed_type)):
                # Find the boundary edges
                ids_ex = sol.ids_exterior_ed[k]
                ids1 = np.zeros(ids_ex.shape, dtype=bool)
                # Select the Dirichlet edge subset
                ids2 = (bcid2type[sol.bcid[k], 0] == 0).ravel()
                ids1[ids_ex] = ids2

                # M is (dof, dof)
                # J is (dof x n_ed)
                # f and g are (dof x n_ed)

#                print "IPB SHAPES:"
#                print sol.master_ed[k].M.shape
#                print sol.jac_ed[k][:, ids2].shape
#                print f[k][:, i, ids2].shape
#                print g[k][:, j, :].shape

                #Do the inner product using quadrature-free integration
                in_prod[i, j] += np.sum(np.dot(sol.master_ed[k].M, \
                    sol.jac_ed[k][:, ids2] * f[k][:, i, :] * g[k][:, j, :]))
    return in_prod

def norm_L2(sol, field1):
    """Calculates the L2 norm of a field
    @param sol (\c object) The solution class for the fields

    @param field1 (\c float) List of numpy arrays giving the fields. For example,
            field1 = sol.trace

    @retval norm_L2 (\c float) Numpy array of the L2 norms of the input fields.
        norm_L2 \f$=\sqrt{\int_\Omega \text{field1}(\mathbf x)^2 d\Omega}\f$
    """
    ip = in_prod(sol, field1, field1)
    norm_L2 = [[np.abs(n) ** (0.5) for n in ip1] for ip1 in ip]
    return norm_L2

#def norm_L1(sol, field):
#    for i in xrange(field1[0].shape[1]):
#        for k in xrange(len(sol.u_elm_type)):
#            norm_L1[k,i] = np.sum(np.dot(sol.master[k].M, \
#                sol.jac[k] * np.abs(field[k][:, i, :])))
#    return norm_L1

def get_extrapcoeffs(N, x=None):
    """Creates the coefficients used to extrapolate a value from a polynomial
    function.

    @param N (\c int):      The degree of the polynomial

    @param x (\c float):    (optional) the points where the polynomial is
                            evaluated. Default value x = [-1, 0, 1, ..., N-1]
                            x[0]: x value where extrapolation is desired
                            x[1 : N + 1 ]: x values of existing data

    OUTPUTS:
    @retval  D (\c float)  Extrapolation coefficients, such that:
                           F(x[0]) ~ D * F(x[1 : N + 1])

    @note Examples:
    To linearly extrapolate the known pairs (x,y) = (0, 0), (1,2) and find
    the y value of (-1,?) use:
    >>> D = extrapcoefs(1)
    >>> y = D * [0;2]

    To linearly extrapolate the known pairs (x,y) = (0, 0), (1,2) and find
    the y value of (3,?) use:
    >>> D = extrapcoefs(1,[3 0 1])
    >>> y = D * [0;2]

    To extrapolate from the known points (x,y) = (0,1), (1,2), (2,1) (3,2) to
    the point (-1,?) use:
    >>> D = extrapcoefs(3)
    >>> y = D * [1;2;1;2]
    >>> plot([-1:3],[y,1,2,1,2],'o')
    >>> hold on
    >>> plot([-1:0.01:3],polyval(polyfit(0:3, [1 2 1 2], 3),[-1:0.01:3]),'r')
    Notice the poor answer: CAREFUL WITH EXTRAPOLATION!

    @author Matt Ueckermann

    @note Creating the coefficients is pretty simple:
    Eventually we want to know the y-value at x*, which can be evaluated
    using:
    [x*^0 x*^1 x*^2 ... x*^n] * [a1;a2;...;an] = y*

    But for that we need to solve for coefficients of polynomial:
    [x1^0 x1^1 x1^2 ... x1^n;
     x2^0 x2^1 x2^2 ... x2^n;
     ...   ...  ...  ... ...;
     xn^0 xn^1 xn^2 ... xn^n;] * [a1;a2;...;an] = [y1;y2;...;yn]

     A = X^-1 * Y
    (X*) * A = (X*) * X^-1 * Y;
             = D * Y;
    """
    #Create the Vandermonde matrix:
    if x == None:
        x = range(-1, N + 1)

    V = np.fliplr(np.vander(x[1 :]))

    x = np.fliplr(np.vander(np.array(x[0]).repeat(N + 1)))
    D = np.dot(x[0, :], np.linalg.inv(V))
    return D


def calc_column_ave(sol, sol2D, phi):
    """This calculates the average pressure in a column of fluid for an
    extruded mesh. It then returns the average copied at each nodal point
    on a 2D field

    @param sol (\c object) The solution data-structure
    @param sol2D (\c object) The solution data-structure, but for sol.dim-1
    @param phi (\c float) A list of numpy arrays -- the field to be
        integrated

    @retval PHI (\c float) A list of numpy arrays -- phi integrated over a
        column of fluid, with the average copied to the 2D field nodal
        points. \f$\int_{\Omega_c} \phi d\Omega_c\f$, where \f$\Omega_c\f$ is
        a column of elements stacked on top of each other. Note, this is much
        different from \f$\int_z \phi dz\f$, which is the point-wise
        integrated value of phi.
    """
    isvec = False
    if len(phi[0].shape) == 4:
        phi = [p.reshape(p.shape[0], p.shape[1] * p.shape[2], p.shape3) \
            for p in phi]
        isvec = True
        vcomp = p.shape[1]
    #Create the elemental averaging operator
    EA = [np.dot(np.ones((1, M.M.shape[0])), M.M) for M in sol.master]
    shape = [phii.shape for phii in phi]
    #Calculate the per-element average
    phi_a = [np.zeros((s[1], s[2])) for s in shape]
    vol= [np.zeros((s[2])) for s in shape]
    for i in range(len(phi)):
        for j in range(phi[0].shape[1]):
            phi_a[i][j, :] = np.dot(EA[i], phi[i][:, j, :] * sol.jac[i])
        vol[i] = np.dot(EA[i], sol.jac[i])

    #Sum the elemntal pressures over each of the columns
    phi_a = [np.sum(pa.reshape(s[1], s[2] / sol.mesh.levels, \
        sol.mesh.levels), axis=2).reshape(1, s[1], s[2] / sol.mesh.levels)\
        for pa, s in zip(phi_a, shape)]
    #Calculate the column volumes
    vol = [np.sum(v.reshape(s[2] / sol.mesh.levels, sol.mesh.levels),\
        axis=1).reshape(1, s[2] / sol.mesh.levels) for v, s in zip(vol, shape)]


     #Divide out the volumes
    for i in range(len(phi)):
        for j in range(phi[0].shape[1]):
            phi_a[i][:, j, :] /= vol[i]

    #Turn this into a nice 2D field
    phi_a = [p.repeat(nb2d, 0) for p, nb2d in zip(phi_a, sol2D.nb)]

    #Reformat if required
    if isvec:
        phi_a = [p.reshape(p.shape[0], vcomp, p.shape[1] / vcomp, p.shape[2])\
            for p in phi_a]

    return phi_a

def calc_column_ave_ed(sol, sol2D, phi_ed, v_ed_ids=None):
    """This calculates the average pressure on the faces of a column of fluid
    for an extruded mesh. It then returns the average copied at each nodal
    point on a 2D field edge field

    @param sol (\c object) The solution data-structure
    @param sol2D (\c object) The solution data-structure, but for sol.dim-1
    @param phi_ed (\c float) A list of numpy arrays -- the field to be
        integrated
    @param v_ed_ids (\c bool) A list of numpy boolean arrays -- gives the
        edge ids of the vertical faces on the sides of fluid columns in the
        interior.

    @retval PHI (\c float) A list of numpy arrays -- phi integrated over a
        column of fluid, with the average copied to the 2D field nodal
        points. \f$\int_{\Omega_c} \phi d\Omega_c\f$, where \f$\Omega_c\f$ is
        a column of elements stacked on top of each other. Note, this is much
        different from \f$\int_z \phi dz\f$, which is the point-wise
        integrated value of phi.
    """
    #shorter names
    levels = sol.mesh.levels
    if v_ed_ids ==None:
        v_ed_ids = [np.zeros((nb_ed), dtype=bool) for nb_ed in sol.n_ed_type]
        #This is based on how edges are extruded in the mesh module
        #Only have to worry about square faces
        i = (sol.u_ed_type == 1).nonzero()[0][0]
        v_ed_ids[i][0:sol2D.ids_interior_ed[0].sum() * levels ]= True

    isvec = False
    if len(phi_ed[0].shape) == 4:
        phi_ed = [p.reshape(p.shape[0], p.shape[1] * p.shape[2], p.shape3) \
            for p in phi_ed]
        isvec = True
        vcomp = p.shape[1]
    #Create the elemental averaging operator
    EA = np.dot(np.ones((1, sol2D.master[i].M.shape[0])), sol2D.master[i].M)

    #Create a subset of phi_ed to work with
    phi_ed_sub = copy.copy(phi_ed[i][:, :, v_ed_ids[i]])
    shape = phi_ed_sub.shape

    #Calculate the per-element average
    phi_a = np.zeros((shape[1], shape[2]))
    vol = np.zeros((shape[2]))

    for j in range(phi_ed[0].shape[1]):
        phi_a[j, :] = np.dot(EA, phi_ed_sub[:, j, :]\
            * sol.jac_ed[i][:, v_ed_ids[i]])
    vol = np.dot(EA[i], sol.jac_ed[i][:, v_ed_ids[i]])

    #Sum the elemntal pressures over each of the columns
    phi_a = np.sum(phi_a.reshape(shape[1], shape[2] / sol.mesh.levels,\
        sol.mesh.levels), axis=2).reshape(\
        1, shape[1], shape[2] / sol.mesh.levels)

    #Calculate the column volumes
    vol = np.sum(vol.reshape(shape[2] / sol.mesh.levels, sol.mesh.levels),\
        axis=1).reshape(1, shape[2] / sol.mesh.levels)

    #Divide out the volumes
    for j in range(phi_ed[0].shape[1]):
        phi_a[:, j, :] /= vol

    #Turn this into a nice 2D field
    phi_a = phi_a.repeat(sol2D.nb_ed[i], 0)

    #Now create the output as if we changed every edge in a 2D field
    n_flds = phi_ed[0].shape[1]
    PHI = [np.zeros((nb, n_flds, ne)) \
        for nb, ne in zip(sol2D.nb_ed, sol2D.n_ed_type)]
    #Finally, populate PHI with phi_a
    PHI[i][:, :, sol2D.ids_interior_ed[0]] = phi_a

    #Reformat if required
    if isvec:
        phi_a = [p.reshape(p.shape[0], vcomp, p.shape[1] / vcomp, p.shape[2])\
            for p in phi_a]

    return PHI

#===============================================================================
# mk_nodes_vol
#===============================================================================
def mk_nodes_vol(sol, vol_pts, elmids=None, vert_vals=None, dgnodes=None):
    """ This creates the volume nodes in real-space so that a function
    defined in real space can be evaluated at the volume points as specified
    on the master element.

    @param sol (\c object) The solution object (sol.sol.Sol)

    @param vol_pts (\c float) List of numpy arrays giving the points in
                   the master element for each element type

    @param elmids (\c int) Numpy arrays of global element ids for which the
        nodal positions should be calculated

    @param vert_vals (\c float) Numpy array of the values of the vertices
        -- from which to create the volume nodes. If unspecified, will use
        sol.mesh.vert.

    @param dgnodes (\c float) For curved meshes, the volume points should be
        created from the dgnodes, not from the linear mesh. Therefore,
        use dgnodes = sol.dgnodes for curved meshes.

    @note To linearly interpolate a field from the element vertices,
        mk_nodes_vol can be called using vert_vals=field.

    @retval vol_nodes (\c float) List of numpy arrays which give the
                       volume nodes for each type of element
    """

    #Check some of the input options
    if dgnodes != None:
        #then just interpolate the dgnodes:
        return pt_eval(sol, dgnodes, vol_pts)
    #Otherwise, points will be created from the straight mesh

    if vert_vals == None:
        vert_vals = sol.mesh.vert[:]

    #Initialize the list for each type of element since different element
    #types will have different number of cubature points
    vol_nodes = [None] * len(sol.mesh.u_elm_type)

    #shorter names
    if elmids == None:
        elm = sol.mesh.elm[:]
        elm_type = sol.mesh.elm_type[:]
    else:
        elm = sol.mesh.elm[elmids]
        elm_type = sol.mesh.elm_type[elmids]

    #Have to make the transformation matrix
    vol_pts2xy = [mk_mapcoords(vol_pts[i], \
        sol.master[i].basis.vol_verts, \
        sol.mesh.u_elm_type[i], sol.dim) for i in range(len(sol.mesh.u_elm_type))]

    #For each type of element
    for i in range(len(sol.mesh.u_elm_type)):
        #Vertices for these elements
        vts = vert_vals[elm[elm_type == i, : sol.master[i].nv], \
            :sol.dim]

        #Some permutation of the vol_nodes array is needed to efficient
        #calculations, so it helps to record the shape of the vertices
        #array
        shape = vts.shape

        #Intermediate value, basically re-arrange the vertex matrix ((which
        #was created so that jacobians could easily be calculated)
        #To calculate the vol_node locations we need
        #vts.shape(n_verts, dim, n_elm)
        vol_nodes[i] = vts.swapaxes(0, 1).swapaxes(1, 2)

        #To actually create the vol-nodes, we use the conveniently-created
        #transformation matrix vol_pts2xy. First we re-shape the
        #vol-nodes so that we can perform a matrix multiply:
        #vol_nodes.shape = (n_verts, dim*n_elm), so basically
        #vol_nodes = [x1, y1, z1, x2, y2, z2 ... ], where x1 is a column
        #vector with the x-coordinates of the vertices of the first element
        vol_nodes[i] = np.dot(vol_pts2xy[i],\
                    vol_nodes[i].reshape(shape[1], shape[0] * shape[2]))

        #Finally, we reshape it again to have it in the format we want
        #vol_nodes.shape = (dim, n_vol_pts, n_elm)
        vol_nodes[i] = vol_nodes[i].reshape(\
            len(vol_pts[i]), shape[2], shape[0]).swapaxes(0,1)

    return vol_nodes


#===============================================================================
# mk_nodes_ed
#===============================================================================
def mk_nodes_ed(sol, ed_pts, edids=None, dgnodes_ed=None):
    """ This creates the edge nodes in real-space so that a function
    defined in real space can be evaluated at the edge points as specified
    on the edge master elements.

    @param sol (\c object) The solution object (sol.sol.Sol)

    @param ed_pts (\c float) List of numpy arrays giving the points in
                   the master element for each element type

    @param edids (\c int) Numpy arrays of global edge ids for which the
        nodal positions should be calculated

    @param dgnodes_ed (\c float) For curved meshes, the volume points should be
        created from the dgnodes_ed, not from the linear mesh. Therefore,
        use dgnodes = sol.dgnodes_ed for curved meshes.

    @retval ed_nodes (\c float) List of numpy arrays which give the
                       volume nodes for each type of element
    """
    #check the inputs
    if dgnodes_ed != None:
        return pt_eval_ed(sol, dgnodes_ed, ed_pts)

    #shorter name
    u_ed_type = sol.mesh.u_ed_type
    #Initialize the list for each type of element since different element
    #types will have different number of cubature points
    ed_nodes = [None] * len(u_ed_type)

    #We only need the vertices and not the edge-connectivity information
    if edids == None:
        ed = sol.mesh.ed2ed[:, 4:]
        #ed_type = np.array((sol.mesh.ed2ed[:, -1] >= 0).ravel(), dtype=int)
        ed_type = sol.mesh.ed_type
    else:
        ed = sol.mesh.ed2ed[edids, 4:]
        #ed_type = np.array((sol.mesh.ed2ed[edids, -1] >= 0).ravel(), dtype=int)
        ed_type = sol.mesh.ed_type[edids]

    #We need to create the mapping matrix
    ed_pts2xy = [mk_mapcoords(ed_pts[i], \
        sol.master_ed[i].basis.vol_verts, u_ed_type[i], sol.dim - 1) \
        for i in range(len(u_ed_type))]

    #For each type of edge
    for i in range(len(u_ed_type)):
        #Vertices for these elements
        vts = sol.mesh.vert[:][ed[ed_type == i, \
            : sol.master_ed[i].nv], : sol.dim]

        #Some permutation of the ed_nodes array is needed to efficient
        #calculations, so it helps to record the shape of the vertices
        #array
        shape = vts.shape

        #Intermediate value, basically re-arrange the vertex matrix ((which
        #was created so that jacobians could easily be calculated)
        #To calculate the cube_node locations we need
        #vts.shape(n_verts, dim, n_elm)
        ed_nodes[i] = vts.swapaxes(0, 1).swapaxes(1, 2)

        #To actually create the ed-nodes, we use the conveniently-created
        #transformation matrix ed_pts2xy. First we re-shape the
        #ed-nodes so that we can perform a matrix multiply:
        #ed_nodes.shape = (n_verts, dim*n_ed), so basically
        #ed_nodes = [x1, y1, z1, x2, y2, z2 ... ], where x1 is a column
        #vector with the x-coordinates of the vertices of the first element
        ed_nodes[i] = np.dot(ed_pts2xy[i],\
            ed_nodes[i].reshape(shape[1], shape[0] * shape[2]))

        #Finally, we reshape it again to have it in the format we want
        #ed_nodes.shape = (dim, n_ed_pts_ed, n_elm)
        ed_nodes[i] = ed_nodes[i].reshape(\
            len(ed_pts[i]), shape[2], shape[0]).swapaxes(0,1)

    return ed_nodes


#===============================================================================
# pt_eval
#===============================================================================
def pt_eval(sol, field, vol_pts, field_num=None, shap=None):
    """Evaluated the basis functions at points in real-space according to points
    defined on the master volume element.

    @param sol    (\c object) A src.sol.Sol object.

    @param field (\c float) A src.sol.Sol volume-field. One of trace, vecs or p.

    @param vol_pts (\c float) List numpy arrays containing the  points in the
                                master volume element where the field should be
                                evaluated. Each entry of the list gives the
                                points associated with that volume element type.
                                The order of the types are determined by the
                                order listed in sol.mesh.u_elm_type

    @param field_num (\c int) Single integer to determine which component should
                                be evaluated -- for multiple scalars, for example
                                field_num = 0 will only evaluate the first one.
                                By default, all values are evaluated.

    @param shap (\c float) Numpy array of the shape matrix. If this is specified
                            it is not neccessary to specify the volume points.

    @retval at_vol_pts (\c float) List of numpy arrays of the function evaluated
         at the points specified for each master volume element type.
         at_vol_pts[i].shape = (len(vol_pts[i]), field.shape[1], num_elm[i]),
         where num_elm[i] are the number of elements in the mesh of that type.
    """
    if shap is None:
        shap = [sol.master[i].basis.eval_at_pts(vol_pts[i])\
            for i in range(len(sol.master))]
    at_vol_pts = []

    for i in range(len(sol.mesh.u_elm_type)):
        #Record the shape of the array to easily reshape it for
        #computational-purposes
        arraysize = field[i].shape
        num_elms = sol.mesh.n_elm_type[i]
        num_pts = len(vol_pts[i])
        if len(arraysize) == 2: # 1 Scalar
            at_vol_pts.append((np.dot(shap[i], \
                field[i].reshape(arraysize[0], num_elms)))\
                .reshape(num_pts, 1, num_elms))
        if len(arraysize) == 3: #Scalars or 1 vector
            if field_num == None:
                at_vol_pts.append((np.dot(shap[i], \
                 field[i].\
                 reshape(arraysize[0], arraysize[1] * num_elms)))\
                 .reshape(num_pts, arraysize[1], num_elms))
            else:
                at_vol_pts.append((np.dot(shap[i], \
                 field[i][:, field_num, :].reshape(arraysize[0], num_elms)))\
                 .reshape(num_pts, 1, num_elms))
        elif len(arraysize) == 4: #Vectors
            if field_num == None:
                at_vol_pts.append(\
                 (np.dot(shap[i], field[i].\
                 reshape(arraysize[0], arraysize[1] * arraysize[2] * num_elms)))\
                 .reshape(num_pts, arraysize[1], arraysize[2], num_elms))
            else:
                at_vol_pts.append(\
                 (np.dot(shap[i], field[i][:, :, field_num, :].\
                 reshape(arraysize[0], arraysize[1] * num_elms)))\
                 .reshape(num_pts, arraysize[1], num_elms))

    return at_vol_pts

#===============================================================================
# pt_eval_ed
#===============================================================================
def pt_eval_ed(sol, field_ed, ed_pts, field_num=None, shap=None):
    """Evaluated the basis functions at points in real-space according to points
    defined on the master edge element.

    @param sol    (\c object) A sol.sol.Sol object.

    @param field_ed (\c float) A sol.sol.Sol edge-field. One of trace_ed,
                                vecs_ed, or p_ed

    @param ed_pts (\c float) List numpy arrays containing the  points in the
                                master edge element where the field should be
                                evaluated. Each entry of the list gives the
                                points associated with that edge element type.
                                'Type' is used loosely here, and the order in
                                the list is really defined by u_elm_type_ed

    @param field_num (\c int) Single integer to determine which component should
                                be evaluated -- for multiple scalars, for example
                                field_num = 0 will only evaluate the first one.
                                By default, all values are evaluated.

    @param shap (\c float) Numpy array of the shape matrix. If this is specified
                            it is not neccessary to specify the edge points.

    @retval at_ed_pts (\c float) List of numpy arrays of the function evaluated
         at the points specified for each master edge element type.
         at_ed_pts[i].shape = (len(ed_pts[i]), field_ed.shape[1], num_elm[i]),
         where num_elm[i] are the number of elements in the mesh of that type.
    """
    #Next we build the shap matrix for each type of edge element, if needed.
    if shap is None:
        shap = [sol.master_ed[i].basis.eval_at_pts(ed_pts[i]) \
            for i in range(len(sol.mesh.u_ed_type))]

    #Initialize the output fueld
    at_ed_pts = []

    #For each edge element type:
    for i in range(len(sol.mesh.u_ed_type)):
        #Record the shape of the array to easily reshape it for
        #computational-purposes
        arraysize = field_ed[i].shape
        num_elms = sol.mesh.n_ed_type[i]
        num_pts = len(ed_pts[i])
        if len(arraysize) == 2: # 1 Scalar
            at_ed_pts.append(\
                (np.dot(shap[i], field_ed[i].\
                reshape(arraysize[0], num_elms)))\
                .reshape(num_pts, 1, num_elms))
        if len(arraysize) == 3: #Scalars or 1 vector
            if field_num == None:
                at_ed_pts.append(\
                 (np.dot(shap[i], field_ed[i].\
                 reshape(arraysize[0], arraysize[1] * num_elms)))\
                 .reshape(num_pts, arraysize[1], num_elms))
            else:
                at_ed_pts.append(\
                 (np.dot(shap[i], field_ed[i][:, field_num, :].\
                 reshape(arraysize[0], num_elms)))\
                 .reshape(num_pts, 1, num_elms))

        elif len(arraysize) == 4: #Vectors
            if field_num == None:
                at_ed_pts.append(\
                 (np.dot(shap[i], field_ed[i].\
                 reshape(arraysize[0], arraysize[1] * arraysize[2] * num_elms)))\
                 .reshape(num_pts, arraysize[1], arraysize[2], num_elms))
            else:
                at_ed_pts.append(\
                 (np.dot(shap[i], field_ed[i][:, :, field_num, :].\
                 reshape(arraysize[0], arraysize[1]  * num_elms)))\
                 .reshape(num_pts, arraysize[1], num_elms))

    return at_ed_pts
