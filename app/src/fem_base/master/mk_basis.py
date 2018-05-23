"""@package src.master.mk_basis

Creates orthonormal modal polynomial basis on arbitrary master element of any
degree.

@note For most users the only function that will be needed is the Basis class

@par General usage example of Basis class is as follows:

@code
>>> from master import mk_basis as mkb
>>> basis = mkb.Basis(n=2, dim=2, element=0)
Working on Preliminaries:
Initializing...
Creating coefficients.
Calculating integrals.
Preliminaries finished, starting basis creating:
Creating basis 1 of 6
Creating basis 2 of 6
Creating basis 3 of 6
Creating basis 4 of 6
Creating basis 5 of 6
Creating basis 6 of 6
>>> print basis.aij
[[2**(1/2)/2, 0, 0, 0, 0, 0],
[1/2, 3/2, 0, 0, 0, 0],
[-3**(1/2)/2, -3**(1/2)/2, 3**(1/2), 0, 0, 0],
[-6**(1/2)/4, 6**(1/2)/2, 0, 5*6**(1/2)/4, 0, 0],
[-9*2**(1/2)/8, -3*2**(1/2), 9*2**(1/2)/4, -15*2**(1/2)/8, 15*2**(1/2)/4, 0],
[30**(1/2)/8, 30**(1/2)/2, -3*30**(1/2)/4, 30**(1/2)/8, -3*30**(1/2)/4,
  3*30**(1/2)/4]]
>>> print basis.pqr[0:6]
[[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]
@endcode

@verbatim In the above example, the first four bases are:
    1: 2**(1/2)2
    2: 1/2 + 3/2 * y
    3: -3**(1/2)/2 - 3**(1/2)/2 * y + 3**(1/2) * x
    4: -6**(1/2)/4 + 6**(1/2)/2 * y + 0 * x + 5*6**(1/2)/4 * y**2

Each basis, then, contains coefficients corresponding to rows in pqr. Each row
in pqr gives the degree of the monomial. That is, for row i we have the basis:
    x**pqr[i][0] * y**pqr[i][1]
@endverbatim

Created on Sat Feb 26 18:30:11 2011

@author: Matt Ueckermann
"""

from sympy import Rational, Symbol, integrate, sqrt
#from sympy import Rational, Symbol
#from sympy.integrals import integrate
#from mpmath import sqrt
from sympy.polys import Poly
#from sympy.polys.polytools import Poly
from sympy.matrices import Matrix
#from sympy.matrices.matrices import Matrix
from src.pyutil import parse, unique
from numpy import shape, array, vstack, hstack, mean, ones, arange, dot, rot90
from numpy import column_stack, sign
from numpy import sum as npsum
from numpy import mgrid, tile
from numpy.linalg import inv
import copy as cpy

import pdb

###############################################################################
def int_el_pqr(pqr=[], element=0, dim=None, X=1):
    """Calculates the integral of monomial basis for pre-defined elements

    @param    pqr    Matrix containing the powers of the monomials, \f$x^p y^q z^r\f$.  If empty, only el_verts and ids_ed are returned (in that order).
    @param    element   The number of the predefined element.
    @param    dim    The dimension of the problem. Usually not necessary to
                    specify this, but if you only want the master element
                    vertices and face id's, this is useful.
    @param    X         Maximum absolute value of master element coordinates.
                        (Default value = 1)

    @retval   int_pqr    Matrix of size (pqr_i x 1) containing the integral of
                        the monomial over the element
    @retval   int_pqr_ed Matrix of size (pqr_i x # ed) containing the integral of
                        the monomial over the element edges
    @retval   el_verts  List containing the coordinates of the vertices
                        Labeled counter-clockwise from bottom to top
    @retval   ids_ed  List containing the ids of the vertices
                        Labeled counter-clockwise from bottom to top. So, edge
                        2 has vertices el_verts[ed_ids[2]]

    @verbatim
    Exsiting elements:
    element =
        0: if dim = 1: A line [-X, X]
                    2: A triangle [[-X,-X], [X,-X], [X,X]]
                    3: A tetrahedral [[-X,-X,-X], [X,-X,-X], [X,X,-X], [X,-X,X]]
        1: if dim = 1: GOTO element 0, dim = 1
                    2: A square [[-X,-X], [X,-X], [X,X], [-X,X]]
                    3: A cube [[-X,-X,-X], [X,-X,-X], [X,X,-X], [-X,X,-X],
                               [-X,-X,X], [X,-X,X], [X,X,X], [-X,X,X]]
        2: if dim = 1: GOTO element 0, dim = 1
                    2: GOTO element 0, dim = 2
                    3: A prism [[-X,-X,-X], [X,-X,-X], [X,X,-X],
                               [-X,-X,X], [X,-X,X], [X,X,X]]


    @endverbatim


    @note for dim==3, triangular faces are numbered such that the x and y axes
          are defined as (v0-v2) and (v1-v0) respectively. For square faces,
          the x and y axes are defined by (v0-v3) and (v1-v0) respectively.

    @note for dim==2, the direction of the edge is defined by (v1-v0)

    @par If an element or dimension is chosen that is not coded, the function
         returns pqr[:] = -1

    @code
    #Example to get only the master element vertices and edge ids
    >>> verts, ids_ed = int_el_pqr(dims=2, element=0)

    #Example to get the integrals also
    >>> int_pqr, int_pqr_ed, vert, ids_ed = int_el_pqr(pqr, element=2)
    @endcode
    @see mk_pqr_coef

    @callgraph
    @callergraph
    @author Matt Ueckermann
    """

    pqr_i = len(pqr[:])
    if dim == None:
        dim = len(pqr[0][:])


    #Initializes int_pqr and int_pqr_ed to be the correct size
    #int_pqr contains volume integral over element for (x^p)(y^q)(z^r)
    int_pqr = [Rational(0) for i in range(pqr_i)]

    #list int_pqr_ed contains edge integrals over each of the elements
    #edges for the same monomial
    #int_pqr_ed = [Rational(0) for i in range(pqr_i)]

    #Loop over every monomial in the provided pqr matrix to calculate
    #the integrals

    #Set here the size of the maximal element coordinate value
    XMAX = X

    #The following is a large SWITCH statement, that determines which
    #elements integration rules are appropriate.

    #Use the first case always for 1D, and also for element==2 in 2D
    if ((element == 0) | (dim == 1) | ((element == 2) & (dim == 2))):
        #Define the sympy variables
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # For a line [-XMAX, XMAX] if dim == 1
        if dim == 1:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(x**pqr[i][0], (x, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX], [XMAX]]

            #Edge vertices
            ed_verts = [[0], [1]]


        # For a triangle [-XMAX,XMAX] [XMAX,-XMAX] [XMAX,XMAX] if dim == 2
        elif dim == 2:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(
                         integrate(x**pqr[i][0] * y**pqr[i][1], \
                         (y, -XMAX, x)), (x, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX, -XMAX], [XMAX, -XMAX], [XMAX, XMAX]]

            #Edge vertices. (Outward facing normals from right-hand rule)
            ed_verts = [[1, 2], \
                        [2, 0], \
                        [0, 1]]

        # For a tetrahedral if dim ==3
        elif dim == 3:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(integrate(
                         integrate(x**pqr[i][0] * y**pqr[i][1] * z**pqr[i][2],
                         (z, -XMAX, x-y-1)),
                         (y, -XMAX, x)),(x, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX, -XMAX, -XMAX], [XMAX, -XMAX, -XMAX], \
                        [ XMAX, XMAX, -XMAX],  [ XMAX,-XMAX, XMAX]]

            #Edge vertices. (Outward facing normals from right-hand rule)
            #Also, the first vertex is at the 90deg(in general) corner of the
            #edge, with v1-v0 defining the y-axis, and v0-v2 defining the z-axis
            ed_verts = [[1, 2, 3], \
                        [0, 3, 2], \
                        [1, 3, 0], \
                        [1, 0, 2]]

    elif (element == 1):
        #Define the sympy variables
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # For a square [-XMAX,-XMAX] [XMAX,-XMAX] [XMAX,XMAX] [-XMAX,XMAX]
        if dim == 2:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(
                     integrate(x**pqr[i][0] * y**pqr[i][1], (y, -XMAX, XMAX)),
                     (x, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX, -XMAX], [XMAX, -XMAX], [XMAX, XMAX], \
                        [-XMAX, XMAX]]

            #Edge vertices. (Outward facing normals from right-hand rule)
            ed_verts = [[1, 2], \
                        [2, 3], \
                        [3, 0], \
                        [0, 1]]
        #or a cube
        elif dim == 3:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(integrate(
                     integrate(x**pqr[i][0] * y**pqr[i][1] * z**pqr[i][2],
                     (y, -XMAX, XMAX)),
                     (x, -XMAX, XMAX)),(z, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX, -XMAX, -XMAX], [XMAX, -XMAX, -XMAX], \
                        [XMAX, XMAX, -XMAX], [-XMAX, XMAX, -XMAX],
                        [-XMAX, -XMAX, XMAX],[XMAX, -XMAX, XMAX], \
                        [XMAX, XMAX, XMAX], [-XMAX, XMAX, XMAX]]

            #Edge vertices. (Outward facing normals from right-hand rule)
            ed_verts = [[0, 3, 2, 1], \
                        [4, 5, 6, 7], \
                        [1, 2, 6, 5], \
                        [3, 7, 6, 2], \
                        [0, 4, 7, 3], \
                        [0, 1, 5, 4]]

    elif (element == 2):
        #Define the sympy variables
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # For a prism
        if dim == 3:
            for i in range(pqr_i):
                #Volume integral
                int_pqr[i] = integrate(integrate(
                         integrate(x**pqr[i][0] * y**pqr[i][1] * z**pqr[i][2],
                         (y, -XMAX, x)),
                         (x, -XMAX, XMAX)),(z, -XMAX, XMAX))

            #Volume vertices
            el_verts = [[-XMAX, -XMAX, -XMAX], [XMAX, -XMAX, -XMAX], \
                        [XMAX, XMAX, -XMAX],\
                        [-XMAX, -XMAX, XMAX], [XMAX, -XMAX, XMAX], \
                        [XMAX, XMAX, XMAX]]

            #Edge vertices. (Outward facing normals from right-hand rule)
            ed_verts = [[1, 0, 2], \
                        [4, 5, 3], \
                        [1, 2, 5, 4], \
                        [0, 3, 5, 2], \
                        [0, 1, 4, 3]]
    else:
        print('Unknown element/configuration requested. Element=', element, \
        ' with dimension=', dim, ' is not supported')
        el_verts = None
        ed_verts = None

    #Return outputs
    if pqr_i > 0:
        return int_pqr, array(el_verts), ed_verts
    else:
        return array(el_verts), ed_verts

## Documentation of this method
#
#  More details.
def get_nrml_jac(dim, element, degree=1):
    """ Returns the outward facing normal multiplied by the "differential edge
    jacobian" for all edges in the master element. This 'normal' is needed to do
    weak derivatives within the master-element. @see src.operators.master_grad

    @param dim (\c int) dimension of the problem
    @param element (\c int) The element type @see int_el_pqr
    @param degree (\c int) Optional parameter affecting the formatting of the
        returned normal. Basically, the normal is repeated based on the
        degree, dimension, and element type.
    @retval nrml (\c float) Numpy array of the normal on each edge number,
        repeated the required number of times. Basically,
        nrml.shape = (n_ed * repeats, dim), where repeats is dependent on the
        number of nodal points on an element edge.
    @note the output format matches the ed2elm space. That is, these normals
        correspond to the format of sol.get_ed2elm_array
    """

    if dim == 1:
        return array([[-1], [1]])

    if dim == 2:
        repeats = degree + 1
        if element == 0: #triangle
            nrml = array([[1, 0], [-1, 1], [0, -1]])
        elif element == 1: #square
            nrml = array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        return nrml.repeat(repeats, 0)

    if dim == 3:
        if element == 0: #tet
            #Only 1 triangle is non-standard in this case
            #x1 = array([1, -1, 1])
            #x2 = array([-1, -1, -1])
            #x3 = array([1, 1, -1])
            #a = cross(x1-x2, x3-x2)
            #jac = sqrt(dot(a, a))/2 / 2 #jacobian of standard triangle is 2
            # = 4 sqrt(3)
            nrml = array([[-1, 0, 0], [-2, 2, 2], [0, -1, 0], [0, 0, -1]])
            repeats = numbase(degree, dim-1)
            return nrml.repeat(repeats, 0)

        if element == 1: #square
            nrml = array([[0, 0, -1], [0, 0, 1],\
                [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
            repeats = (degree + 1) ** 2
            return nrml.repeat(repeats, 0)

        if element == 2: #prism
            nrml = array([[0, 0, -1], [0, 0, 1], \
                [1, 0, 0], [-1, 1, 0], [0, -1, 0]])
            repeats1 = numbase(degree, dim-1)
            repeats2 = (degree + 1) ** 2
            return vstack((nrml[:2, :].repeat(repeats1, 0),\
                nrml[2:, :].repeat(repeats2, 0)))


###############################################################################
def uniformlocalpts(basis, refine_it=None):
    """Creates uniform points locally on the elements as defined by
    int_el_pqr. This function is NOT very general, and creates nodes for the
    elements as shown in the next figure.

    @param basis A basis object (needs to have the following fields)
           basis.n -- order of basis
           basis.element -- element type
           basis.dim -- dimension
           basis.vol_verts -- vertices in the volume of the basis
    @param refine_it The order to which the points should be created. Note
                this can also be a list, where the second entry in the list
                gives the order for the z-coordinate direction. That is, you
                can have a different order in z than in x-y.

    @retval plocal (\c float) Numpy array of uniformly spaced points
                    [n_elmpts x dim]
    tlocal (\c int) Local triangulation matrix
    (TODO: NOT IMPLEMENTED)

    @retval flocal (\c int) List of ids that record pick out the points on the
                            edges of an element [n_edges x n_edpts]

    @author Matt Ueckermann
    """
    if refine_it == None:
        n = basis.n
    else:
        if type(refine_it) is int:
            n = refine_it
        elif type(refine_it) is list:
            n = refine_it[0]

    if n == 0:
        plocal = mean(basis.vol_verts, 0)
        tlocal = [0, 1, 2]
        flocal = [[]]#basis.ids_ed
        return plocal, flocal
    else:
        u = mgrid[0:n + 1]
        u = u / float(n) * 2. - 1
        plocal = 0
        if basis.dim == 1:
            plocal = u
            tlocal = [[i, i+1] for i in range(len(u)-1)]
            flocal = [[0],[len(u)]]
            return plocal, flocal
        else:
            flocal =[[] for i in range(3 + (basis.element == 1))]
            #First create the 2D main face
            for i in range(len(u)):
                if basis.element == 1:
                    j = 0 #For squares we basically extrude accross
                else:
                    j = i #For triangles, we knock off one point at every
                          #new level
                #Points to be added
                newpts = vstack((u[j:], tile(u[i], len(u) - j))).transpose()
                #First time through the loop, we have initialize plocal
                if i == 0:
                    plocal = newpts

                else:
                    plocal = vstack((plocal, newpts))
                #Now build the face-id matrices (for points that are only on the
                #element faces
                if basis.element == 1:
                    flocal[3] = range(0, len(u))
                    flocal[1] = range(len(plocal) -len(u), len(plocal))
                    flocal[0].append(len(plocal) - 1)
                    flocal[2].append(len(plocal) - len(newpts))
                else:
                    flocal[0].append(len(plocal) - 1)
                    flocal[1].append(len(plocal) - len(newpts))
                    flocal[2] = range(0, len(u))

        #Now, if dim == 2, we're done otherwise, we have to continue
        if basis.dim == 3:
            #If the z-direction needs to be lower order, we handle that case
            #NOTE, it does not work for tets, only prisms and quads
            if type(refine_it) == list:
                if (len(refine_it) > 1) and (basis.element is not 0):
                    u = mgrid[0:refine_it[1] + 1]
                    u = u / float(refine_it[1]) * 2. - 1

            #Need re-organize flocal, and accomodate the extra faces
            flocal.append([])
            if basis.element == 0:
                pass #nothing to do!
            elif basis.element == 1:
                flocal.append([])
                order = [4, 5, 0, 1, 2, 3]
                flocal = [flocal[i] for i in order]
            elif basis.element == 2:
                flocal.append([])
                order = [3, 4, 0, 1, 2]
                flocal = [flocal[i] for i in order]

            #These are the base 2D nodes that will be extruded to 3D
            uu = plocal
            #The base 2D nodes are restricted for tets, ids keeps track of
            #which nodes in uu will be kept at the next level
            ids = array([True for i in range(len(uu))])
            newpts = [0] * len(u)
            for i in range(len(u)):
                #Note, the new points at each level only consists of the
                #non-false-flagged points (in ids) [only relevant for tets]
                newpts = column_stack((uu[ids], tile(u[i], npsum(ids))))
                if i == 0:
                    plocal = newpts
                else:
                    plocal = vstack((plocal, newpts))

                #because we've already added parts of the face when doing the
                #2D operations, we start only after the second level
                if i < len(u) - 1:
                    if basis.element == 0:
                        flocal[3] = range(0, len(uu))
                        addids = array(range(\
                                   npsum(ids) - 1, npsum(ids) - len(u) + i, -1))
                        flocal[0].extend(list( \
                                    flocal[0][-len(u) + i:-1] + addids))
                        flocal[1].extend(list( \
                                    flocal[1][-len(u) + i:-1] + addids + 1))
                        flocal[2].extend(list( \
                                    flocal[2][-len(u) + i + 1:] + addids[0]))

                    elif basis.element == 1:
                        flocal[0] = range(len(uu))
                        flocal[1] = range(len(plocal), len(uu) + len(plocal))
                        flocal[2].extend(list(\
                                    flocal[2][-len(u):] + npsum(ids)))
                        flocal[3].extend(list(\
                                    flocal[3][-len(u):] + npsum(ids)))
                        flocal[4].extend(list(\
                                    flocal[4][-len(u):] + npsum(ids)))
                        flocal[5].extend(list(\
                                    flocal[5][-len(u):] + npsum(ids)))

                    elif basis.element == 2:
                        flocal[0] = range(len(uu))
                        flocal[1] = range(len(plocal), len(uu) + len(plocal))
                        flocal[2].extend(list(\
                                    flocal[2][-len(u):] + npsum(ids)))
                        flocal[3].extend(list(\
                                    flocal[3][-len(u):] + npsum(ids)))
                        flocal[4].extend(list(\
                                    flocal[4][-len(u):] + npsum(ids)))

                if basis.element == 0:
                    #For tets, we knock off points at every
                    #new level
                    #This is very dense code. For tets, the bottom face 2 is
                    #the one that starts losing points, so basically we're
                    #marching in from that face. The ids work out as you see
                    #below, but it's not too easy to tell from the code what's
                    #going on.
                    falsify = flocal[1][0:len(u) - i]
                    falsify = [fls + i for fls in falsify]
                    ids[falsify] = False

    #Make sure flocal has unique points
    flocal = [unique(fl).tolist() for fl in flocal]

    #The problem now is that the orientation of flocal does not match that of
    #the edge numbering -- so this has to be fixed.
    #First, create the matrix of all possible edge orientations:
    if basis.dim == 2:
        ed_type = [0, 0, 0, 0]
    elif basis.dim == 3:
        if basis.element == 0:
            ed_type = [0, 0, 0, 0]
        elif basis.element == 1:
            ed_type = [1, 1, 1, 1, 1, 1]
        elif basis.element == 2:
            ed_type = [0, 0, 1, 1, 1]

    orient = mk_ed_orient(flocal, ed_type, n, basis.dim)

    #Now we just to figure out what orientation each edge has, and fix it.
    for i in range(len(flocal)):
        tru_ed_pts = basis.ed_verts[i]

        flocstart = plocal[flocal[i][0]]
        #Grab the corner point
        flocnext = plocal[flocal[i][n]]

        ids = [tru_ed_pts.tolist().index(flocstart.tolist()),  \
            tru_ed_pts.tolist().index(flocnext.tolist())]
        orient_num = get_orient_num(ids, basis.dim, one_num=True)
        flocal[i] = orient[i][orient_num]
    return plocal, flocal

###############################################################################
def numbase(n, dim):
    """Calculates the number of basis in a complete degree 'n' polynomial basis
    in 'dim' dimensions

    @param    n    The degree of the polynomial
    @param    dim  The dimension of the basis
    @retval   nb   The number of basis in a complete degree 'n' polynomial basis
                   in 'dim' dimensions.
    @note    Uses nb = 1,    for i in range(dim):
        nb = nb * (i + 1 + n) / (i + 1)
    @author Matt Ueckermann
    """
    nb = 1
    for i in range(dim):
        nb = nb * (i + 1 + n) / (i + 1)
    return int(nb)

###############################################################################
def polymul(a1, a2, pqr):
    """Function for multiplying to monomials together using the
    data-structures in this module (mk_basis)

        @param    a1    Coefficients of first polynomial to multiply
        @param    a2    Coefficients of second polynomial to multiply
        @param    pqr   Matrix containing the powers of the monomials,
                         \f$x^p y^q z^r\f$
        @retval   b     Coefficients of multiplied polynomial.

        @note  a1, a2, and b are 1 colum vectors of coefficients. pqr is a
               matrix such that each row holds the monomial powers that has the
               corresponding coefficient in the same row as a1

        @verbatim
        Basically this function sums the appropriate lines in pqr, since
        a1(x^P1 * y^q1 * z^r1) * a2(x^P2 * y^q2 * z^r2)
           = a1 * a2 * (x^P * y^q * z ^r), where P = P1 + P2, etc.
        It then finds the line in pqr corresponding to the new powers (PQR)
        and places that value in the appropriate row of b
        @endverbatim

        @see polyder
        @author Matt Ueckermann
    """
    #Get dimensions of structurs
    np1 = len(a1)
    np2 = len(a2)
    pqr_i = len(pqr)

    #Find the maximum degree of the polynomials
    Np1 = npsum(pqr[np1 - 1])
    Np2 = npsum(pqr[np2 - 1])

    nb = numbase(Np1 + Np2, len(pqr[0]))

    #Error checking
    if (nb > pqr_i):
        print("polymul cannot multiply these polynomials together because", \
        " the matrix defining the polynomials, pqr, only contains ", pqr_i, \
        " entries, but requires ", nb, " entries.")
        return -1

    # Initialize b
    b = [Rational(0)] * (nb)

    for i in range(np1):
        for j in range(np2):
            # when you multiply a1(x^p1 * y^q1 * z^r1) with
            # a2(x^p2 * y^q2 * z^r2) you get
            # a1 * a2 * [x^(p1+p2) * y^(q1+q2) * z^(r1+r2)]

            #Do a search to find where to add the multiplied coefficients
            #This monomial corresponds with which column in pqr?
            pq = pqr[i][:]
            for k in range(len(pq)):
                pq[k] = pq[k] + pqr[j][k]

            ids = pqr.index(pq)

            #Therefore that row needs to have the coefficient
            b[ids] = b[ids] + a1[i] * a2[j]
    return b

###############################################################################
def polyder(a, pqr, der_dim):
    """ Function for taking derivatives of polynomials using the
        data-structures in this module (mk_basis)

        @param    a        Coefficients of 1 polynomial to take derivative
        @param    pqr      Matrix containing the powers of the monomials, \f$x^p y^q z^r\f$
        @param    der_dim  Dimension along which to take the derivative
                            [0 --> x], [1 --> y], [2 --> z], etc.
        @return   Coefficients of the polynomial after taking the derivative
                  Returns (-1) if there is an error.

        @note  a, is a 1 colum vectors of coefficients. pqr is a
               matrix such that each row holds the monomial powers that has the
               corresponding coefficient in the same row as a

        @par
        This function follows simple derivative rules
        \f{eqnarray*}{
            \frac{\partial}{\partial x_i}(x_1^{a_1}x_2^{a_2}\ldots
                x_i^{a_i}\ldots x_n^{a_n}) = a_i (x_1^{a_1}x_2^{a_2}\ldots
                x_i^{a_i-1}\ldots x_n^{a_n})
        \f}

        @see polymul
        @author Matt Ueckermann
    """

    #Error checking
    if (der_dim + 1 > len(pqr[0])):
        print("polyder cannot take derivative with respect to", \
        " component 1 + ", der_dim, "since we have a", len(pqr[0]), \
        "component (or dimension) polynomial.")
        return -1

    if (len(a) > len(pqr)):
        print("The 'pqr' matrix has too few entries.", \
        "The 'pqr' matrix only has",\
        len(pqr), "entries whereas it needs at least", len(a), "entries.")
        return -1

    # Initialize the output vector b
    b = [Rational(0) for i in range(len(a))]

    for i in range(len(a)):
        #If the current monomial is constant in der_dim, then the derivative is
        #zero, so we do nothing.
        if (pqr[i][der_dim] > 0):
            pq = pqr[i][:]
            #Take the derivative of that monomial
            pq[der_dim] = pq[der_dim] - 1
            #Find out which row that contribution should be added to.
            ids = pqr.index(pq)
            #Modify the coefficient appropriately
            b[ids] = b[ids] + a[i] * pqr[i][der_dim]

    return b

###############################################################################
def polyder_mat(a, pqr, der_dim):
    """ Function for taking derivatives of polynomials using the
        data-structures in this module (mk_basis)

        @param    a        Coefficients of many polynomials to take derivatives
        @param    pqr      Matrix containing the powers of the monomials, \f$x^p y^q z^r\f$
        @param    der_dim  Dimension along which to take the derivative
                            [0 --> x], [1 --> y], [2 --> z], etc.
        @return   Coefficients of the polynomial after taking the derivative
                  Returns (-1) if there is an error.

        @note  a, is at least a 2 colum vector of coefficients. pqr is a
               matrix such that each row holds the monomial powers that has the
               corresponding coefficient in the same row as a

        @par
        This function follows simple derivative rules
        \f{eqnarray*}{
            \frac{\partial}{\partial x_i}(x_1^{a_1}x_2^{a_2}\ldots
                x_i^{a_i}\ldots x_n^{a_n}) = a_i (x_1^{a_1}x_2^{a_2}\ldots
                x_i^{a_i-1}\ldots x_n^{a_n})
        \f}

        @see polymul, polyder
        @author Matt Ueckermann
    """

    #Error checking
    if (der_dim + 1 > len(pqr[0])):
        print("polyder_mat cannot take derivative with respect to", \
        " component 1 + ", der_dim, "since we have a", len(pqr[0]), \
        "component (or dimension) polynomial.")
        return -1

    if (len(a[0]) > len(pqr)):
        print("The 'pqr' matrix has too few entries.", \
        "The 'pqr' matrix only has",\
        len(pqr), "entries whereas it needs at least", len(a), "entries.")
        return -1

    # Initialize the output vector b
    b = [[Rational(0) for i in range(len(a[0]))] for j in range(len(a))]

    for i in range(len(a)):
        b[i] = polyder(a[i], pqr, der_dim)

    return b

###############################################################################
def inprod(a1, a2, pqr, int_pqr):
    """Function for taking the innner-product between two polynomials

    @param  a1      Coefficients of first polynomial to multiply
    @param  a2      Coefficients of second polynomial to multiply
    @param  pqr     Matrix containing the powers of the monomials \f$x^p y^q z^r\f$
    @param  int_pqr Matrix of pre-computed integrals of monomials over the element \f$\Omega\f$

    @retval coeff   Coefficients of multiplied polynomial.

        @note a1 and a2 may only be 1d arrays where a1[:] gives the
               coefficients of the first basis, and a2[:] gives the
               coefficients of the second basis. In this case, len(coeff) = 1
        @par  pqr is a matrix such that each row holds the monomial powers
              that has the corresponding coefficient in the same row as a1/a2

        @par The inner product is defined as:
        \f{eqnarray*}{
            \mathbf b &=& \int_\Omega \mathbf a_1 \cdot \mathbf a_2 d\Omega \\
                      &=& \sum_i \sum_j \int_\Omega a_{1,i} a_{2,i} d\Omega,
        \f}
        where \f$\mathbf a_1 = [a_{1,0},a_{1,1},a_{1,2},...,a_{1,n}]\f$

        @see inprod_mat

        @author Matt Ueckermann
    """

    coeff = Rational(0)
    b = polymul(a1[:], a2[:], pqr)
    for ii in range(len(b)):
        coeff = coeff + b[ii] * int_pqr[ii]

    return coeff

 ###############################################################################
def inprod_mat(a1, a2, pqr, int_pqr):
    """Function for taking the innner-product between many polynomials

    @param  a1      Matrix of coefficients of first polynomial to multiply
    @param  a2      Matrix of coefficients of second polynomial to multiply
    @param  pqr     Matrix containing the powers of the monomials \f$x^p y^q z^r\f$
    @param  int_pqr Matrix of pre-computed integrals of monomials over the element \f$\Omega\f$

    @retval coeff   Coefficients of multiplied polynomial

    @note  a1 and a2 may be arrays where a[0][:] gives the coefficients of
    the first basis, and a[1][:] gives the coefficients of the
    second basis. In this case, coeff = [len(a1) x len(a2)]

    @verbatim IMPORTANT:
    If a1 and a2, both only contain 1 basis, it is better to use inprod.
    This function will return a list of a list: coeff = [[.]], where '.'
    is the inner product between the two basis.
    @endverbatim

    @par   pqr is a matrix such that each row holds the monomial powers
    that has the corresponding coefficient in the same row as a1/a2

    @par The inner product is defined as:
    \f{eqnarray*}{
    \mathbf b &=& \int_\Omega \mathbf a_1 \cdot \mathbf a_2 d\Omega \\
    &=& \sum_i \sum_j \int_\Omega a_{1,i} a_{2,i} d\Omega,
    \f}
    where \f$\mathbf a_1 = [a_{1,0},a_{1,1},a_{1,2},...,a_{1,n}]\f$

    @see inprod
    @author Matt Ueckermann
    """
    nb_a1 = len(a1)
    nb_a2 = len(a2)
    if len(shape(a1)) == 1:
        nb_a1 = 1
        a1 = [a1]
    if len(shape(a2)) == 1:
        nb_a2 = 1
        a2 = [a2]

    coeff = [[Rational(0) for j in range(nb_a2)] for i in range(nb_a1)]

    for i in range(nb_a1):
        for j in range(nb_a2):
            coeff[i][j] = inprod(a1[i], a2[j], pqr, int_pqr)

    return coeff

###############################################################################
def mk_coef_n(pqr, n, dim, row, col=0):
    """ Function for creating all monomials of a particular degree (n)

        This is a recusrive function, that can handle any dimension

        @param    pqr   Matrix containing the powers of the monomials, \f$x^p y^q z^r\f$. The pqr array is modified within this function.
        @param    n     The total degree of this monomial
        @param    dim   Dimension of this monomial
        @param    row   Starting row in pqr (needed for recursion, specified in
                        mk_pqr_coeff() function)
        @param    col   Starting column in pqr (needed for recursion, user
                        should not specify)
        @retval   num   The number of bases created.

        @note
        Every column in pqr represents a dimension, every row a monomial

        @author Matt Ueckermann

        @see mk_pqr_coeff
    """
    num = 0
    #If dimension is 1, there are no free choices, so assign the current row
    # and column with degree n
    if (dim <= 1):
        num = 1
        pqr[row][col] = n
    else :
    #If dimension is not 1, cycle through all the possible degrees for this
    # dimension
        for j in range(n + 1):
            #This dimension's basis has taken the value of 'j', now we need to
            #create the basis for the other dimensions such that the total
            # degree is n. That means the degree of the other dimensions need
            #to add up to n - j because (n-j) + j = n
            #Here we have the recursive call
            numloc = mk_coef_n(pqr, n - j, dim - 1, row, col + 1)

            #So, for every basis created, we need to assign 'j' to that row of
            #this basis.
            for k in range(numloc):
                pqr[row + k][col] = j

            #Advance the row number for this dimension
            row = row + numloc

            #Increase total number of bases
            num = num + numloc
    return num

###############################################################################
def mk_pqr_coeff(n, dim):
    """ Function for creating array of powers for monomials

        This function is the basic function used to create pqr datastructure.

        @param    n    Degree of polynomial
        @param    dim  Dimension of polyomial

        @retval   pqr  Matrix containing the powers of the monomials,
                         \f$x^p y^q z^r\f$.

        @see    mk_coef_n

        @author Matt Ueckermann
    """
    #if 0D, we're done
    if dim==0:
        return [[0]]

    #Figure out the number of bases
    nb = numbase(n, dim)
    #Initialize coefficient matrix with all zeros
    pqr = [[0 for i in range(dim)] for j in range(nb)]
    num = 0
    row = 0

    #For each maximal degree of i, create all bases of that order
    for i in range(n + 1):
        #Create all bases with degree i
        num = mk_coef_n(pqr, i, dim, row)
        row = row + num

    #Just a quick check for debugging reasons
    if row != nb:
        print(num, "is not equal to", nb)
    return pqr


###############################################################################
def xyz2uvw_TM(x, monoms_xyz, monoms_uvw=None):
    """ This function creates the transformations matrix used to evaluated
    a basis in a coordinate system x-y-z to a coordinate system u-v-w, where
    x = f(u,v,w)

    @param    x    A list defining the transformation. For example, x=[-w,-v,-u]
                   where u, v, w are all sympy.Symbol objects is a valid input.
                   u, v, and w do not need to be Symbol objects, but could be
                   strings x=['-u','-v'] for example.
                   @see sympy.polys.Poly for more information.
    @param    monoms_xyz (\c int) The list of monomials defining the polynomial
                                  basis in xyz coordinates.
    @param    monoms_uvw (\c int) The list of monomials defining the polynomial
                                  basis in uvw coordinates. This is generally
                                  not required as an input, since the monomial
                                  list will be the same in 3D-3D
                                  transformations, but doing a 3D-2D
                                  transformation, the list of 2D monomials is
                                  required.
    @retval   TM  The Transformation Matrix.
              TM * F(monoms_xyz) = f(monomz_uvw)

    @note For example:
    \f{align*}{
        \theta &= Ax + Bx^2 ~~\text{The basis}\\
        x &= a + bu + cv + du^2 ~~\text{The transformation}\\
        x^2 & = e + fu + gv + hu^2 \\
        \mathbf {TM} &= \begin{bmatrix}
             a & e & \cdots  \\
             b & f & \cdots  \\
             c & g & \cdots  \\
             d & h & \cdots
         \end{bmatrix} \\
        \begin{bmatrix}
        1 \\
        u \\
        v \\
        u^2
        \end{bmatrix} &= \mathbf {TM}
        \begin{bmatrix}
        Ax \\
        Bx^2 \\
        \vdots
        \end{bmatrix}
    \f}

    @see elm2edge
    @see sympy.polys.Poly

    @author Matt Ueckermann
    """
    #Parse inputs
    if monoms_uvw == None:
        monoms_uvw = monoms_xyz

    #Shorter variable name
    dim1 = len(x)
    dim2 = len(monoms_uvw[0])
    nb_xyz = len(monoms_xyz)
    nb_uvw = len(monoms_uvw)

    u = Symbol('u')

    if dim1 > 1:
        v = Symbol('v')
    if dim1 > 2:
        w = Symbol('w')

    #handle the case of x is a list of strings
    if type(x[0]) == str:
        if dim1 == 1:
            x[0] = Poly(x[0], u)
        elif dim1 == 2:
            x[0] = Poly(x[0], u, v)
            x[1] = Poly(x[1], u, v)
        elif dim1 == 3:
            x[0] = Poly(x[0], u, v, w)
            x[1] = Poly(x[1], u, v, w)
            x[2] = Poly(x[2], u, v, w)

    #Initialize the mapping or transform matrix
    TM = Matrix([[0] * nb_xyz] * nb_uvw)

    #Now create the mapping or transform matrix
    for i in range(nb_xyz):
        #Create the on-edge monomial for this volume monomial
        expression = 1
        for j in range(dim1):
            expression = expression * x[j] ** monoms_xyz[i][j]
        #Then using Sympy's polynomial object, create polynomial
        if dim2 > 2:
            p = Poly(expression, u, v, w)
        elif dim2 > 1:
            p = Poly(expression, u, v)
        else:
            p = Poly(expression, u)

        #Finally, find the coefficients of the new monomials, and add it to the
        #appropriate row/column in the transformation matrix
        for j in range(len(p.coeffs)):
            ids = monoms_uvw.index(list(p.monoms[j]))
            TM[ids, i] = TM[ids, i] + p.coeffs[j]

    return TM

###############################################################################
def elm2edge(basis, ed, basis_e=None):
    """ This function evaluates a basis in dimension = dim on a specified edge
    of the element, and returns a new basis in dimension=(dim-1) that lives on
    the specified edge (in u-v coordinates).

    @param    basis    An instance of the Basis class
    @param    ed (\c int) Edge (number) on which to evaluate the basis
    @param    basis_e  (OPTIONAL) The basis onto which it should be projected.
                        By default we project onto a monomial basis.
    @retval   coeffs_e  The new edge basis with coefficients corresponding to
                        the monomials in basis_e.monom

    @see Basis
    @see xyz2uvw_TM

    @note   The 'u' direction is defined by vert[0] - vert[-1] {or vert(end)
            for MATLAB users}, and 'v' by vert[1] - vert[0], where vert is the
            array giving the vertices of the edge.
    @par    Some of this function is repeated in elm_xyz2ed_uv below

    @author Matt Ueckermann
    """
    #Shorter variable name
    dim = basis.dim
    n = basis.n
    nb = basis.nb

    #Define shorter names for the vertices:
    vert = basis.vol_verts[basis.ids_ed[ed]]

    #Define symbolic variables
    u = Symbol('u')

    if dim > 2:
        v = Symbol('v')

    #initialize basis_e if needed
    if basis_e == None:
        element = 0
        if len(vert == 4):
            element = 1
        basis_e = Basis(n, dim - 1, element)

    coeffs_e = [[Rational(0) for i in range(basis_e.nb)] \
                             for j in range(nb)]

    #Define the parametric coordinates
    #x-axis
    dv = vert[0] - vert[-1]

    #NOw get the middle vertex (still needs to be divided by 2)
    if dim == 3:
        mv = vert[1] + vert[-1]
    elif dim == 2:
        mv = vert[0] + vert[1]
    x = [None] * dim
    for i in range(dim):
        x[i] = mv[i] / 2 + dv[i] * u / 2
        if dim > 2:
            #y-axis (3D only)
            dv2 = vert[1] - vert[0]
            #mv2 = vert[1] + vert[0]
            x[i] = x[i] + dv2[i] * v / 2

    #So, x now creates the parametric coordinates, x[i] = f(u,v)
    #Create transformation matrix:
    TM = xyz2uvw_TM(x, basis.monoms[0:nb], basis_e.monoms[0:basis_e.nb])

    #Finally, do all the mappings for all the existing bases
    for i in range(nb):
        newcoeffs = TM * Matrix(basis.coeffs[i])
        for k in range(basis_e.nb):
            coeffs_e[i][k] = newcoeffs[k]

    return coeffs_e

def elm_xyz2ed_uv(basis, ed, n = None):
    """ This function creates the the uniformlocalpts evaluated in x-y-z to a
    coordinate system u-v existing on the edge of an element. This function is
    solely used for checking purposes!

    @param    basis    An instance of the Basis class
    @param    ed (\c int) Edge (number) on which to evaluate the basis
    @param    n (\c int) (OPTIONAL) number of points to create along each
              dimension. By default n = basis.n * 5.
    @retval   pts_uv_float The u-v coordinates of the transformed x-y-z
                            coordinates.

    @note This function repeats part of elm2edge function. The two should be
          changed together.

    @see elm2edge
    @see xyz2uvw_TM
    @see sympy.polys.Poly

    @author Matt Ueckermann
    """
    #Shorter variable name
    dim = basis.dim
    if n == None:
        n = basis.n * 5

    #Define shorter names for the vertices:
    vert = basis.vol_verts[basis.ids_ed[ed]]

    #Define symbolic variables
    u = Symbol('u')

    if dim > 2:
        v = Symbol('v')

    #Define the parametric coordinates
    #x-axis
    dv = vert[0] - vert[-1]
    if dim == 3:
        mv = vert[1] + vert[-1]
    elif dim == 2:
        mv = vert[0] + vert[1]

    x = [None] * dim
    for i in range(dim):
        x[i] = mv[i] / 2 + dv[i] * u / 2
        if dim > 2:
            #y-axis (3D only)
            dv2 = vert[1] - vert[0]
            #mv2 = vert[1] + vert[0]
            x[i] = x[i] + dv2[i] * v / 2

    monoms_xyz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    monoms_uv  = [[0, 0], [1, 0], [0, 1]]

    #Create transformation matrix:
    TM = xyz2uvw_TM(x, monoms_xyz ,monoms_uv )
    #We need to take the transpose because the columns of TM stores in the first
    #column a, b, c, where x = a +bu+cv
    TM = TM.transpose()

    #Make the xyz points
    pts3, f3 = uniformlocalpts(basis, n)
    pts = pts3[f3[ed], :]

    #In case the transformation is not unique, we need to find the sub-matrix
    #that IS unique (This is a little hacky...)
    ids = range(3)
    ids2S = 1
    ids2E = 3
    if TM.det() == 0:
        ids2S = 0
        ids2E = 2
        if TM[1:3,1:3].det() != 0:
            #remove x
            TM = TM[1:3,1:3]
            ids.pop(0)
        elif TM[0:2, 0:2].det() != 0:
            TM = TM[0:2, 0:2]
            ids.pop(2)
        else:
            TM.delRowCol(1,1)

    #Solve for the uv points in terms of the xyz points
    pts_uv = (TM.inv() * pts[:,ids].T).T[:, ids2S:ids2E]
    #Convert this to a nicer format: the sympy matrix format is not convenicnt.
    pts_uv_float = [[float(pts_uv[i, 0].evalf()), float(pts_uv[i, 1].evalf())] \
                    for i in range(pts_uv.shape[0])]

    return array(pts_uv_float)

###############################################################################

def mk_mapcoords(xi, v, elt_type, dim):
    """Creates the matrix of shape functions that maps points in the master
    element to points in the physical element

    @param[in] xi        Numpy array of points in the master element
    @param[in] v         Unused
    @param[in] elt_type  Element type
    @param[in] dim       Dimension of the element

    @pre Dimension of the element must be 1, 2, or 3

    @retval T Numpy array which is the transformation matrix such that T * V = Xpts, where V is a numpy array storing the vertices of the element in real space, V = [x1, y1, z1, x2, y2, z2, ... xn, yn, zn], with each of xi, yi, and zi column vectors of length len(x) = len(vol_verts), and Xpts gives xi in realspace, Xpts.shape = (len(xi), dim * num_elm)

    @note The mappings for the triangles and tetrahedrals are from the barycentric coordinates. Basically, solve for \f$\lambda_1\f$ etc. by using the master element vertices and the fact that \f$\sum \lambda_i = 1 \f$. The current implementation is very specific to how the master elements are defined.

    The mappings for quads and the z-coordinate for prisms is based on a linear nodal basis.

    @note This function is copied/pasted and modified in cal_jaco_code. Any changes here, should also require changes there.

    @note Triangles and tets are coded for general transforms. Prism are ONLY for the master element as defined in int_el_pqr, and squares/cubes ONLY for master elements with vertices of &amp; type (+/-1, +/-1, +/-1)

    @author Matt Ueckermann
    @author Chris Mirabito
    """


    assert(dim in [1, 2, 3])

    T = ones((len(xi), len(v)), dtype=float)

    if dim == 1:
        # LINE
        pass
    elif dim == 2 and elt_type == 0:
        # TRI
        pass
    elif dim == 2 and elt_type == 1:
        # QUAD
        pass
    elif dim == 3 and elt_type == 0:
        # TET
        pass
    elif dim == 3 and elt_type == 1:
        # BRICK
        pass
    elif dim == 3 and elt_type == 2:
        # WEDGE
        pass
    else:
        print("Bad element type / dimension combo")



    if elt_type == 0: #Tet, line, or triangle
        for j in range(len(xi)):
            if dim == 1:
                #assumed only two vertices:
                #CM  These correspond to N_a(xi) in TJR Hughes's book pp. 38--39
                #CM  with x^e(xi)=N_a(xi)x_a^e  (sum over a)
                #CM  N_a(xi)=1/2*(1+xi_a*xi)
                T[j, 0] = (1. + sign(v[0, 0]) * xi[j, 0]) / 2.
                T[j, 1] = (1. + sign(v[1, 0]) * xi[j, 0]) / 2.
            elif dim == 2:
                #Non-general version
                #T[j, 0] = (1. - xi[j, 0]) / 2.
                #T[j, 1] = (xi[j, 0] - xi[j, 1]) / 2.
                #T[j, 2] = (xi[j, 1] + 1.) / 2.
                #General version
                A = array([v[:, 0], v[:, 1], [1, 1, 1]], dtype='float') ## CF array needs to know what type
                T[j, :] = dot(inv(A), array([xi[j, 0], xi[j, 1], 1.]))

            elif dim == 3:
                #Non-general version
                #T[j, 0] = (1. - xi[j, 0]) / 2.
                #T[j, 1] = (-1. + xi[j, 0] - xi[j, 1] - xi[j, 2]) / 2.
                #T[j, 2] = (1. + xi[j, 1]) / 2.
                #T[j, 3] = (1. + xi[j, 2]) / 2.
                #General version
                A = array([v[:, 0], v[:, 1], v[:, 2], [1, 1, 1, 1]])
                T[j, :] = dot(inv(A), \
                    array([xi[j, 0], xi[j, 1], xi[j, 2], 1.]))

    elif elt_type == 1: #square or cube
        for j in range(len(xi)):
            for i in range(len(v)):
                for k in range(dim):
                    #CM  Tensor product of 1D N_a functions
                    T[j, i] = T[j, i] * (1 + sign(v[i, k]) * xi[j, k]) / 2.

    elif elt_type == 2: #prism
        for j in range(len(xi)):
            #Non-general version
            T[j, 0] = (1. - xi[j, 0]) * (1. - xi[j, 2]) / 4.
            T[j, 1] = (xi[j, 0] - xi[j, 1]) * (1. - xi[j, 2])  / 4.
            T[j, 2] = (xi[j, 1] + 1) * (1. - xi[j, 2])  / 4.
            T[j, 3] = (1. - xi[j, 0]) * (1. + xi[j, 2]) / 4.
            T[j, 4] = (xi[j, 0] - xi[j, 1]) * (1. + xi[j, 2])  / 4.
            T[j, 5] = (xi[j, 1] + 1.) * (1. + xi[j, 2])  / 4.
    return T

###############################################################################
def mk_ed_orient(ed_ids, ed_type, n, dim):
    """Creates the permutation arrays that allows one to match the points on
    faces that are rotated relative to each other.

    @param ed_ids (\c int) List of edges with the ids that define which vertices
                            make up the edge. For example, ed_ids[0] is a list
                            of indices, e.g. [0, 1, 2, 3].
                            len(ed_ids) = n_edges
                            len(ed_ids[i]) = n_point_in_edge_i

    @param ed_type (\c int) List of the edge types. len(ed_type) == len(ed_ids)

    @param n (\c int) The order of the basis

    @param dim (\c int) The dimension of the problem. i.e. a 3D problem with 2D
                        eges has dim == 3

    @retval (\c int) The orientation matrix -- it retuns the edge ids oriented
                        in all the possible combinations (As given by
                        src.master.mk_basis.get_orient_num)

    @see src.master.mk_basis.get_orient_num)
    """
    #First we grab the list that gives the ids of each face on each edge


    #Next we generalize n a little bit
    nn = n
    if type(n) == list:
        if len(n) < 2:
            nn = [n[0], n[0]]
    if type(nn) == int:
        nn = [nn, nn]

    NN = nn[0]

    orient = []
    if dim == 1 or dim == 0:
        orient = [[[0]]]
    elif dim == 2:
        for ed in ed_ids:
            e_orient = []
            e_orient.append(ed)
            e_orient.append(ed[::-1])
            orient.append(e_orient)
    else:
        for i in range(len(ed_ids)):
            e_orient = []
            e_type = ed_type[i]
            if e_type == 0: #Triangles
                #I couldn't find an elegant way to do this, so on a case
                #by case basis, here we go:
                #Type where opposing face had vertices 012 compared to 012
                e_orient.append(ed_ids[i])
                #Type where opposing face had vertices 120 compared to 012
                e_o = []
                start = len(ed_ids[i]) - 1
                for m in range(NN + 1):
                    prev = start
                    e_o.append(ed_ids[i][start])
                    for n in range(NN - m):
                        e_o.append(ed_ids[i][prev - (2 + n + m)])
                        prev += - (2 + n + m)
                    start -= (1 + m)
                e_orient.append(e_o)
                #Type where opposing face had vertices 201 compared to 012
                e_o = []
                start = NN
                for m in range(NN + 1):
                    prev = start
                    e_o.append(ed_ids[i][start])
                    for n in range(NN - m):
                        e_o.append(ed_ids[i][prev + NN - n])
                        prev += NN - n
                    start -= 1
                e_orient.append(e_o)
                #now a spacer (so that the orientation numbers for triangles
                #and squares can be the same...):
                e_orient.append([])
                #Type where opposing face had vertices 021 compared to 012
                e_o = []
                start = 0
                for m in range(NN + 1):
                    prev = start
                    e_o.append(ed_ids[i][start])
                    for n in range(NN - m):
                        e_o.append(ed_ids[i][prev + (NN + 1 - n)])
                        prev += (NN + 1 - n)
                    start += 1
                e_orient.append(e_o)
                #Type where opposing face had vertices 102 compared to 012
                e_o = []
                start = NN
                for m in range(NN + 1):
                    prev = start
                    e_o.append(ed_ids[i][start])
                    for n in range(NN - m):
                        e_o.append(ed_ids[i][prev - 1])
                        prev += -1
                    start += NN - m
                e_orient.append(e_o)
                #FINALLY! Last one:
                #Type where opposing face had vertices 210 compared to 012
                e_o = []
                start = len(ed_ids[i]) - 1
                for m in range(NN + 1):
                    prev = start
                    e_o.append(ed_ids[i][start])
                    for n in range(NN - m):
                        e_o.append(ed_ids[i][prev - (1 + n + m)])
                        prev += - (1 + n + m)
                    start -= (2 + m)
                e_orient.append(e_o)

            elif (e_type == 1) or (e_type == 3): #Rectangles
                #handle the case where we have n as a list
                if (e_type == 1):
                    e_n = [nn[0], nn[0]]
                else:
                    e_n = nn

                #The rectangle can be done much more elegantly -- first we
                #make an ids list - then we flip and rotate it as necessary
                ids = arange(len(ed_ids[i]))
                ids = ids.reshape(e_n[0] + 1, e_n[1] + 1)
                for jj in range(2):
                    for ii in range(4):
                        e_orient.append(\
                            (array(ed_ids[i])[ids.ravel()]).tolist())
                        ids = rot90(ids)
                    ids = ids.T
                #we have to switch two of the entries
                tmp = e_orient[1]
                e_orient[1] = e_orient[3]
                e_orient[3] = tmp
                #And we have to reverse the order for the last 3
                e_orient[-3:] = e_orient[-1:-4:-1]

            orient.append(e_orient)
    return orient
###############################################################################
def get_orient_num(orient, dim, one_num=False, reverse=False, ed_type=0):
    """Method that defines the number for a particular orientation of an
    edge.

    @param orient (\c int) List containing at least the first 2 points of
                           the node ordering in the edge

    @param dim (\c int) Dimension of the problem. If the problem is 3D with 2D
                        edges, then dim = 3

    @param one_num (\c bool) Flag to indicate that a single number should
                           be given instead

    @param reverse (\c bool) Flag to indicate that the coordinates are known
                           and the ordering is requestion.

    @param ed_type (\c int) The edge element type -- needs to be specified
                           for the reverse lookup.
                           0 -- Triangle
                           1 -- Quadrilateral

    @retval orient_num (\c int) The number (or coordinates) of where to
                               find the re-ordering for that orientation

    @verbatim
    Dimension = 1: Edge is defined by 2 points, [a, b]
        Other edge: [a, b], orient number [0, 0], 0 if one_num
        Other edge: [b, a], orient number [0, 1], 1 if one_num
    Dimension = 2:
     Triangle: Edge is defined by 3 points, [a, b, c]
        Other edge [a, b, ...], orient number [0, 0], 0 if one_num
        Other edge [b, c, ...], orient number [1, 0], 1 if one_num
        Other edge [c, a, ...], orient number [2, 0], 2 if one_num
        Other edge [a, c, ...], orient number [1, 0], 4 if one_num
        Other edge [b, a, ...], orient number [1, 1], 5 if one_num
        Other edge [c, b, ...], orient number [1, 2], 6 if one_num
     Rectangle: Edge is defined by 4 points, [a, b, c, d]
        Other edge [a, b, ...], orient number [0, 0], 0 if one_num
        Other edge [b, c, ...], orient number [1, 0], 1 if one_num
        Other edge [c, d, ...], orient number [2, 0], 2 if one_num
        Other edge [d, a, ...], orient number [3, 0], 3 if one_num
        Other edge [a, d, ...], orient number [1, 0], 4 if one_num
        Other edge [b, a, ...], orient number [1, 1], 5 if one_num
        Other edge [c, b, ...], orient number [1, 2], 6 if one_num
        Other edge [d, c, ...], orient number [1, 3], 7 if one_num
    @endverbatim
    """

    if reverse: #Do a reverse lookup
        #Note the empty spot for the triangles -- this is so we can use one
        #dictionary for both element types
        rev_orient = [[[0, 1, 2], [1, 2, 0], [2, 0, 1],\
                  [], [0, 2, 1], [1, 0, 2], [2, 1, 0]], \
                  [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], \
                  [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0]]]
        if (type(orient) is list):
            #Copied below
            dict1 = {12:0, 23:1, 34:2, 41:3, 31:2,\
               14:4, 21:5, 32:6, 43:7, 13:4}
            dict2 = {12:[0, 0], 23:[0, 1], 34:[0, 2], 41:[0, 3], 31:[0, 2],\
               14:[1, 0], 21:[1, 1], 32:[1, 2], 43:[1, 3], 13:[1, 0]}
            for k in dict2:
                if dict2[k] == orient:
                    orient = dict1[k]
                    break
        return rev_orient[ed_type][orient]

    #Otherwise, do a forward lookup
    if dim == 2:
        tmp = [[0, 0], [0, 1]]
        if orient[0] == 0:
            orient_num = 0
        else:
            orient_num = 1
        if not one_num:
            return tmp[orient_num]
        else:
            return orient_num

    if dim == 3:
        #define the orientation dictionary (copied from above)
        dict1 = {12:0, 23:1, 34:2, 41:3, 31:2, 14:4, 21:5, 32:6, 43:7, 13:4}
        dict2 = {12:[0, 0], 23:[0, 1], 34:[0, 2], 41:[0, 3], 31:[0, 2],\
               14:[1, 0], 21:[1, 1], 32:[1, 2], 43:[1, 3], 13:[1, 0]}
        if one_num:
            return dict1[10 * orient[0] + orient[1] + 11]
        else:
            return dict2[10 * orient[0] + orient[1] + 11]

###############################################################################
class Basis:
    """@class Basis
       @brief Class that contains the important datastructures
              that contain the basis
       @param     n        Degree of polynomial basis
       @param     dim      Dimension of basis
       @param     element   The type of element (see int_el_pqr)
    """
    def __init__(self, n, dim, element, mkbasis=True):
        """The class constructor.
        @brief The class constructor. Handles initialization of the class.

        @param     n       (\c int) Degree of polynomial basis
        @param     dim     (\c int) Dimension of basis
        @param     element (\c int) The type of element (see int_el_pqr)
        @param     mkbasis (\c bool) Flag (default = True). if mkbasis:
                                        do the GramSchmidt orthonormalization

        @note This function populates the basis class with the most important
              datastructures
        """

        self.TOL = 1e-14

        #print "Working on Preliminaries:\nInitializing..."
        nb = numbase(n, dim)
        #PQ_j = dim

        #Initialize the modal basis
        #BUG: Neglecting int() causes crash in sympy 0.7.1
        #     but works in sympy 0.6.7
        #if sympy.__version__ == '0.7.1':
        aij = [[Rational(int(i==j)) for i in range(nb)] for j in range(nb)]
        #aij = [[Rational(i==j) for i in range(nb)] for j in range(nb)]
        #aij = [[Rational(0)] * nb] * nb
        #for i in range(nb):
        #    aij[i][i] = Rational(1)

        #Figure out the coefficients of all the monomials
        #print "Creating coefficients."
        pqr = mk_pqr_coeff(n * 3, dim)
        #pqr_i = len(pqr)

        #Calculate all the integrals
        #print "Calculating integrals."
        #int_pqr, int_pqr_ed, el_verts, ids_ed = int_el_pqr(pqr, element)
        int_pqr, el_verts, ids_ed = int_el_pqr(pqr, element)
        ed_verts = [el_verts[ids_ed[i]] for i in range(len(ids_ed))]


        if mkbasis:
            #print "Preliminaries finished, starting basis creating:"
            #Do Gram-Shmidt orthonormalization
            for j in range(0, nb):
                #print "Creating basis", j+1, "of", nb
                #Now orthogonalize wrt old basis
                for k in range(0, j):
                    coeff = inprod(aij[j], aij[k], pqr, int_pqr)
                    for ii in range(nb):
                        aij[j][ii] = aij[j][ii] - aij[k][ii] * coeff

                #And Normalize
                coeff = inprod(aij[j], aij[j], pqr, int_pqr)
                for k in range(nb):
                    aij[j][k] = aij[j][k] / sqrt(coeff)
        else:
            pass
            #print "Preliminaries finished."
        #Assign created structures to public member variables
        #doxygen documentation is also created HERE TODO: detailed doc for these
        #variables should go here...

        ##Contains the coefficients of each of the bases. As and example,
        # aij[0] contains the coefficients for the first basis.
        self.coeffs = aij
        if dim == 0:
            self.coeffs = [[1.0]]

        ##Number of bases
        self.nb = nb

        ##Matrix defining what each monomial basis means -- that is, it
        # gives the degree of each monomial component x^p y^q z^r.
        # pqr[1] = [0 0 1] could give x^0 y^0 z^1, for example.
        self.monoms = pqr

        ##Contains the value of the integral of the monomial
        # over the element. volint_pqr[0] gives the volume of the
        # element, for example.
        self.elmint_monoms = int_pqr
        if dim==0:
            self.elmint_monoms = [1]

        ##Array containing the x,y,z coordinates of the element.
        # vol_verts[0] = [-1, -1, -1], for example. These are
        # labeled, in general counter-clockwise from bottom
        # (z smallest) to top (z largest).
        self.vol_verts = el_verts
        if dim==0:
            self.vol_verts = array([[0.0]])

        ##Array containing the x,y,z coordinates of the element
        # edges. ed_verts[0][0] = [-1, -1, -1], for example gives the
        # first vertex of the first edge. These are labeled, in
        # general counter-clockwise to give outward pointing normals
        # according to the right-hand rule
        self.ed_verts = ed_verts
        if dim==0:
            self.ed_verts = [array([0.0])]

        ##Array containing the ids of the vertices that make up the coordinates
        # of the element eges. vol_verts[ids_ed[0]] gives the
        # coordinates of the first edge, for example.
        # These are labeled, in general counter-clockwise to give outward
        # pointing normalsaccording to the right-hand rule
        self.ids_ed = ids_ed
        if dim==0:
            self.ids_ed = [[0]]

        ##Array containing the type of each edge. In 1D and 2D this is always
        # zeros everywhere. For 3D prisms, the element has both triangles (0's)
        # and squares (1's)
        self.ed_type = [0 for i in range(len(ids_ed))]
        for i in range(len(ids_ed)):
            if len(ids_ed[i]) == 4:
                self.ed_type[i] = 1
        if dim==0:
            self.ed_type = [0]

        ##Number of active monomials -- basically how many coefficients each
        #basis has.
        self.nm = nb

        ##Order of the created basis
        self.n = n

        ##Number of edges
        self.ne = len(self.ids_ed)

        ##The element type
        self.element = element

        ##Dimension of basis
        self.dim = dim

    def check(self, output=False):
        """Checks if the created basis is actually orthonormal over
        the volume of the element.

        This is not a thorough check since it depends on that the pqr_int and
        pqr arrays are absolutely correct.

        """
        aij = self.coeffs
        pqr = self.monoms
        int_pqr = self.elmint_monoms

        if output:
            a = ones((self.nb, self.nb))

        print("Checking if basis is orthogonal")
        checkfail = False
        for i in range(self.nb):
            print(".")
            for j in range(self.nb):
                #Find inner Products
                coeff = inprod(aij[i], aij[j], pqr, int_pqr)
                if output:
                    a[i][j] = coeff
                if i == j:
                    if abs(coeff - 1) > self.TOL:
                        checkfail = True
                else:
                    if abs(coeff) > self.TOL:
                        checkfail = True

        if checkfail:
            print("\nOrthogonality check: FAILED! Basis formed is not" + \
            " orthogonal")
        else:
            print("\nOrthogonality check: PASSED.")
        if output:
            return a

    def evalf(self):
        """ function that returns the floating-point values of aij in a numpy
        array
        """
        #aij = zeros((self.nb, self.nm))
        #for i in range(self.nb):
        #    for j in range(self.nm):
        #        #Note, python float has the precision of a 'double' in C
        #        aij[i, j] = float(self.coeffs[i][j].evalf())
        #return aij
        #The faster way:
        return array(self.coeffs, dtype=float)

    def evalf_der(self, der_dim):
        """ function that returns the floating-point values of the derivatives
        of aij with respect to the dimension der_dim in a numpy array

        @param der_dim (\c int) the dimension along which to take a derivative.

        @note der_dim < dim. 0-->x, 1-->y, 2-->z
        """

        coeffs = polyder_mat(self.coeffs, self.monoms, der_dim)
        #aij = zeros((self.nb, self.nm))
        #for i in range(self.nb):
        #    for j in range(self.nm):
        #        #Note, python float has the precision of a 'double' in C
        #        aij[i, j] = float(coeffs[i][j].evalf())
        #return aij

        #The faster way
        return array(coeffs, dtype=float)

    def eval_at_pts(self, pts, der_dim=None, bases=None):
        """ Evaluate the bases, or their derivatives, at specified points.

        @param pts    Points to evaluate basis pts = [[x1,y1,z1],[x2,y2,z2],...]

        @param bases  (\c int) List of bases that  should be evaluated at the
                       specified points
                       (Optional) by default all bases are evaluated at all

                      points. For bases = [0, 3, 4], only bases 0, 3, and 4 will
                      be evaluated at all points.

        @param der_dim (\c int) (Optional)Dimension along which a derivative
                        should be taken. If None, no derivative is taken.

        @retval shap numpy array objects of size (len(pts) x nb).
                     shap(i,j) gives the value of basis 'j' evaluated at pts[i]
        """

        #By default, all bases are evaluated at all points
        if bases == None:
            bases = range(len((self.coeffs)))

        #Form the vandermonde matrix
        V = self.mk_vander(pts)

        #Evaluate the bases at the points using the Vandermonde matrix
        #Note the transpose, since each row of aij stores the coefficients for
        #one basis
        if der_dim == None:
            shap = dot(V, self.evalf()[bases, :].T)
        else:
            shap = dot(V, self.evalf_der(der_dim)[bases, :].T)

        return shap

    def mk_vander(self, pts):
        """
        @param pts    Points to evaluate basis pts = [[x1,y1,z1],[x2,y2,z2],...]

        @retval V     Vandermonde matrix. V_ij = monom_i (pts_j)
        """
        pts = array(pts)
        V = ones((len(pts), self.nm))
        for i in range(self.nm):
            for j in range(self.dim):
                V[:, i] = V[:, i] * pts[:, j]**self.monoms[i][j]
        return V

    def mkmapcoords(self, pts):
        """Creates the linear mapping coordinates for the current basis

        @param pts (\c float) Numpy array of points existing on the master
                   element that need to be mapped to an arbitrary element

       @retval T (\c float) Numpy array transformation such that
                   pts_xy = dot(T, verts_xy), where pts_xy are the points in
                   real x-y space, and verts_xy are the vertices of the element
                   defined in x-y space

       @see master.mk_basis.mk_mapcoords

       @code
       >>> import master.mk_basis as mkb
       >>> verts = array([[-1.1,-1.2,-1.3],[1.3, -1.4, -1.5],[1.5, 1.6, -1.7],\
                        [1.7, -1.8, 1.9]])
       >>> basis = mkb.Basis(1, 3, 0)
       >>> pts, fid = mkb.uniformlocalpts(basis, 10)
       >>> T = basis.mkmapcoords(pts)
       >>> pts2 = dot(T, verts)
       @endcode
        """
        return(mk_mapcoords(pts, self.vol_verts, self.element, self.dim))

    def copy(self, newcoeffs=None):
        """Creates a shallow copy of the basis object. Everything references
        the original basis except for the new coefficients

        @param newcoeffs (\c Symbol) Numpy array of Sympy numbers that represent
                         the coefficients of the new basis.

        @retval The new basis that completely references this basis, but with
                a different coefficient matrix
        """

        b = cpy.copy(self)
        if newcoeffs is not None:
            b.coeffs = newcoeffs
            b.nb = len(newcoeffs)
            b.nm = len(newcoeffs[0])

        return b

    def plot_elm(self, fignum=1):
        """ function that plots the element and labels the order of the vertex
        @param fignum (\c int) What figure window number should be used for
                      plotting
        """
        import matplotlib as mpl
        if (self.dim >= 3):
            #This module is used in the projection='3d' flag when creating ax
            from  mpl_toolkits.mplot3d import Axes3D

        import matplotlib.pyplot as plt

        mpl.rcParams['legend.fontsize'] = 10

        verts = self.vol_verts
        fig = plt.figure(fignum)

        if (self.dim == 1):
            verts = hstack([vstack(verts), vstack([0]*len(verts))])

        x = hstack([verts[:, 0], verts[0, 0]])
        y = hstack([verts[:, 1], verts[0, 1]])
        if (self.dim >= 3):
            ax = fig.gca(projection='3d')
            z = hstack([verts[:, 2], verts[0, 2]])
            ax.plot(x, y, z, 'bo-', label='Volume')
            #ax.axis3d([-1.1, 1.1, -1.1, 1.1, -1.1, 1.1])
        else:
            ax = fig.gca()
            ax.plot(x, y, 'bo-', label='Volume')
            ax.axis([-1.1, 1.1, -1.1, 1.1])

        #Add vertex labels
        for i in range(len(verts)):
            if (self.dim >= 3):
                label = 'V%d (%d,%d,%d)' % (i, verts[i, 0], verts[i, 1], \
                                                            verts[i, 2])
                ax.text(verts[i, 0], verts[i, 1], verts[i, 2], label, \
                                                            color='blue')
            else:
                if (self.dim == 1):
                    label = 'V%d (%d)' % (i, verts[i, 0])
                else:
                    label = 'V%d (%d,%d)' % (i, verts[i, 0], verts[i, 1])

                ax.text(verts[i, 0], verts[i, 1], label, color='blue')
        ax.legend(loc='best')
        plt.show()

    def plot_ed(self, ednum=None, fignum=1):
        """ function that plots the element edges, labels the edge number, and
        labels the the order of the vertex

        @param ednum    The number of the edges to plot. By default all edges
                        are plotted. NOTE, ednum should be a list!
                        ednum = [0] is valid.
        @param fignum (\c int) What figure window number should be used for
                      plotting
        """
        if not ednum:
            ednum = range(len(self.ed_verts))

        import matplotlib as mpl
        if (self.dim >= 3):
            #This module is used in the projection='3d' flag when creating ax
            from  mpl_toolkits.mplot3d import Axes3D

        import matplotlib.pyplot as plt

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure(fignum)
        plotcol = ['r:x', 'c:x', 'g:x', 'b:x', 'm:x', 'k:x']

        for ed in ednum:
            verts = array(self.ed_verts[ed])

            if (self.dim >= 3):
                from matplotlib.colors import colorConverter
                import mpl_toolkits.mplot3d.art3d  as ar3

                cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.4)

                poly = ar3.Poly3DCollection([verts], \
                    facecolors = [cc(plotcol[ed][0])])
                poly.set_alpha(0.4)
                ax = fig.gca(projection='3d')
                ax.add_collection3d(poly)
                ax.set_xlim3d([-1, 1])
                ax.set_ylim3d([-1, 1])
                ax.set_zlim3d([-1, 1])
            else:
                if (self.dim == 1):
                    verts = hstack([vstack(verts), vstack([0] * len(verts))])
                ax = fig.gca()
                x = verts[:, 0]
                y = verts[:, 1]
                ax.plot(x, y, plotcol[ed], linewidth=4, label='Edge %d' % (ed))
                ax.axis([-1.1, 1.1, -1.1, 1.1])

            #Add face labels
            xmid = mean(verts[:, 0])
            ymid = mean(verts[:, 1])
            if (self.dim >= 3):
                zmid = mean(verts[:,2])
                label = 'E%d' % (ed)
                ax.text(xmid, ymid, zmid, label, color=plotcol[ed][0])
            else:
                if (self.dim == 1):
                    label = 'E%d' % (ed)
                else:
                    label = 'E%d' % (ed)

                ax.text(xmid, ymid, label, color=plotcol[ed][0])

            #Add vertex labels
            offset = 0.3
            for i in range(len(verts)):
                #Offset label location slightly (so not to overlap with volume
#                labels)
                x = xmid * offset + (1 - offset) * verts[i, 0]
                y = ymid * offset + (1 - offset) * verts[i, 1]

                if (self.dim >= 3):
                    label = 'E%dV%d \n(%d,%d,%d)' \
                        % (ed, i, verts[i, 0], verts[i, 1], verts[i, 2])
                    z = zmid * offset + (1 - offset) * verts[i, 2]
                    ax.text(x, y, z, label, color=plotcol[ed][0])
                else:
                    if (self.dim == 1):
                        label = 'E%dV%d \n(%d)' % (ed, i, verts[i, 0])
                    else:
                        label = 'E%dV%d \n(%d,%d)' \
                             % (ed, i, verts[i, 0], verts[i, 1])

                    ax.text(x, y, label, color=plotcol[ed][0])

        if self.dim < 3:
            ax.legend(loc='best')
        plt.show()

###############################################################################
def main():
    """ The main function simply makes arrangements to call this script from
    the command line.

    Usage: To create a 3rd degree (n=3) basis in 2d (dim=2) using one of the
    pre-defined elements (element=2), use:
    @code python mk_basis -n 3 -dim 2 -element 2
    @endcode

    @note
    The coefficient matrix (aij) will then be printed to standard output, as
    well as the matrix (pqr) with powers of x, y, and z of each monomial basis.
    For example, the row of x of axj has a coefficient, where each coefficient
    corresponds with a monomial described by row j of pqr, where
    pqr[j,:] = [p, q, r], where the monomial is then x^p * y^q * z^r

    @see mk_aij
    @see pyutil.parse
    """
    import sys
    #Set defaults
    n = 0
    dim = 1
    element = 0
    (n, dim, element) = parse(sys.argv, ("-n", "-dim", "-element"))
    n = int(n)
    dim = int(dim)
    element = int(element)
    if dim < 1:
        print("Please specify a dimensions greater than 1")
    if n < 0:
        print("Please specify a polynomial degree greater than or equal to 0")
    if element < 0 | element > 3:
        print("Please specify a predefined element from 0 to 3")

    basis = Basis(n, dim, element, False)
    print('aij=', basis.coeffs)
    print('pqr=', basis.monoms)

###############################################################################
if __name__ == "__main__":
    main()
