# -*- coding: utf-8 -*-
"""
@package src.master.mk_nodal_basis Makes the nodal polynomial bases.

Created on Fri Jul 15 09:47:51 2011

@author: Matt Ueckermann
"""
from sympy import Rational, Symbol, integrate, sqrt, binomial

from sympy.polys import Poly
from sympy.matrices import Matrix
import mk_basis as mkb
import mk_cubature as mkc
from numpy import  array, mgrid, ones, dot, abs, column_stack, rot90, zeros, eye
from numpy import arange, argsort, append
from numpy.linalg import inv
import copy
import pdb

#Just a dummy class to use in uniformlocalpts
class dmy_struct:
    pass

def _lagrange_1d(xpts, evalpts):
    """Evaluate the 1D Lagrange (nodal) polynomial basis in 1D at evalpts.
    @param xpts: (\c float) List or array of points that define the nodal points for the Lagrange basis
    @param evalpts: (\c float) List or array of points where the basis will be evaluated
    @retval latpts (\c float) Array of the Lagrange bases evaluated at the
                    the points. latpts[i, m] evaluates the Lagrange basis that
                    is 1 at xpts[i] and zero elsewhere at the point evalpts[m]
    """
    latpts = ones((len(xpts), len(evalpts)))
    for j in xrange(len(xpts)):
        for m in xrange(len(xpts)):
            if j is not m:
                latpts[j, :] *= (evalpts - xpts[m]) / (xpts[j] - xpts[m])
    return latpts

def _mk_lagrange_1d_sym(xpts, coord):
    """Make the symbolic 1D Lagrange (nodal) polynomial basis.
    @param xpts: (\c float) List or array of points that define the nodal
                    points for the Lagrange basis
    @param coord: (\c Symbol) Symbol to use for making the 1D polynomial
    @retval lag (\c Symbol) List of the symbolic Lagrange bases
    """
    lag = [Rational('1') for i in range(len(xpts))]
    for j in xrange(len(xpts)):
        for m in xrange(len(xpts)):
            if j is not m:
                lag[j] *= (coord - xpts[m]) / (xpts[j] - xpts[m])
    return lag

def _warpfactor(r, oldpts, newpts):
    """Calculate the warp-factor to move points on the triangle. This is
    basically a coordinate transformation calculation.

    @param r (\c float) Points where the warpfactor will be evaluated
    @param oldpts (\c float) The old coordinate frame points.
    @param newpts (\c float) The new coordinate frame points.
    @retval w (\c float) The warpfactor calculated at the points r.
    @note The warpfactor is scale and calculated as per
        Hesthaven and Warburton (2008) book on Nodal Discontinuous Galerkin
        methods. """

    #Evaluate the Lagrange basis at the points r, using the u_pts as the nodal
    #points
    latpts = _lagrange_1d(oldpts, r)

    #Need to find places where the denominator will not be
    #zero in the scale factor
    notzero = (abs(r) - 1.0) < -1e-10
    scale = 1 - r ** 2
    scale[notzero==False] = 1

    w = dot(latpts.T, newpts - oldpts)# / scale
    w[notzero==False] = 0

    #Return the warpfactor
    return w

def _warpshift_2Dtri(lamb, LGLpts, basis):
    """Calculate the warp-factor to move points on the triangle. This is
    basically a coordinate transformation calculation.

    @param lamb (\c float) The uniform point set defined using the barycentric coordinate system.
    @param LGLpts (\c float) The 1d 'nice' point set, from which uniform points will be shifted too
    @param basis (\c float) The basis object. In this case, basis needs to
                                contain a few fields:
                  basis.n -- the order of the nodal points
                  basis.dim -- the dimension of the nodal set
                  basis.element -- the element type
                  basis.vol_verts -- vertices defining the master element
                          -- given in a numpy array [xpts, ypts]
    @retval ptshift (\c float) The shift required on an equil-lateral master
        triangle defined by verts = array([[-1, 0], [1, 0], [0, sqrt(3.)]])
    @note The warpfactor is scale and calculated as per
        Hesthaven and Warburton (2008) book on Nodal Discontinuous Galerkin
        methods. """

    #Calculate the blending factors
    b = [4 * lamb[:, i] * lamb[:, j] \
        / (2 * lamb[:, i] + lamb[:, k] \
        + ((abs(lamb[:, i]) < 10e-10) & (abs(lamb[:, k]) < 10e-10)))\
        / (2 * lamb[:, j] + lamb[:, k] + \
        + ((abs(lamb[:, j]) < 10e-10) &  (abs(lamb[:, k]) < 10e-10)))\
        for i, j, k in zip([1, 2, 0], [0, 1, 2], [2, 0, 1])]

    #Calculate the warp factors!
    u_pts_1d = mgrid[-1: 1: complex(basis.n + 1)]
    w = [b[0] * _warpfactor(lamb[:, 1] - lamb[:, 0], u_pts_1d, LGLpts), \
        b[1] * _warpfactor(lamb[:, 2] - lamb[:, 1], u_pts_1d, LGLpts), \
        b[2] *_warpfactor(lamb[:, 0] - lamb[:, 2], u_pts_1d, LGLpts)]

    #Now calculate how much the points need to be shifted
    ptshift = column_stack((1.0 * w[0] - 0.5 * w[1] - 0.5 * w[2], \
        0.0 * w[0] + sqrt(3.)/2. * w[1] - sqrt(3.)/2. * w[2]))

    #Return the warpfactor shifts
    return ptshift

def mk_nodal_pts(n, dim, element):
    """Make nicely-spaced nodal points for which to define a nodal basis.

    @param n (\c int) An integer of the order of the basis.
    @param dim (\c int) The dimension of the basis
    @param element (\c int) The element type (see master.mk_basis.int_el_pqr)
    @retval newpts (\c float) Numpy array of the nodal point locations
    @retval flocal (\c int) Indexing array that gives the ids of the points that
                    are on the the edges. For example: newpts[flocal[0], :]
                    gives all the points on edge 0.

    @note What is meant by 'nice'? Nice basically means that the Lebesgue
        constant is small -- or smaller than what you would find if using
        uniformly-spaced nodes. The Lesesgue constant gives an error bound
        on how good the interpolation is for a function compared to the optimal
        interpolation using a polynomial of the same degree.

    @note Ideas from Hesthaven and Warburton's (2008) book were used to create
        The nodes for the triangle and tetrahedral elements

    @see src.master.mk_basis.int_el_pqr
    """
    #if in 0D, we're done
    if dim == 0:
        return array([[0.0]], dtype=float), [[1]]

    #tolerance value for comparisons
    tol = 10e-10
    basis = dmy_struct()
    basis.n = n
    basis.dim = dim
    basis.element = element
    basis.vol_verts, fid = mkb.int_el_pqr(element=element, dim=dim)
    basis.ed_verts = [basis.vol_verts[fid[i]] for i in range(len(fid))]
    #We always like mkb_uniformlocalpts for all elements because it gives us
    #the flocal structure -- tells use which points are on boundaries and such
    u_pts, flocal = mkb.uniformlocalpts(basis, n)

    #Now we transform these points to nicely-behaved points -- which we first
    #have to fetch
    if type(n) is list:
        if len(n) == 1:
            n = n[0]
    if type(n) is int:
        u_pts_1d = mgrid[-1: 1: complex(n + 1)]
        if n == 0:
            u_pts_1d = array([0])
            #CM added 7/28/14
            u_pts = array([u_pts])
        u_pts_1d_z = u_pts_1d
        if n == 0:
            LGLpts_x = array([0])
        elif n==1:
            LGLpts_x = array([-1, 1])
        else:
            LGLpts_x = array([-1] + \
                mkc.gauss_quad(n * 2 - 3, alpha=1, beta=1)[0].tolist() + [1])
        LGLpts_z = LGLpts_x
    elif type(n) is list:
        u_pts_1d = mgrid[-1: 1: complex(n[0] + 1)]
        u_pts_1d_z = mgrid[-1: 1: complex(n[1] + 1)]
        if n[0] == 0:
            u_pts_1d = array([0])
        if n[1] == 0:
            u_pts_1d_z = array([0])

        #First to the points for the x-y direction
        if n[0] == 0:
            LGLpts_x = array([0])
        elif n[0]==1:
            LGLpts_x = array([-1, 1])
        else:
            LGLpts_x = array([-1] + \
                mkc.gauss_quad(n[0] * 2 - 3, alpha=1, beta=1)[0].tolist() + [1])
        #Then do it for the z-direction
        if n[1] == 0:
            LGLpts_z = array([0])
        elif n[1]==1:
            LGLpts_z = array([-1, 1])
        else:
            LGLpts_z = array([-1] + \
                mkc.gauss_quad(n[1] * 2 - 3, alpha=1, beta=1)[0].tolist() + [1])

    #if in 1D, we're done
    if dim == 1:
        return LGLpts_x.reshape((len(LGLpts_x),1)), flocal
    #Now we have nice points -- next is to move the uniform-created points to
    #the points where we want them. This will be different depending on the
    #element type.

    #print element, dim

    if element == 0:
        if dim == 2:
            if type(basis.n) == list:
                basis.n = n[0]
            #It's easier to do everything in barycentric coordinates:
            lamb = 0.5 * array([1 - u_pts[:, 0], u_pts[:, 0] - u_pts[:, 1], \
                u_pts[:, 1] + 1]).T
            #print lamb
            #Calculate the point-shift required on an equil-lateral triangle
            ptshift = _warpshift_2Dtri(lamb, LGLpts_x, basis)
            #Now, this only really works for equil-lateral triangles -- so we
            #have to transform the current points to an equil-lateral triangle
            T = mkb.mk_mapcoords(u_pts, basis.vol_verts, element, dim)
            eq_tri_verts = array([[-1, 0], [1, 0], [0, sqrt(3.)]])
            eq_pts = dot(T, eq_tri_verts)

            #Now move the points based on the shift calculated
            neweqpts = eq_pts + ptshift

            #And move the points back to the master-element coordinates
            T = mkb.mk_mapcoords(neweqpts, eq_tri_verts, element, dim)
            newpts = dot(T, basis.vol_verts)

            #Done!
            return newpts, flocal

        elif dim == 3:
            #It's easier to do everything in barycentric coordinates
            lamb = 0.5 * array([1 - u_pts[:, 0], \
                u_pts[:, 0] - u_pts[:, 1] - 1 - u_pts[:, 2], \
                u_pts[:, 1] + 1, 1 + u_pts[:, 2]]).T
            #Make the vertex list for the 3D equilateral tetrahedral
            eq_tri_verts = array([[-1, -1/sqrt(3.), -1/sqrt(6.)],\
                [1, -1/sqrt(3.), -1/sqrt(6.)], \
                [0, 2 / sqrt(3.), -1/sqrt(6.)], [0, 0, 3. / sqrt(6.)]])
            shift = zeros((len(lamb), 3))
            #we have to do this dance for every face
            for i in range(4):
                #One of the barycentric coordinates will be zero for this face,
                #so only use the other 3:
                loc_lamb = lamb[:, fid[i]]
                #Now find the shifts required in this face (on the orhogonal
                #directions on this face
                shiftt1, shiftt2 = _warpshift_2Dtri(loc_lamb, \
                    LGLpts_x, basis).T

                #Next we have to construct the two orthogonal vectors that are
                #tangent to the current face
                f_verts = eq_tri_verts[fid[i]]
                v1 = array(f_verts[1] - f_verts[0], float)
                v2 = array(f_verts[2] - (f_verts[1] + f_verts[0]) / 2., float)
                #And normalize the vectors (with 1 iteration)
                v1 = v1 / sqrt(dot(v1, v1))
                v2 = v2 / sqrt(dot(v2, v2))
                v1 = v1 / sqrt(dot(v1, v1))
                v2 = v2 / sqrt(dot(v2, v2))

                #Finally, we have to create the 3D versions of the Blending
                #functions -- which gets multiplied through the two shifts

                b = 8 * loc_lamb[:, 0] * loc_lamb[:, 1] * loc_lamb[:, 2] \
                    / (2 * loc_lamb[:, 0] + lamb[:, i] \
                    + ((abs(loc_lamb[:, 0]) < tol) & (abs(lamb[:, i]) < tol)))\
                    / (2 * loc_lamb[:, 1] + lamb[:, i] + \
                    + ((abs(loc_lamb[:, 1]) < tol) & (abs(lamb[:, i]) < tol)))\
                    / (2 * loc_lamb[:, 2] + lamb[:, i] + \
                    + ((abs(loc_lamb[:, 2]) < tol) & (abs(lamb[:, i]) < tol)))

                #Now we have to make sure that the blending function weight is
                #equal to 1 on the faces and edges, and that there is no double
                #dipping
                ids1 = (lamb[: ,i] < tol)
                b[ids1] = 1.

                shiftt1 *= b
                shiftt2 *= b
                shiftt1 = dot(shiftt1.reshape(len(shiftt1), 1), v1.reshape(1, 3))
                shiftt2 = dot(shiftt2.reshape(len(shiftt2), 1), v2.reshape(1, 3))
                #add these contributions to the global shift matrix
                shift = shift + shiftt1 + shiftt2
                #By not adding -- that is setting shift explicitly equal, we
                #avoid the double-dipping.
                shift[ids1] = shiftt1[ids1] + shiftt2[ids1]

            #Now we have the shifts necessary for points in the equil-lateral
            #tetrahedral element, so first transform the current points to the
            #equil-lateral:
            T = mkb.mk_mapcoords(u_pts, basis.vol_verts, element, dim)
            eq_pts = dot(T, eq_tri_verts)

            #Now move the points based on the shift calculated
            neweqpts = eq_pts + shift

            #And move the points back to the master-element coordinates
            T = mkb.mk_mapcoords(neweqpts, eq_tri_verts, element, dim)
            newpts = dot(T, basis.vol_verts)

            return newpts, flocal

    if element == 1:
        #Basically we make use of the 1D warpfactor -- no blending needed here
        #print u_pts
        #print u_pts_1d
        #print LGLpts_x
        #print _warpfactor(u_pts[:, 0], u_pts_1d, LGLpts_x)

        w = [_warpfactor(u_pts[:, i], u_pts_1d, LGLpts_x) for i in range(dim)]
        if dim == 3 and type(n) is list:
            if len(n) > 1:
                w[2] = _warpfactor(u_pts[:, 2], u_pts_1d_z, LGLpts_z)

        newpts = array([u_pts[:, i] + w[i] for i in range(dim)]).T
        return newpts, flocal

    if element == 2:
        if dim != 3:
            print "WARNING: NO ELEMENT 2 for dim !=3"
        #This actually ends up pretty easy. We simply use the code for the 2D
        #triangle for the x-y components, then use the 1D idea (as in element1)
        #for the z components
        #Initialize
        newpts = zeros((len(u_pts), 3))
        #It's easier to do everything in barycentric coordinates:
        lamb = 0.5 * array([1 - u_pts[:, 0], u_pts[:, 0] - u_pts[:, 1], \
            u_pts[:, 1] + 1]).T
        #Calculate the point-shift required on an equil-lateral triangle
        #We need a dummy basis for this:
        dumbase = dmy_struct()
        dumbase.dim = 2
        dumbase.element = 0
        if type(n) is list:
            dumbase.n = n[0]
        else:
            dumbase.n = n
        dumbase.vol_verts = mkb.int_el_pqr(element=0, dim=2)[0]
        ptshift = _warpshift_2Dtri(lamb, LGLpts_x, dumbase)

        #Now, this only really works for equil-lateral triangles -- so we
        #have to transform the current points to an equil-lateral triangle
        T = mkb.mk_mapcoords(u_pts[:, :2], dumbase.vol_verts, \
            dumbase.element, dumbase.dim)
        eq_tri_verts = array([[-1, 0], [1, 0], [0, sqrt(3.)]])
        eq_pts = dot(T, eq_tri_verts)

        #Now move the points based on the shift calculated
        neweqpts = eq_pts + ptshift

        #And move the points back to the master-element coordinates
        T = mkb.mk_mapcoords(neweqpts, eq_tri_verts, \
            dumbase.element, dumbase.dim)
        newpts[:, :2] = dot(T, dumbase.vol_verts)

        #Now do the z-direction
        ptshift = _warpfactor(u_pts[:, 2], u_pts_1d_z, LGLpts_z)
        newpts[:, 2] = u_pts[:, 2] + ptshift
        #Done
        return newpts, flocal


class Basis_nodal(mkb.Basis):
    """Make a nodal basis
    def __init__(self, n, dim, element, expand_monoms=False):
    """
    #CM  Python note: This class is a child class of Basis, in mk_basis.py
    #    so this class inherits Basis' members and methods.
    def __init__(self, n, dim, element, expand_monoms=False):
        """
        @param n (\c int) The polynomial degree of the basis
        @param dim (\c int) The dimension of the basis
        @param element (\c int) The element type (see mk_basis.Basis.int_el_pqr)
        @param expand_monoms (\c bool) Flag that, if true, calculates the
            integrals of all higher-order monomials needed for exact integration.
            If this flag is not set, the FEM operators should be created using
            quadrature instead of exact integration for elements 1, 2 in 3D
        """
        #Start by creating the modal basis -- which we will need to create
        #the nodal basis on the element
        if type(n) == list:
            nmax = max(n)
        else:
            nmax = n

        tol = 10e-15
        #We never want to initialize self with the modal basis, we will create
        #a modal basis specifically when we need to.
        mkb.Basis.__init__(self, nmax, dim, element, mkbasis=False)

        #Make sure n is correctly assigned -- not just nmax
        ##The polynomial degree of the basis
        self.n = n

        #Make the nodal points
        nodal_pts, ed_ids = mk_nodal_pts(n, dim, element)

        ##The nodal points that define where the nodal basis has value 1, and 0
        self.nodal_pts = nodal_pts
        #print nodal_pts
        ##list of ids that give the nodal points that lie on each face
        self.nodal_ed_ids = ed_ids

        if dim == 0:
            self.coeffs = [[1]]
            self.nb = 1
            self.nm = 1

        elif element == 0:
            if type(n) == list:
                n = n[0]
            basis_modal = mkb.Basis(n, dim, element)

            #Make the vandermonde matrix
            vand = basis_modal.eval_at_pts(nodal_pts)
            vand[abs(vand) < tol] = 0
            #print vand
            ivand = inv(vand)
            ivand[abs(ivand) < tol] = 0

            #Make the coefficients for this basis (mix between sympy and
            #floating point numbers -- no way around it because the nodal
            #points are ugly floating point numbers... )
            self.coeffs = dot(ivand.T, basis_modal.coeffs).tolist()

            self.nb = mkb.numbase(n, dim)
            self.nm = self.nb

        elif element == 1:
            #We can simply create the Lagrange basis directly using tensor
            #products and our friend sympy

            #First some annoying input parsing -- to take care of the case if
            #the z-direction has a different order than the x-y directions
            if type(n) == list:
                if (len(n) >= 2) and (dim == 3):
                    xypts = nodal_pts[:n[0] + 1, 0]
                    zpts = nodal_pts[::(n[0] + 1)**2, 2]
                    nx = n[0] + 1
                    ny = n[0] + 1
                    nz = n[1] + 1
                    nb = (n[0] + 1) ** 2 * (n[1] + 1)
                else:
                    xypts = nodal_pts[:n[0] + 1, 0]
                    zpts = xypts
                    nb = (n[0] + 1) ** dim
                    nx = n[0] + 1
                    ny = n[0] + 1
                    nz = n[0] + 1
            else:
                xypts = nodal_pts[:n + 1, 0]
                zpts = xypts
                nb = (n + 1) ** dim
                nx = n + 1
                ny = n + 1
                nz = n + 1

            x = Symbol('x')
            y = Symbol('y')
            z = Symbol('z')
            polys = [Rational('1') for i in range(nb)]
            coord = [x, y, z]

            #print "Building 1D bases."
            p = [_mk_lagrange_1d_sym(xypts, coord[i]) for i in range(dim)]
            #Handle the case when the z-pts are different from the xy-pts
            if type(n) == list:
                if (len(n) >= 2) and (dim == 3):
                    p[2] = _mk_lagrange_1d_sym(zpts, z)

            #Now we have to assemble the bases
            #print "Building tensor products of 1D bases."
            ndex = [0, 0, 0]
            for i in xrange(nb):
                for j in range(dim):
                    polys[i] *= p[j][ndex[j]]
                if dim == 3:
                    polys[i] = Poly(polys[i], x, y, z)
                elif dim ==2:
                    polys[i] = Poly(polys[i], x, y)
                else:
                    polys[i] = Poly(polys[i], x)
                #Now we have to move the indices along -- there is some
                #non-generality here for higher dimensions -- but it shouldn't
                #matter
                ndex[0] += 1
                if ndex[0] == nx:
                    ndex[0] = 0
                    ndex[1] += 1
                if ndex[1] == ny:
                    ndex[1] = 0
                    ndex[2] += 1
                if ndex[2] == nz:
                    ndex[2] = 0

            #Okay, next we have to take the POLY format, and put it into our
            #own format (because I wrote the original basis before knowing
            #about the poly class in sympy...


            n_monoms = len(self.monoms)
            #print "self.monoms = ", self.monoms[0:2]
            #print type(polys[0]), polys[0]
            #First we have to expand the size of self.monoms if we want to
            #analytically calculate the FEM operators
            if expand_monoms:
                self.monoms = mkb.mk_pqr_coeff(nmax * 2 * dim, dim)
                #And calculate the additional integrals
                print "Calculating additional integrals..."
                self.elmint_monoms.append(\
                    mkb.int_el_pqr(self.monoms[n_monoms:], element)[0])
            else:
                pass
#                print "WARNING: mk_basis.in_prod_mat CANNOT be used to create",\
#                   "the FEM operators. Use the tensor-product cubature instead."

            #Finally, make the coefficients matrix in our format
            coeffs = [[0 for i in range(n_monoms)] for j in range(nb)]
            for i in xrange(nb):
                #CM Neglecting () causes crash in sympy 0.7.1
                tmpcoeffs = array(polys[i].coeffs())
                #tmpcoeffs = array(polys[i].coeffs)
                #print polys[i]
                #print polys[i].coeffs()
                #print tmpcoeffs.shape
                for j in xrange(len(tmpcoeffs)):
                    if tmpcoeffs[j] is not 0:
                        #CM  self.monoms is a list
                        #    but Poly.monoms() is a method
                        #Need () here to prevent crash in sympy 0.7.1
                        ids = self.monoms.index(list(polys[i].monoms()[j]))
                        coeffs[i][ids] += tmpcoeffs[j]

            #Assign it to the class
            self.coeffs = coeffs

            #NOTE: It would be about 2 times as efficient if we strip out the
            #monomials which are not in use -- but because we don't have to use
            #The basis very often -- I'm not going to bother with the extra
            #Logic, althought I'll leave it as a TODO

            #And repair the rest of the variables that need to change
            self.nb = nb
            self.nm = n_monoms

        elif (element == 2) and (dim == 3):
            if type(n) == list:
                nb2d = mkb.numbase(n[0], 2)
                xypts = nodal_pts[:nb2d, 0:2]
                zpts = nodal_pts[::nb2d, 2]
                nxy = n[0]
                if (len(n) >= 2):
                    nz = n[1] + 1
                    nb = nb2d * nz
                else:
                    nz = n[0] + 1
                    nb = nb2d * nz
            else:
                nb2d = mkb.numbase(n, 2)
                xypts = nodal_pts[:nb2d, 0:2]
                zpts = nodal_pts[::nb2d, 2]
                nxy = n
                nz = n + 1
                nb = nb2d * nz

            #First make the 2D triangular nodal basis:
            base2d = Basis_nodal(nxy, dim=2, element=0)
            #Convert it to a sybolic representation
            x = Symbol('x')
            y = Symbol('y')
            z = Symbol('z')
            p1 = [Rational('0') for i in range(base2d.nb)]
            for i in xrange(base2d.nb):
                for j in xrange(base2d.nb):
                    #Create the polynomial for this basis
                    p1[i] += base2d.coeffs[i][j] * \
                        (x ** base2d.monoms[j][0]) * (y ** base2d.monoms[j][1])

            #Then make the 1D basis in z:
            p = _mk_lagrange_1d_sym(zpts, z)

            #Now combine these two to make our full basis
            polys = [Rational('1') for i in range(nb)]
            ndex = [0, 0]
            for i in xrange(nb):
                polys[i] = Poly(p[ndex[1]] * p1[ndex[0]], x, y, z)
                ndex[0] += 1
                if ndex[0] == nb2d:
                    ndex[0] = 0
                    ndex[1] += 1

            #Okay, next we have to take the POLY format, and put it into our
            #own format (because I wrote the original basis before knowing
            #about the poly class in sympy...


            n_monoms = len(self.monoms)
            #First we have to expand the size of self.monoms if we want to
            #analytically calculate the FEM operators
            if expand_monoms:
                self.monoms = mkb.mk_pqr_coeff(nmax * 2 * dim, dim)
                #And calculate the additional integrals
                print "Calculating additional integrals..."
                self.elmint_monoms.append(\
                    mkb.int_el_pqr(self.monoms[n_monoms:], element)[0])
            else:
                print "WARNING: mk_basis.in_prod_mat CANNOT be used to create",\
                   "the FEM operators. Use the tensor-product cubature instead."

            #Finally, make the coefficients matrix in our format
            coeffs = [[0 for i in range(n_monoms)] for j in range(nb)]
            for i in xrange(nb):
                #CM  Neglecting parens causes crash in sympy 0.7.1
                #but works in sympy 0.6.7
                tmpcoeffs = array(polys[i].coeffs())
                for j in xrange(len(tmpcoeffs)):
                    if tmpcoeffs[j] != 0:
                        ids = self.monoms.index(list(polys[i].monoms()[j]))
                        coeffs[i][ids] += tmpcoeffs[j]

            #Assign it to the class
            self.coeffs = coeffs

            #NOTE: It would be about 2 times as efficient if we strip out the
            #monomials which are not in use -- but because we don't have to use
            #The basis very often -- I'm not going to bother with the extra
            #Logic, althought I'll leave it as a TODO

            #And repair the rest of the variables that need to change
            self.nb = nb
            self.nm = n_monoms

        #Finally, fix the edge types. Type 0 is triangle, 1 is rectangle of
        #uniform order, and 3 is a rectangle of variable order
        #NOTE, variable order rectangles is NOT properly implemented, this is
        #a remnant of that attempt.
        if dim == 3:
            if type(n) is list:
                if len(n) > 1:
                    self.ed_type[2:] = [3] * len(self.ed_type[2:])

    def check(self):
        """Checks if the created basis is actually \f$\delta_{i\!j}\f$ -- that
             is nodal.

        This is not a thorough check since derivatives are not evaluated, but
            it's a reasonable things to do.

        """
        tol = 10E-14
        #Check
        #Rename variables for shorter code within function
        a = self.eval_at_pts(self.nodal_pts)
        b = sum(sum(a - eye(len(a)))) / len(a)

        checkfail = b > tol
        if checkfail:
            print "\nNodal check: FAILED! at", b," Basis formed is not" + \
            " a nodal basis as the points specified by self.nodal_pts."
        else:
            print "\nNodal check: PASSED."

class Basis_tensor_modal(mkb.Basis):
    """Make a modal basis that's the tensor product of 1D modal bases. This
    is for the filter, so that we convert the nodal basis (which is a
    tensor product of 1D bases) to a modal basis for filtering purposes.

    def __init__(self, n, dim, element, expand_monoms=False):
    """
    #CM  Python note: This class is a child class of Basis, in mk_basis.py
    #    so this class inherits Basis' members and methods.
    def __init__(self, n, dim, element, expand_monoms=False):
        """
        @param n (\c int) The polynomial degree of the basis
        @param dim (\c int) The dimension of the basis
        @param element (\c int) The element type (see mk_basis.Basis.int_el_pqr)
        @param expand_monoms (\c bool) Flag that, if true, calculates the
            integrals of all higher-order monomials needed for exact integration.
            If this flag is not set, the FEM operators should be created using
            quadrature instead of exact integration for elements 1, 2 in 3D
        """
        #Start by creating the modal basis -- which we will need to create
        #the nodal basis on the element
        if type(n) == list:
            nmax = max(n)
        else:
            nmax = n

        tol = 10e-15
        #We never want to initialize self with the modal basis, we will create
        #a modal basis specifically when we need to.
        mkb.Basis.__init__(self, nmax, dim, element, mkbasis=False)

        #Make sure n is correctly assigned -- not just nmax
        ##The polynomial degree of the basis
        self.n = n

        #Make the nodal points
        nodal_pts, ed_ids = mk_nodal_pts(n, dim, element)

        ##The nodal points that define where the nodal basis has value 1, and 0
        self.nodal_pts = nodal_pts
        #print nodal_pts
        ##list of ids that give the nodal points that lie on each face
        self.nodal_ed_ids = ed_ids

        if dim == 0:
            self.coeffs = [[1]]
            self.nb = 1
            self.nm = 1

        elif element == 0:
            if type(n) == list:
                n = n[0]
            basis_modal = mkb.Basis(n, dim, element)

            #Make the coefficients for this basis (mix between sympy and
            #floating point numbers -- no way around it because the nodal
            #points are ugly floating point numbers... )
            self.coeffs = basis_modal.coeffs

            self.nb = mkb.numbase(n, dim)
            self.nm = self.nb

        elif element == 1:
            #We can simply create the Lagrange basis directly using tensor
            #products and our friend sympy

            #First some annoying input parsing -- to take care of the case if
            #the z-direction has a different order than the x-y directions
            if type(n) == list:
                if (len(n) >= 2) and (dim == 3):
                    xypts = nodal_pts[:n[0] + 1, 0]
                    zpts = nodal_pts[::(n[0] + 1)**2, 2]
                    nx = n[0] + 1
                    ny = n[0] + 1
                    nz = n[1] + 1
                    nb = (n[0] + 1) ** 2 * (n[1] + 1)
                else:
                    xypts = nodal_pts[:n[0] + 1, 0]
                    zpts = xypts
                    nb = (n[0] + 1) ** dim
                    nx = n[0] + 1
                    ny = n[0] + 1
                    nz = n[0] + 1
            else:
                xypts = nodal_pts[:n + 1, 0]
                zpts = xypts
                nb = (n + 1) ** dim
                nx = n + 1
                ny = n + 1
                nz = n + 1

            x = Symbol('x')
            y = Symbol('y')
            z = Symbol('z')
            polys = [Rational('1') for i in range(nb)]
            coord = [x, y, z]

            #print "Building 1D bases."
            p = [_mk_legendre_1d_sym(n + 1, coord[i]) for i in range(dim)]
            #print n+1, coord
            #print p[0][2]
            #Handle the case when the z-pts are different from the xy-pts
            if type(n) == list:
                if (len(n) >= 2) and (dim == 3):
                    p[2] = _mk_legendre_1d_sym(len(zpts), z)

            #Now we have to assemble the bases
            #print "Building tensor products of 1D bases."
            ndex = [0, 0, 0]
            #print nb
            for i in xrange(nb):
                for j in range(dim):
                    polys[i] *= p[j][ndex[j]]
                    #print p[j][ndex[j]]
                #print "i: ", i, "=", polys[i], " ndex:", ndex
                if dim == 3:
                    polys[i] = Poly(polys[i], x, y, z)
                elif dim ==2:
                    polys[i] = Poly(polys[i], x, y)
                else:
                    polys[i] = Poly(polys[i], x)
                #Now we have to move the indices along -- there is some
                #non-generality here for higher dimensions -- but it shouldn't
                #matter
                ndex[0] += 1
                if ndex[0] == nx:
                    ndex[0] = 0
                    ndex[1] += 1
                if ndex[1] == ny:
                    ndex[1] = 0
                    ndex[2] += 1
                if ndex[2] == nz:
                    ndex[2] = 0

            #Okay, next we have to take the POLY format, and put it into our
            #own format (because I wrote the original basis before knowing
            #about the poly class in sympy...


            n_monoms = len(self.monoms)
            #print "self.monoms = ", self.monoms[0:2]
            #print type(polys[0]), polys[0]
            #First we have to expand the size of self.monoms if we want to
            #analytically calculate the FEM operators
            if expand_monoms:
                self.monoms = mkb.mk_pqr_coeff((nmax+2) ** dim, dim)
                #And calculate the additional integrals
                print "Adding additional monomial integrals"
                self.elmint_monoms = mkb.int_el_pqr(self.monoms, element)[0]

            #Finally, make the coefficients matrix in our format
            coeffs = [[0 for i in range(n_monoms)] for j in range(nb)]
            for i in xrange(nb):
                #CM Neglecting () causes crash in sympy 0.7.1
                tmpcoeffs = array(polys[i].coeffs())
                #tmpcoeffs = array(polys[i].coeffs)
                #print polys[i]
                #print polys[i].coeffs()
                #print tmpcoeffs.shape
                for j in xrange(len(tmpcoeffs)):
                    if tmpcoeffs[j] is not 0:
                        #CM  self.monoms is a list
                        #    but Poly.monoms() is a method
                        #Need () here to prevent crash in sympy 0.7.1
                        ids = self.monoms.index(list(polys[i].monoms()[j]))
                        coeffs[i][ids] += tmpcoeffs[j]

            #Assign it to the class
            self.coeffs = coeffs

            #NOTE: It would be about 2 times as efficient if we strip out the
            #monomials which are not in use -- but because we don't have to use
            #The basis very often -- I'm not going to bother with the extra
            #Logic, althought I'll leave it as a TODO

            #And repair the rest of the variables that need to change
            self.nb = nb
            self.nm = n_monoms

        elif (element == 2) and (dim == 3):
            if type(n) == list:
                nb2d = mkb.numbase(n[0], 2)
                xypts = nodal_pts[:nb2d, 0:2]
                zpts = nodal_pts[::nb2d, 2]
                nxy = n[0]
                if (len(n) >= 2):
                    nz = n[1] + 1
                    nb = nb2d * nz
                else:
                    nz = n[0] + 1
                    nb = nb2d * nz
            else:
                nb2d = mkb.numbase(n, 2)
                xypts = nodal_pts[:nb2d, 0:2]
                zpts = nodal_pts[::nb2d, 2]
                nxy = n
                nz = n + 1
                nb = nb2d * nz

            #First make the 2D triangular modal basis:
            base2d =  mkb.Basis(nxy, dim=2, element=0)
            #Convert it to a sybolic representation
            x = Symbol('x')
            y = Symbol('y')
            z = Symbol('z')
            p1 = [Rational('0') for i in range(base2d.nb)]
            for i in xrange(base2d.nb):
                for j in xrange(base2d.nb):
                    #Create the polynomial for this basis
                    p1[i] += base2d.coeffs[i][j] * \
                        (x ** base2d.monoms[j][0]) * (y ** base2d.monoms[j][1])

            #Then make the 1D basis in z:
            p = _mk_legendre_1d_sym(len(zpts), z)

            #Now combine these two to make our full basis
            polys = [Rational('1') for i in range(nb)]
            ndex = [0, 0]
            for i in xrange(nb):
                polys[i] = Poly(p[ndex[1]] * p1[ndex[0]], x, y, z)
                ndex[0] += 1
                if ndex[0] == nb2d:
                    ndex[0] = 0
                    ndex[1] += 1

            #Okay, next we have to take the POLY format, and put it into our
            #own format (because I wrote the original basis before knowing
            #about the poly class in sympy...
            n_monoms = len(self.monoms)
            #First we have to expand the size of self.monoms if we want to
            #analytically calculate the FEM operators
            if expand_monoms:
                self.monoms = mkb.mk_pqr_coeff((nmax+1) ** dim, dim)
                #And calculate the additional integrals
                print "Adding additional monomial integrals"
                self.elmint_monoms = mkb.int_el_pqr(self.monoms, element)[0]
            else:
                print "WARNING: mk_basis.in_prod_mat CANNOT be used to create",\
                   "the FEM operators. Use the tensor-product cubature instead."

            #Finally, make the coefficients matrix in our format
            coeffs = [[0 for i in range(n_monoms)] for j in range(nb)]
            for i in xrange(nb):
                #CM  Neglecting parens causes crash in sympy 0.7.1
                #but works in sympy 0.6.7
                tmpcoeffs = array(polys[i].coeffs())
                for j in xrange(len(tmpcoeffs)):
                    if tmpcoeffs[j] != 0:
                        ids = self.monoms.index(list(polys[i].monoms()[j]))
                        coeffs[i][ids] += tmpcoeffs[j]

            #Assign it to the class
            self.coeffs = coeffs

            #NOTE: It would be about 2 times as efficient if we strip out the
            #monomials which are not in use -- but because we don't have to use
            #The basis very often -- I'm not going to bother with the extra
            #Logic, althought I'll leave it as a TODO

            #And repair the rest of the variables that need to change
            self.nb = nb
            self.nm = n_monoms

        #Finally, fix the edge types. Type 0 is triangle, 1 is rectangle of
        #uniform order, and 3 is a rectangle of variable order
        #NOTE, variable order rectangles is NOT properly implemented, this is
        #a remnant of that attempt.
        if dim == 3:
            if type(n) is list:
                if len(n) > 1:
                    self.ed_type[2:] = [3] * len(self.ed_type[2:])

        #Finally, we want to order the basis by their degrees, from lowest
        #to highest degree
        nb = self.nb
        self.basis_degree = [0 for i in range(nb)]
        for i in xrange(nb):
            ids = abs(array(self.coeffs[i])) > tol
            padLength = len(self.monoms) - len(ids)
            pad = zeros(padLength, dtype=bool)
            paddedIds = append(ids, pad)
            #pdb.set_trace()
            #self.basis_degree[i] = array(self.monoms)[ids].sum(1).max()
            self.basis_degree[i] = array(self.monoms)[paddedIds].sum(1).max()
        ids = argsort(self.basis_degree).tolist()
        self.basis_degree = array(self.basis_degree)[ids].tolist()
        self.coeffs = array(self.coeffs)[ids, :].tolist()

def _mk_legendre_1d_sym(nb, coord):
    """Make the symbolic 1D Legendre (modal) polynomial basis.
    @param nb: (\c int) The number of terms in the legendre basis
    @param coord: (\c Symbol) Symbol to use for making the 1D polynomial
    @retval lag (\c Symbol) List of the symbolic Lagrange bases
    """
    leg = [Rational('1') for i in range(nb)]
    leg[0] = leg[0] / 2.0 ** 0.5
    if nb >= 2:
        leg[1] = coord / (2.0 / 3.0) ** 0.5
        for n in range(2, nb):
            leg[n] = Rational('0')
            for k in range(n + 1):
                #print "(n,k) (", n, ",",k, ")"
                #print (2 ** n), (coord ** k), binomial(n, k),\
                #    binomial((n + k - 1) / 2.0, n)
                leg[n] += (2 ** n) * (coord ** k) * binomial(n, k) \
                    * binomial((n + k - 1.0) / 2.0, n) \
                    / (2.0 / (2.0 * n +1.0)) ** 0.5 #normalization constant
                #print leg[n]

    return leg
