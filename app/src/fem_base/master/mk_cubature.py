# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:46:31 2011

@author: -
"""
from scipy.special.orthogonal import j_roots as jacobi_roots
from numpy import ceil, array, dot, vstack, column_stack, real
from src.fem_base.master.mk_basis import int_el_pqr, mk_mapcoords

def gauss_quad(n, alpha=0, beta=0):
    """Calculates the 1D gauss quadrature points \f$ x_i\f$ and weights
    \f$\mathbf w_i\f$ for 1D integration over the domain [-1,1]

    @param n (\c int) The order of integration for which the quadrature will be
              exact

    @param alpha (\c int) The \f$\alpha\f$ parameter for Jacobi-polynomials.
                  DEFAULT = 0, choose alpha=1 for Gauss-Radau quadrature

    @param beta (\c int) The \f$\beta\f$ parameter for Jacobi-polynomials.
                  DEFAULT = 0. I see no reason to change this, but coded just
                  in case.

    @retval x (\c float) Quadrature points

    @retval w (\c float) Quadrature weights

    @note To create the quadrature rules, we follow Golub, G. H. and
    J. H. Welsch. Calculation of Gauss Quadrature. Math. Comp 23(1969), 221-230.

    @note Turns out this function already exists in scipy. Thus, we basically
    wrote a new function JUST to document the procedure more thoroughly, AND
    to have a more convenient inputs for n. The procedure is as follows:
    Basically, to find the  quadrature points, we need to find the roots of the
    n'th order Jacobi polynomial. For this we need the re-currence relation for
    Jacobi polynomials
    \f{align*}{
    P_n^{\alpha,\beta} &= (a_n x + b_n) P_{n-1}^{\alpha,\beta}
    - c_n P_{n-2}^{\alpha,\beta}
    \f}
    where
    \f{align*}{
    a_n &= \frac{2n + \alpha + \beta + 2}{2} \sqrt{
        \frac{(2n + \alpha + \beta + 1)(2n + \alpha + \beta + 3)}
        {(n + 1)(n + \alpha + \beta + 1)(n + \alpha + 1)(n + \beta + 1)}} \\
    b_n &= {a_n}
        \frac{\alpha^2 - \beta^2}{(2n + \alpha + \beta)(2n+\alpha+\beta+2)} \\
    c_n &= \frac{a_{n}}{a_{n-1}}
    \f}
    Now, from Golub and Welsh (1969), we need to form the tri-diagonal matrix,
    where the main diagonal is populated by \f$A_i\f$ and the symmetric
    diagonals are populated by \f$B_i\f$, where
    \f{align*}{
    A_n &= -\frac{b_n}{a_n}\\
       &= -\frac{\alpha^2 - \beta^2}{(2n + \alpha + \beta)(2n+\alpha+\beta+2)}\\
    B_n &= \sqrt{\frac{c_{n + 1}}{a_na_{n - 1}}} \\
       &= \sqrt{\frac{a_{n + 1}}{a_{n}}\frac{1}{a_{n}a_{n + 1}}} \\
       &= \frac{1}{a_{n}} \\
       &= \frac{2}{2n + \alpha + \beta + 2} \sqrt{
        \frac{(n + 1)(n + \alpha + \beta + 1)(n + \alpha + 1)(n + \beta + 1)}
        {(2n + \alpha + \beta + 1)(2n + \alpha + \beta + 3)}} \\
    \f}

    Then, the quadrature points are the eigen-values \f$\mathbf V\f$ of this
    matrix. The quadrature weights can be calculated from the first
    component of the eigen-vectors as:
    \f{align*}{
        w_j &= V_{1,j}^2 \cdot \int_{-1}^{1} \omega(x) dx \\
         &=V_{1,j}^2 \cdot \int_{-1}^{1} (1-x)^\alpha (1+x)^\beta P_0 P_0 dx\\
         &=V_{1,j}^2 \cdot \frac{2^{\alpha + \beta + 1}}{\alpha + \beta + 1}
           \frac{\Gamma(\alpha + 1)\Gamma(\beta+1)}{\Gamma(\alpha + \beta + 1}
    \f}

    And apparently this is already implemented in scipy.

    @see scipy.special.orthogonal.j_roots
    """
    N = ceil((n + 1.) / 2.)
    return real(jacobi_roots(N, alpha, beta))

def mk_cube_rule(n, dim, elm):
    """Function that makes cubature rules using tensor products of 1D gauss
    quadrature rules.

    @param n (\c int) Order of polynomial that should be integrated exactly

    @param dim (\c int) Dimension of problem (1, 2, or 3)

    @param elm (\c int) Type of element (0, 1, or 2)

    @retval x (\c float) List of points and weights. x[0][1:dim] contains the
              coordinates of the first point, and x[0][-1] contains the weight

    @author Matt Ueckermann

    @note The method for creating tetrahedrals, prisms, and triangles cubature
    is taken from Karniadakis and Sherwin, Spectral/hp Element Methods for
    Computational Fluid Dynamics, 2nd Edition, (2005).
    """
    from src.master.mk_jacobian import cal_jacobian_at_pts
    #First get the 1D rules
    pts, wghts = gauss_quad(n)

    # IF 1D we're done, else do the necessary tensor products:
    if dim <= 1:
        #Next if else if for data-formatting purposes
        if n <= 1:
            return [[pts[0], wghts[0]]]
        else:
            return column_stack((pts, wghts)).tolist()
    else:
        x = []
        w = []
        #We need to use different points/weights for prisms, tets, tri's, see
        #following comments
        if (elm == 0) or (elm == 2):
            #Turns out that an elegant method for including the Jacobian when
            #doing the transformation is to use alpha = 1 for the y direction
            #when doing triangles, prisms, and tetrahedrals.
            # See Karniadakis and Sherwin 2005 (2nd edition), pgs 141-146
            ptsy, wghtsy = gauss_quad(n + 1, 1, 0)
            wghtsy = wghtsy / 2.
            if (dim == 3) and (elm == 0):
                #Turns out that an elegant method for including the Jacobian
                # when doing the transformation is to use alpha = 2 for the
                #z direction when tetrahedrals.
                # See Karniadakis and Sherwin 2005 (2nd edition), pgs 141-146
                ptsz, wghtsz = gauss_quad(n + 2, 2, 0)
                wghtsz = wghtsz / 4.
            else:
                ptsz = pts
                wghtsz = wghts
        else: #Otherwise, different points/weights are not needed
            ptsy = pts
            wghtsy = wghts
            ptsz = pts
            wghtsz = wghts

        #Do the Tensor products
        for x_pt, w1 in zip(pts, wghts):
            for y_pt, w2 in zip(ptsy, wghtsy):
                if dim == 2:
                    x = x + [[x_pt, y_pt, w1 * w2]]
                else:
                    for z_pt, w3 in zip(ptsz, wghtsz):
                        x = x + [[x_pt, y_pt, z_pt, w1 * w2 * w3]]

        #Now do the transformations for the no-square elements
        if elm == 0:
            #we have to do a coordinate transformation and include the Jacobian

            #First we have to grab the vertices of the square (1) and the
            #triangle/tetrahedral (2)
            verts1, dum = int_el_pqr(element=1, dim=dim)
            verts2, dum = int_el_pqr(element=0, dim=dim)

            #Now we need to copy the vertices in the triangle to have the same
            #number as in verts1 (the square/cube). It basically indicates how
            #the cube's vertices are transformed.
            if dim == 2:
                verts2 = vstack([verts2, verts2[-1, :]])
                pts = array([[x1, y] for x1,y,w in x])
                wghts = [[w] for x1,y,w in x]
            else:
                verts2 = vstack([verts2[:3 , :], verts2[-2, :], \
                    verts2[-1, :], verts2[-1, :], verts2[-1, :], verts2[-1, :]])
                pts = array([[x1, y, z] for x1,y,z,w in x])
                wghts = [[w] for x1,y,z,w in x]


            T = mk_mapcoords(pts, verts1, element=1, dim=dim)

            pts2 = dot(T, verts2).tolist()


            #Previously, without changing alpha, I had to manually include
            #the jacobian in the weights. Now this is no longer needed. i keep
            #it in comments for reference. Also, look at svn r17 for the
            #original code. The original code needs modification to exactly
            #integrate prisms, and never worked for tets. The jacobian and
            #point transformations where verified as correct, but the
            #integration was always wrong.
            #jac = cal_jacobian_at_pts(1, dim, verts2, pts2)
            #wghts = wghts * jac.T
            #wghts = (array(wghts) / 8.).tolist()

            #Finally, assign x
            x = column_stack((pts2, wghts)).tolist()

        elif elm == 2:
            #we have to do a coordinate transformation and include the Jacobian

            #First we have to grab the vertices of the square and the
            #triangle
            verts1, dum = int_el_pqr(element=1, dim=dim)
            verts2, dum = int_el_pqr(element=2, dim=dim)

            if dim == 2:
                print('WARNING: No element #2 in 2D.')
                print('No quadrature rule returned')
                return None
            else:
                verts2 = vstack([verts2[:3 , :], verts2[2, :], \
                    verts2[3:, :], verts2[-1, :]])
                pts = array([[x1, y,z] for x1,y,z,w in x])
                wghts = [[w] for x1,y,z,w in x]

            T = mk_mapcoords(pts, verts1, element=1, dim=dim)

            pts2 = dot(T, verts2).tolist()

            #Previously, without changing alpha, I had to manually include
            #the jacobian in the weights. Now this is no longer needed. i keep
            #it in comments for reference. Also, look at svn r17 for the
            #original code. The original code needs modification to exactly
            #integrate prisms: need to use a order n+1 polynomial in the
            #y-direction.
            #jac = cal_jacobian_at_pts(1, dim, verts2, pts2)
            #wghts = wghts * jac.T

            #Finally, assign x
            x = column_stack((pts2, wghts)).tolist()
        return x

def get_pts_weights(n, dim, elm, force_mk=False):
    """Looks up and/or generates the quadrature points and weights for the
    specified element/dimension and order 'n'.

    @param n (\c int) order of the polynomial which should be integrated exactly

    @param dim (\c int) dimension of the polynomial

    @param elm (\c int) The element type
    @see src.master.mk_basis.int_el_pqr

    @param force_mk (\c bool) Flag that, if true, will make a quadrature-rule
                    from gauss quadrature. Default=False.

    @retval x (\c float) Numpy array of quadrature points

    @retval w (\c float) Numpy array of quadrature weights

    @author Matt Ueckermann

    @note The look-up table is generated from the Cools tables:
    Cools, R. "Monomial Cubature Rules Since "Stroud": A Compilation--Part 2."
    J. Comput. Appl. Math. 112, 21-27, 1999.
    Cools, R. "Encyclopaedia of Cubature Formulas."
    http://www.cs.kuleuven.ac.be/~nines/research/ecf/ecf.html
    """
    #if n == 0:
    #    n = 1

    #This is basically a large switch to select the correct rule
    if dim == 1:
        x = array(mk_cube_rule(n, dim, 0))

    if (dim == 2) and (elm == 0):
        from src.master.cubature.cubature_tables import dim2_elm0
        if (n < len(dim2_elm0)) and not (force_mk):
            x = array(dim2_elm0[n])
        else:
            x = array(mk_cube_rule(n, dim, elm))

    elif (dim == 2) and (elm == 1):
        from src.master.cubature.cubature_tables import dim2_elm1
        if (n < len(dim2_elm1)) and not (force_mk):
            x = array(dim2_elm1[n])
        else:
            x = array(mk_cube_rule(n, dim, elm))

    elif (dim == 3) and (elm == 0):
        from src.master.cubature.cubature_tables import dim3_elm0
        if (n < len(dim3_elm0)) and not (force_mk):
            x = array(dim3_elm0[n])
        else:
            x = array(mk_cube_rule(n, dim, elm))

    elif (dim == 3) and (elm == 1):
        from src.master.cubature.cubature_tables import dim3_elm1
        if (n < len(dim3_elm1)) and not (force_mk):
            x = array(dim3_elm1[n])
        else:
            x = array(mk_cube_rule(n, dim, elm))

    elif (dim == 3) and (elm == 2):
        from src.master.cubature.cubature_tables import dim3_elm2
        if (n < len(dim3_elm2)) and not (force_mk):
            x = array(dim3_elm2[n])
        else:
            x = array(mk_cube_rule(n, dim, elm))
    return x[:, :dim], x[:, dim]
