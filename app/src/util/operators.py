# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:49:15 2012

@author: empeeu
"""


import scipy as sp
from scipy import sparse
import scipy.linalg as splinalg
from scipy.sparse import linalg as sparselinalg
import copy
from time import time
from src.pyutil import unique
from src.master.mk_basis import mk_ed_orient
from src.master.mk_basis import get_nrml_jac
import pdb, traceback

class Integrate_z:
    """ This class creates the necessary data-structures/matrices to integrate
    a field in the z-direction, that is, the dim-1 direction.

       def __init__(sol):
       def __call__(self, sol, field):
    """
    def __init__(self, sol, top2bot=True):
        """Here we make the elemental integration matrices

        @param sol (\c object) The solution data-structure

         @param top2bot (\c bool) Flag which indicates the direction of the
            integration.
            If True (Default), integration will start with the top element.
            If False, it will start from the bottom element.

        @note The starting order determines how the boundary condition will be
            determined. That is, if top2bot, then whatever boundary condition is
            prescribed at the surface, it will be exactly satisfied.
        """
        ##Set the top-to-bottom preferences
        self.top2bot = top2bot

        ##Tolerance for comparisons to zero
        self.TOL = 1e-10

        ##Save some useful numbers for indexing:
        self.nb = sol.nb

        ##Calculate the element type offset
        self.elm_offset = [nb * n_elm \
            for nb, n_elm in zip(sol.nb, sol.n_elm_type)]
        self.elm_offset = [0] + self.elm_offset[:-1]

        ##The total number of edge degrees of freedom for each element type.
        #Also, save the indexes for the start and stop of each edge number
        #of dofs.
        self.index_ed = [sp.zeros((M.ne, 2)) for M in sol.master]
        self.tot_nb_ed = [0] * len(sol.u_elm_type)
        for i in range(len(sol.u_elm_type)):
            k = 0
            for eds in sol.master[i].basis.ed_type:
                j = (sol.u_ed_type == eds).nonzero()[0][0]
                self.index_ed[i][k, 0] = self.tot_nb_ed[i]
                self.index_ed[i][k, 1] = self.index_ed[i][k, 0] + sol.nb_ed[j]
                self.tot_nb_ed[i] += sol.nb_ed[j]
                k += 1

        #Create oft-used FEM operators
        mass_ed = [None for i in range(len(self.tot_nb_ed))]
        for i in range(len(self.tot_nb_ed)):
            mass_ed[i] = sp.zeros((self.tot_nb_ed[i], self.tot_nb_ed[i]))
            ns = 0
            for eds in sol.master[i].basis.ed_type:
                j = (sol.u_ed_type == eds).nonzero()[0][0]
                ne = ns + sol.nb_ed[j]
                mass_ed[i][ns:ne, ns:ne] = sol.master_ed[j].M
                ns = ne

        self.mass_ed = mass_ed

        #Build the element-local matrix inverses
        tic = time()
        print "Making the global A matrix for z-integration.",
        ##The global A matrix which links elements through the boundary
        #boundary conditions.
        self.A = None
        ##The global righ-hand-side matrix, which adds the top/bottom boundary
        #data to the solution. This is not always required or desired. For e.g.,
        #if calculating \f$\int_z^0 u = u(0) - u(z)\f$, the standard function has
        #assumed that u(0) == 0. When using the global rhs matrix
        #(Setting addbc=Tru) in the __call__), u(0) is taken directly from the
        #3d field supplied. If the boundary data does not come from the 3D field
        #supplied, it has to be added after the fact (using Add2Dto3D for e.g.)
        #The later situation happens when calculating
        #\f$w = \int_z^0 \nabla \cdot u \f$ since here
        #\f$w(0) = \frac{\partial \eta}{\partial t}\f$, which is a 2D field.
        self.RHS = None
        self.A, self.RHS = self.mk_A(sol)
        print "DONE in", time()-tic
        self.A_factored = None

        ##To calculate the edge values, the solution from the left and right
        #edges need to be combined using the coefficients in this matrix. The
        #value of the coefficients sum to 1, and will depend on top2bot
        self.lr_coeffs = [0.5 * sp.ones((ne, 2)) for ne in sol.n_ed_type]
        for i in range(len(self.lr_coeffs)):
            self.lr_coeffs[i][sol.ids_exterior_ed[i], :] = 0
            ids_l = (sol.nrml[i][0, sol.dim-1, :] * (1 - 2 * self.top2bot) > 0) \
                & (abs(sol.nrml[i][0, sol.dim-1, :]) > self.TOL)
            ids_r = (sol.nrml[i][0, sol.dim-1, :] * (1 - 2 * self.top2bot) < 0) \
                & (abs(sol.nrml[i][0, sol.dim-1, :]) > self.TOL)
            self.lr_coeffs[i][ids_l, 0] = 1
            self.lr_coeffs[i][ids_l, 1] = 0
            self.lr_coeffs[i][ids_r, 0] = 0
            self.lr_coeffs[i][ids_r, 1] = 1
            self.lr_coeffs[i][sol.ids_exterior_ed[i], 1] = 0


    def __call__(self, sol, field, addbc=False):
        """Integrate the field over the last dimension

        @param sol (\c object) The solution data-structure

        @param field (\c float) A list of numpy arrays of the field on the
            element space. Each list is for an element type.

        @param addbc (\c bool) Flag to add boundary conditions to the
            integration (Does not really make sense)
        """
        #Reshape the rhs and boundary data vectors if needed
        #Also Figure out how many unique boundary condition types we have
        isvector = False
        shp = field[0].shape
        n_flds = shp[1]
        if len(shp) == 4:
            isvector = True
            n_flds = shp[2] * shp[1]


        field = [r.swapaxes(-1, -2).reshape(r.shape[0] * r.shape[2], n_flds, \
            order='F') for r in field]

        #Now stack the different element types
        tmp = field.pop(0)
        for fs in field:
            tmp = sp.concatenate((tmp, fs))

        #Add the boundary conditions (which are already present in the given 3D field)
        #print "adding bcs", addbc
        if addbc:
            #print "adding bcs"
            tmp -= self.RHS * tmp


        #Negate the rhs if top2bot (WHY? WHY DO I NEED TO DO THIS TO GET IT RIGHT?)
        if self.top2bot:
            tmp = -tmp

        #Integrate using the matrix:
        tmp = self.solve(sol, tmp)

        #Now re-shape and re-distribute the solution
        tmp = [tmp[eo : eo + nb * ne, :]\
            .reshape(nb, ne, n_flds, order='F').swapaxes(1, 2)\
            for eo, nb, ne in zip(self.elm_offset, sol.nb, sol.n_elm_type)]
        #Final reshaping if required
        if isvector:
            tmp = [tmp.reshape[nb, shp[1], shp[2], ne]\
                for nb, ne in zip(self.nb, self.n_elm_type)]

        return tmp

    def solve(self, sol, field):
        """Integrates the field in the z-direction (or dim-1 direction).
        The base class uses a direct solver. To
        change this, create a new class, inhereting from this one, and change
        or update this function only.

        @param sol (\c object) The solution data-structure

        @param field (\c float) The field to be integrated

        @retval integrate_z (\c float) The z-integral of the given field
        """

        if self.A_factored == None:
            #This is the first time we're solving this system, so for the first
            #time we have to factorize the matrix
            tic = time()
            print "Factoring (LU-decomposition) of the global HDG Matrix ... ",
            self.A_factored = sparselinalg.factorized(self.A)
            print "DONE in", time()-tic

        n_flds = field.shape[1]
        integrate_z = sp.zeros_like(field)
        for i in xrange(n_flds):
            tic = time()
            #print "Solving HDG Matrix ... ",
            integrate_z[:, i] = self.A_factored(field[:, i].ravel())
            #print "DONE in", time()-tic

        return integrate_z

    def mk_locA(self, sol, jac, jac_fact, jac_ed_on_elm, nrml, elm_type):
        """Makes the unique elemental matrix -- this is made in the real
        coordinate space, so it changes based on the jacobians etc.

        @param sol (\c object) The solution data-structure

        @param jac (\c float) The element-local volume integral Jacobians

        @param jac_fact (\c float) The geometric factors
            @see master.mk_jacobian_nodal.jacobian

        @param jac_ed_on_elm (\c float) The edge-local edge integral Jacobians
            formatted on the element. @see sol.get_ed2elm_array

        @param nrml (\c float) The edge-local normal vector, with proper sign,
            formatted on the element. @see sol.get_ed2elm_array

        @param elm_type (\c int) The type of the current element

        @retval locA (\c float) The element-local diffusion matrix -- excluding
            the lambda-space edge integrals (assumed to be on rhs).
        """

        #shorter variable names
        master = sol.master[elm_type]
        L = master.L
        mass_ed = self.mass_ed[elm_type]
        nb = master.nb
        dim = master.dim

        #Initialize locA
        locA = sp.zeros((nb, nb))

        #Build the unique elemental operators
        M = splinalg.inv(sp.dot(master.M, sp.diag(jac)))
        B = sp.zeros((nb, nb))
        i  = dim - 1
        for j in range(dim):
            #Note, the derivative matrix is in the weak form
            B -= sp.dot(master.K[j], sp.diag(jac_fact[:, j, i] * jac))

        #Build the edge-integral operators
        nz = nrml[:, -1]
        #The (1-2 *top2bot) adds a negative sign if top2bot is false, in which
        #case our rule for the normal direction has to be reversed
        if (abs(nz[0]) > self.TOL):
            E_in = sp.dot(mass_ed, \
                sp.diag(jac_ed_on_elm * nz * ((nz * (1 - 2 * self.top2bot)) > 0)))
        else:
            E_in = sp.zeros(mass_ed.shape)
        #Note the L.T at the end, which is correct
        locA = sp.dot(M, B + sp.dot(L, sp.dot(E_in, L.T)))

        #And return
        return locA

    def mk_A(self, sol):
        """Makes the local real-space matrix inverses

        @param sol (\c object) The solution data-structure

        @retval locAinv (\c float) A list of the element-local diffusion
            matrices, pre-inverted. locAinv[i][j] contains the inverse matrix
            of the jth element of type i

        @note The inverse is calculated using sp.linalg.inv -- that is, sp.inv
        is used for speed instead of the pseudo-inverse .pinv, which could be
        used for stability and accuracy, but is about 10x more expensive.
        """
        #shorter variable names
        dim = sol.dim

        #Get the edge jacobians formatted on the element:
        jac_ed2elm = sol.get_ed2elm_array(sol.jac_ed)

        #Get the edge normals formatted on the element:
        nrml_ed2elm = sol.get_ed2elm_array(sol.nrml, negate_right=True)

        #Initialize the data and index arrays used to create the matrix
        #(We do the data allocation here for speed)
        tot_data = 0
        for tne, te, ne in zip(sol.nb, sol.n_elm_type, sol.master):
            #Note, this is overkill by a good amount, since only the nb_ed
            #columns will be used in the off-diagonal entries
            tot_data += tne**2 * te * (1 + ne.ne)

        #Initialize the list for storing the elements of the global matrix
        data = sp.zeros(tot_data, dtype=float)
        row = sp.zeros(data.shape, dtype=int)
        col = sp.zeros(data.shape, dtype=int)

        data_bc = sp.zeros(sol.mesh.ids_exterior_ed.sum() * (max(sol.nb) ** 2),\
            dtype=float)
        row_bc = sp.zeros(data_bc.shape, dtype=int)
        col_bc = sp.zeros(data_bc.shape, dtype=int)


        #index used to keep track where we are in data, row, and col
        index = [0, 0]
        index_bc = [0, 0]
        #Loop over every element
        for k in xrange(sol.n_elm):
            elm_type = sol.mesh.elm_type[k]

            #Get the element-type id number from the global element number
            j = sol.glob2type_elm[k]

            elm_ids_in_matrix = self._get_glob_dof_index(j, elm_type)
            COL_id, ROW_id = \
                sp.meshgrid(range(sol.nb[elm_type]), range(sol.nb[elm_type]))
            COL_id += elm_ids_in_matrix[0]
            ROW_id += elm_ids_in_matrix[0]

            #get the local jacobians and geometric factors
            jac = sol.jac[elm_type][:, j]
            jac_ed_on_elm = jac_ed2elm[elm_type][:, j]
            jac_fact = sol.jac_factor[elm_type][:, :, :, j]
            nrml = nrml_ed2elm[elm_type][:, :, j]

            #Build the element-specific matrix
            locA = self.mk_locA(sol, \
                jac, jac_fact, jac_ed_on_elm, nrml, elm_type=elm_type)

            #Add these contributions to the data, row and col
            #structures
            index[1] += self.nb[elm_type]**2
            data[index[0]:index[1]] = locA.ravel()
            col[index[0]:index[1]] = COL_id.ravel()
            row[index[0]:index[1]] = ROW_id.ravel()
            index[0] = index[1]

        #Next we have to add the off-diagonal entries.
        #For this it is simpler to loop over the edges
        for k in xrange(sol.n_ed):
            #Get the local type and type-number
            ed_type = sol.mesh.ed_type[k]

            #Get the element-type id number from the global element num
            k_loc = sol.glob2type_ed[k]

            #Now, figure out which element number is 'beneath' the other
            #The one element will be integrated before the next, as such the
            #boundary condition calculated from the first element will apply to
            #the second,
            #Where 'first' is 'beneath' 'second', and 'beneath' depends on the
            #direction of the integration i.e. depends on top2bot
            nrml_z = sol.nrml[ed_type][:, -1, k_loc]
            el_num = [sol.mesh.ed2ed[k, 0], sol.mesh.ed2ed[k, 2]]
            loc_ed_num = [sol.mesh.ed2ed[k, 1], sol.mesh.ed2ed[k, 3]]
            #If the outward-pointing normal is pointing down, then the left-edge
            #holds the boundary condition UNLESS the right edge is a boundary edge
            if nrml_z[0] * (1 - 2 * self.top2bot) > 0:
                lr = 0 #left is above
            else:
                lr = 1 #right is above
            #This checks the boundary condition. If true, we continue
            #Otherwise, we're done. BCs are handled on the right-hand side,
            #and the edge-terms on the element are already taken care of
            if (el_num[1] >= 0) and (abs(nrml_z[0]) > self.TOL):
                jac_ed = sol.jac_ed[ed_type][:, k_loc]
                M_ed = sol.master_ed[ed_type].M

                #Calculate the integral on the edge space
                contrib = sp.dot(M_ed, sp.diag(nrml_z * jac_ed))

                #Now we have to figure out which bases this corresponds to in
                #the volume space, and then arrange appropriately
                ed_orient_id = [None, None]
                glob_ed_verts = sol.mesh.ed2ed[k, 4:]
                elm_type = [sol.mesh.elm_type[el_num[0]],\
                    sol.mesh.elm_type[el_num[1]]]
                for ii in range(2):
                    elm_num_loc = sol.glob2type_elm[el_num[ii]]

                    #Find the element to edge orientation
                    loc_ed_ids = \
                        sol.master[elm_type[ii]].basis.ids_ed[loc_ed_num[ii]]

                    glob_elm_verts = sol.mesh.elm[el_num[ii],\
                        loc_ed_ids].ravel()
                    #For periodic meshes, we have to be careful, since these
                    #vertex numbers may not match the edge numbers.
                    if sol.mesh.vertmap != None:
                        if not any(glob_elm_verts[0] == glob_ed_verts):
                            glob_elm_verts = sol.mesh.vertmap[glob_elm_verts]

                    #Here we compare the global element numbering in the volume to that
                    #of the edge. This returns a orientation integer value, which can
                    #be used in conjunction with master[elm_type].ed_orient
                    ed_orient = sol._get_orient_num(\
                        [(glob_elm_verts[0] == glob_ed_verts).nonzero()[0][0], \
                        (glob_elm_verts[1] == glob_ed_verts).nonzero()[0][0]], \
                        one_num=True)

                    ed_orient_id[ii] = sol.master[elm_type[ii]]\
                        .ed_orient[loc_ed_num[ii]][ed_orient]

                #Now that we know the column id, we calculate the integral that
                #needs to be added to the other element
                rl = not lr

                #Make the correct contribution matrix, with the bases and test
                #functions correctly aligned
                contrib_full = sp.zeros((sol.nb[elm_type[rl]], \
                    sol.nb[elm_type[lr]]))
                contrib_full[sp.ix_(ed_orient_id[rl], ed_orient_id[lr])]\
                    = contrib

                #Get the element-type id number from the global element number
                elm_num_loc = sol.glob2type_elm[el_num[rl]]

                #Get the matrix operators to complete the calculation of the edge
                #contribution
                M = splinalg.inv(\
                    sp.dot(sol.master[elm_type[rl]].M, \
                    sp.diag(sol.jac[elm_type[rl]][:, elm_num_loc])))

                contrib_loc = (1 - 2 * rl) * sp.dot(M, contrib_full)

                #And finally, add this to the appropriate row/columns of the matrices
                COL_id, ROW_id = \
                    sp.meshgrid(range(sol.nb[elm_type[lr]]), \
                    range(sol.nb[elm_type[rl]]))
                elm_ids_in_matrix = self._get_glob_dof_index(\
                    sol.glob2type_elm[el_num[lr]], elm_type[lr])
                COL_id += elm_ids_in_matrix[0]
                elm_ids_in_matrix = self._get_glob_dof_index(elm_num_loc,\
                    elm_type[rl])
                ROW_id += elm_ids_in_matrix[0]
                #Add these contributions to the data, row and col
                #structures
                #Now, to do the correct alignment of bases, if rl == 1, that is,
                #if the row ids belong to the RIGHT edge, the edge integral
                #calculated above is in the wrong order, so we have to flip
                #the rowid of the matrix
                #if rl:
                #    ROW_id = sp.flipud(ROW_id)

                index[1] += self.nb[elm_type[rl]] * self.nb[elm_type[lr]]
                data[index[0]:index[1]] = contrib_loc.ravel()
                col[index[0]:index[1]] = COL_id.ravel()
                row[index[0]:index[1]] = ROW_id.ravel()
                #test = col[index[0]:index[1]]
                #test2 = row[index[0]:index[1]]
                #print (test.min() > test2.min()) and (test.max() < test2.max())
                index[0] = index[1]
            #Else, if this is a boundary node and we need boundary conditions
            #from it
            elif (el_num[1] < 0) and (abs(nrml_z[0]) > self.TOL) and lr == 1:
                jac_ed = sol.jac_ed[ed_type][:, k_loc]
                M_ed = sol.master_ed[ed_type].M

                #Calculate the integral on the edge space
                contrib = sp.dot(M_ed, sp.diag(nrml_z * jac_ed))

                #Now we have to figure out which bases this corresponds to in
                #the volume space, and then arrange appropriately
                ed_orient_id = [None, None]
                glob_ed_verts = sol.mesh.ed2ed[k, 4:]
                elm_type = [sol.mesh.elm_type[el_num[0]],\
                    sol.mesh.elm_type[el_num[0]]]
                #Note the zero in above line, needed because the right-element
                #is imaginary, i.e. it gives the boundary condition number
                for ii in range(1):
                    elm_num_loc = sol.glob2type_elm[el_num[ii]]

                    #Find the element to edge orientation
                    loc_ed_ids = \
                        sol.master[elm_type[ii]].basis.ids_ed[loc_ed_num[ii]]

                    glob_elm_verts = sol.mesh.elm[el_num[ii],\
                        loc_ed_ids].ravel()
                    #For periodic meshes, we have to be careful, since these
                    #vertex number may not match the edge numbers.
                    if sol.mesh.vertmap != None:
                        if not any(glob_elm_verts[0] == glob_ed_verts):
                            glob_elm_verts = sol.mesh.vertmap[glob_elm_verts]

                    #Here we compare the global element numbering in the volume to that
                    #of the edge. This returns a orientation integer value, which can
                    #be used in conjunction with master[elm_type].ed_orient
                    ed_orient = sol._get_orient_num(\
                        [(glob_elm_verts[0] == glob_ed_verts).nonzero()[0][0], \
                        (glob_elm_verts[1] == glob_ed_verts).nonzero()[0][0]], \
                        one_num=True)

                    ed_orient_id[ii] = sol.master[elm_type[ii]]\
                        .ed_orient[loc_ed_num[ii]][ed_orient]

                #Now that we know the column id, we calculate the integral that
                #needs to be added to the other element
                #For the boundary data, this actually comes from the interior
                #of the same element, so rl = 0
                rl = 0
                lr = 0

                #Make the correct contribution matrix, with the bases and test
                #functions correctly aligned
                contrib_full = sp.zeros((sol.nb[elm_type[rl]], \
                    sol.nb[elm_type[lr]]))
                contrib_full[sp.ix_(ed_orient_id[rl], ed_orient_id[lr])]\
                    = contrib

                #Get the element-type id number from the global element number
                elm_num_loc = sol.glob2type_elm[el_num[rl]]

                #Get the matrix operators to complete the calculation of the edge
                #contribution
                M = splinalg.inv(\
                    sp.dot(sol.master[elm_type[rl]].M, \
                    sp.diag(sol.jac[elm_type[rl]][:, elm_num_loc])))

                contrib_loc = (1. - 2. * rl) * sp.dot(M, contrib_full)

                #And finally, add this to the appropriate row/columns of the matrices
                COL_id, ROW_id = \
                    sp.meshgrid(range(sol.nb[elm_type[lr]]), \
                    range(sol.nb[elm_type[rl]]))
                elm_ids_in_matrix = self._get_glob_dof_index(\
                    sol.glob2type_elm[el_num[lr]], elm_type[lr])
                COL_id += elm_ids_in_matrix[0]
                elm_ids_in_matrix = self._get_glob_dof_index(elm_num_loc,\
                    elm_type[rl])
                ROW_id += elm_ids_in_matrix[0]
                #Add these contributions to the data, row and col
                #structures
                #Now, to do the correct alignment of bases, if rl == 1, that is,
                #if the row ids belong to the RIGHT edge, the edge integral
                #calculated above is in the wrong order, so we have to flip
                #the rowid of the matrix
                #if rl:
                #    ROW_id = sp.flipud(ROW_id)

                index_bc[1] += self.nb[elm_type[rl]] * self.nb[elm_type[lr]]
                data_bc[index_bc[0]:index_bc[1]] = contrib_loc.ravel()
                col_bc[index_bc[0]:index_bc[1]] = COL_id.ravel()
                row_bc[index_bc[0]:index_bc[1]] = ROW_id.ravel()
                index_bc[0] = index_bc[1]

                #index[1] += self.nb[elm_type[rl]] * self.nb[elm_type[lr]]
                #data[index[0]:index[1]] = contrib_loc.ravel()
                #col[index[0]:index[1]] = COL_id.ravel()
                #row[index[0]:index[1]] = ROW_id.ravel()
                #index[0] = index[1]

        #Create the sparse matrix, bcs are already correct
        A = sparse.csc_matrix((data, (row, col)),\
            shape=(sol.n_dofs_elm_1, sol.n_dofs_elm_1))
        A.eliminate_zeros()
        RHS = sparse.csc_matrix((data_bc, (row_bc, col_bc)),\
            shape=(sol.n_dofs_elm_1, sol.n_dofs_elm_1))
        RHS.eliminate_zeros()

        return A, RHS

    def _get_glob_dof_index(self, j, elm_type):
        start = self.elm_offset[elm_type] + j * self.nb[elm_type]
        return [start, start + self.nb[elm_type]]

    def get_elm2edhat(self, sol, field):
        field_ed = sol.get_elm2ed_array(field)
        for i in range(len(field_ed)):
            for j in range(field_ed[0][0].shape[0]):
                for k in range(field_ed[0][0].shape[1]):
                    field_ed[i][0][j, k, :] = \
                        field_ed[i][0][j, k, :] * self.lr_coeffs[i][:, 0] \
                        + field_ed[i][1][j, k, :] * self.lr_coeffs[i][:, 1]
            field_ed[i] = field_ed[i][0]

        return field_ed

    def get_tot_integral(self, sol, field, addbc=False):
        """ONLY WORKS FOR EXTRUDED MESHES!"""

        allzint = self.__call__(sol, field, addbc)
        return get_tot_integral(sol, allzint, self.top2bot)

def get_tot_integral(sol, allzint, top2bot=True):
    """ONLY WORKS FOR EXTRUDED MESHES!"""
    ##Tolerance for comparisons to zero
    TOL = 1e-10
    dim = sol.dim
    try:
        #If not an extruded mesh, we have to try it
        #the longer way
        levels = sol.mesh.levels
        if top2bot: #Then take the bottom elements
            elm_ids = sp.arange(levels-1, sol.n_elm, levels)
            stid = 0
            in_elm_ids = [sp.arange( stid * nb / (sol.n + 1), \
                (stid + 1) * nb / (sol.n + 1)) for nb in sol.nb]
        else: #Take the top elements
            elm_ids = sp.arange(0, sol.n_elm, levels)
            stid = sol.n
            in_elm_ids = [sp.arange( stid * nb / (sol.n + 1), \
                (stid + 1) * nb / (sol.n + 1)) for nb in sol.nb]

    except:
        #Get the average z-normals for the boundary faces
        bnd_nz = [n[:, dim-1, ids].sum(0) / nb \
            for n, nb, ids in zip(sol.nrml, sol.nb_ed, sol.ids_exterior_ed)]

        #This sign has to be the same as the condition above because we want the
        #element ABOVE the boundary if top2bot == True
        if top2bot:
            ids = [abs(bnz + 1) < TOL\
                for bnz in bnd_nz ]
        else:
            ids = [abs(bnz - 1) < TOL\
                for bnz in bnd_nz ]

        #Now we have to convert the local element ids to the global element ids
        glob_ed_id = sp.zeros(sol.n_ed, dtype=bool)
        for i in range(len(ids)):
            glob_ed_id[((sol.mesh.ed_type == i) & (sol.mesh.ids_exterior_ed))\
                .nonzero()[0][ids[i]]] = True

        #Get the global element numbers from where we will extract the 2D field
        elm_ids = sol.mesh.ed2ed[glob_ed_id, 0]

        #This approach has one flaw, in that we don't know what the
        #local element ids are, so this will NOT work for triangular meshes
        #To make it work for triangles, you'd have to loop through each of
        #the bottom elements, and only take the boundary edge values
        if top2bot: #Then take the bottom elements
            stid = 0
            in_elm_ids = [sp.arange( stid * nb / (sol.n + 1), \
                (stid + 1) * nb / (sol.n + 1)) for nb in sol.nb]
        else: #Take the top elements
            stid = sol.n
            in_elm_ids = [sp.arange( stid * nb / (sol.n + 1), \
                (stid + 1) * nb / (sol.n + 1)) for nb in sol.nb]

    #Get the local element id numbers
    elm_ids_loc = sol.glob2type_elm[elm_ids]

    #Separate into type
    elm_ids_loc = [elm_ids_loc[sol.mesh.elm_type[elm_ids] == i] \
        for i in range(len(sol.nb))]

    #Finally, strip out the 2D data, and return in an array
    if len(allzint[0].shape) == 3:
        fnlzonly = [az[:, :, eid] \
           for az, eid in zip (allzint, elm_ids_loc)]
        fnlzonly = [az[in_ids, :, :] \
           for az, in_ids in zip (fnlzonly, in_elm_ids)]
    elif len(allzint[0].shape) == 4:
        fnlzonly = [az[:, :, :, eid] \
           for az, eid in zip (allzint, elm_ids_loc)]
        fnlzonly = [az[in_ids, :, :, :] \
           for az, in_ids in zip (fnlzonly, in_elm_ids)]

    return fnlzonly

def get_tot_integral_ed(sol, allzint_ed, topbotbcid, bot=False):
    """ONLY WORKS FOR EXTRUDED MESHES!"""
    #Find the boundary ids for this edge
    out2D = []
    vectrue = len(allzint_ed[0].shape) == 4

    #Calculate the element ordering
    mv = []
    if bot:
        ed_ids = [[range(len(nb)) for nb in M.basis.nodal_ed_ids] \
            for M in sol.master]

        index = [6, 6]
        index_type = sp.array([0, 1])

        for i in range(len(ed_ids)):
            mv.append(mk_ed_orient(ed_ids[i],\
                sol.master[i].basis.ed_type, sol.n, sol.dim)\
                [0][index[(sol.u_ed_type[i] == index_type).nonzero()[0][0]]])
    else:
        ed_ids = [[range(len(nb)) for nb in M.basis.nodal_ed_ids] \
            for M in sol.master]
        for i in range(len(sol.nb)):
            mv.append(ed_ids[i][0])


    for j in range(len(sol.nb)):
        ids_ex = sol.ids_exterior_ed[j]
        ids1 = sp.zeros(ids_ex.shape, dtype=bool)
        ids2 = (sol.bcid[j] == topbotbcid[bot]).ravel()
        ids1[ids_ex] = ids2
        #print j, len(ids1), allzint_ed[j].shape
        if vectrue:
            out2D.append(allzint_ed[j][:, :, :, ids1][mv[j], :, :, :])
        else:
            out2D.append(allzint_ed[j][:, :, ids1][mv[j], :, :])

    return out2D


class Add2Dto3D:
    """ONLY WORKS FOR EXTRUDED MESHES!
    def __init__(self, sol, sol2D):
    def __call__(self, field3D, field2D, mul=1):
    """
    def __init__(self, sol, sol2D):
        self.levels = sol.mesh.levels
        n = sol.n
        self.copies = [sp.arange(nb).repeat(n+1, 0)\
            .reshape(nb, n+1).reshape(nb*(n+1), order='F') for nb in sol2D.nb]
    def __call__(self, field3D, field2D, mul=1):
        shp = field2D[0].shape
        for i in range(len(field3D)):
            for j in range(self.levels):
                if len(shp) == 3:
                    field3D[i][:, :, j::self.levels] \
                        += mul * field2D[i][self.copies[i], :, :]
                elif len(shp) == 4:
                    field3D[i][:, :shp[1], :, j::self.levels] \
                        += mul * field2D[i][self.copies[i], :, :, :]
        return field3D

class Add2Dto3D_ed:
    """ONLY WORKS FOR EXTRUDED MESHES!
    def __init__(self, sol, sol2D):
    def __call__(self, field3D, field2D, field2D_ed, sol2D, mul=1.0):

        """
    def __init__(self, sol, sol2D):
        self.levels = sol.mesh.levels
        n = sol.n
        self.copies = [sp.arange(nb).repeat(n+1, 0)\
            .reshape(nb, n+1).reshape(nb*(n+1), order='F') for nb in sol2D.nb_ed]
        #The segments member helps to deliminate the 3D edge field, so that the
        #correct data can be added -- the delimination is STRONGLY dependent on
        #how the edges are extruded. This is copied from the mesh module comments:
        #The edge ids are divided into 4 major categories:
        #1. Interior vertical edges (0:n_ed_in * levels)
        #2. Interior horizontal edges
        #   <start> : len(_elm2D) * (levels - 1) + <start>
        #3. Boundary vertical edges
        #   <start> : len(ed2ed - n_ed_in) + <start>
        #4. Boundary horizontal edges
        #   <start> : len(_elm2D) * 2 + <start>
        #
        # and each category should be dealt with separately

        #Figure out which element type is the square type, there may or may not
        #be square faces in the mesh
        sqr = (sol2D.u_elm_type == 1).nonzero()[0]
        if sqr.shape[0] == 0:
            #If there are no horizontal square edges in the 2D mesh, then there
            #are ONLY vertical meshes which are square in the 3D mesh, as such,
            #the number in the specific 3D edge-type for squares will be
            #continuous
            sqr = None
            self.segs_v = [0, sol2D.ids_interior_ed[0].sum() * self.levels,\
                0,
                sol2D.ids_exterior_ed[0].sum() * self.levels]
        else:
            #If there are horizontal square edges in the 2D mesh, we need to
            #follow the convention used in the mesh module (copied above). That
            #means that we will have some offset in how the edges are numbered
            #in the 3D mesh.
            sqr = sqr[0]
            self.segs_v = [0, sol2D.ids_interior_ed[0].sum() * self.levels,\
                sol2D.n_elm_type[sqr] * (self.levels - 1),
                sol2D.ids_exterior_ed[0].sum() * self.levels]
        #Format for the vertical faces' segment member is start, end, start, end
        #(since it is not continuously indexed)

        for i in range(len(self.segs_v) - 1):
            self.segs_v[i + 1] += self.segs_v[i]

        #Format for the horizontal faces' segment member is:
        #start, end, start, end (since there might be a jump
        #in the middle for square meshes)
        self.segs_h = [[0, n_type * (self.levels - 1),\
            n_type * (self.levels - 1), n_type * (self.levels + 1)] \
            for n_type in sol2D.n_elm_type]

        #There WILL be square faces in the 3D mesh, but there may or may not be
        #horizontal square faces
        self.square = (sol.u_ed_type == 1).nonzero()[0][0]

        #Note, second condition is to account for meshes 1 element high
        if sqr != None and (self.levels - 1 != 0):
            self.segs_h[self.square] = \
                [self.segs_v[1], self.segs_h[self.square][1] + self.segs_v[1],\
                self.segs_v[3],\
                self.segs_v[3] + self.segs_h[self.square][1] / (self.levels - 1) * 2]
        self.u_ed_type = sol.u_ed_type

        #For the extruded meshes, the bottom element is renumbered so that the
        #normal will be outward pointing. Hence, the ordering will not be the
        #same as in the 2D mesh, so we have to figure out how to move the
        #move the vertices for each element type
        self.mv = []
        #ed_ids = [[range(len(nb)-1,-1,-1) for nb in M.basis.nodal_ed_ids] \
        #    for M in sol.master]
        ed_ids = [[range(len(nb)) for nb in M.basis.nodal_ed_ids] \
            for M in sol.master]

        index = [6, 6]
        index_type = sp.array([0, 1])

        for i in range(len(ed_ids)):
            self.mv.append(mk_ed_orient(ed_ids[i],\
                sol.master[i].basis.ed_type, sol.n, sol.dim)\
                    [0][index[(sol.u_ed_type[i] == index_type).nonzero()[0][0]]])


    def __call__(self, field3D_ed, field2D, field2D_ed, sol2D, mul=1.0):
        shp = list(copy.copy(field3D_ed[0].shape))
        if len(shp) == 4:
            shp[1] -= 1
        segs = self.segs_v

        #Figure out which type of edge is a square:
        i = self.square
        #The boundary edges should not require a correction if they
        #Are dirichlet, however, we will update all the boundary edges
        #regardless because -- very tricky -- it won't matter. The
        #correct boundary value will be used based on how hdg and
        #advect are implemented.
        for j in range(self.levels):
            #Add the interior vertical faces on the j'th level
            if len(shp) == 3: #Scalar field
                #Interior
                tmp = field2D_ed[0][:, :, sol2D.ids_interior_ed[0]]
                field3D_ed[i][:, :, (j + segs[0]):segs[1]:self.levels] \
                    += mul * tmp[self.copies[0], :, :]
                tmp = field2D_ed[0][:, :, sol2D.ids_exterior_ed[0]]
                #Exterior
                field3D_ed[i][:, :, \
                    (j + segs[2]):segs[3]:self.levels] \
                    += mul * tmp[self.copies[0], :, :]
            elif len(shp) == 4: #vector field
                #Interior
                tmp = field2D_ed[0][:, :, :, sol2D.ids_interior_ed[0]]
                field3D_ed[i][:, :shp[1], :, (j + segs[0]):segs[1]:self.levels] \
                    += mul * tmp[self.copies[0], :shp[1], :, :]
                #Exterior
                tmp = field2D_ed[0][:, :, :, sol2D.ids_exterior_ed[0]]
                field3D_ed[i][:, :shp[1], :, (j + segs[2]):segs[3]:self.levels] \
                    += mul * tmp[self.copies[0], :shp[1], :, :]

        #CM Now add the horizontal faces
        segs_h = self.segs_h
        mv = self.mv
        for i in range(len(field2D)):
            i_3d = (sol2D.u_elm_type[i] == self.u_ed_type).nonzero()[0][0]

            #Add the horizontal faces on the interior
            for j in range(self.levels - 1):
                if len(shp) == 3:
                    field3D_ed[i_3d][:, :, \
                        (segs_h[i_3d][0]+j):segs_h[i_3d][1]:(self.levels-1)] \
                        += mul * field2D[i]
                elif len(shp) == 4:
                    field3D_ed[i_3d][:, :shp[1], :,\
                        (segs_h[i_3d][0]+j):segs_h[i_3d][1]:(self.levels - 1)] \
                        += mul * field2D[i][:, :shp[1], :, :]

            #Add the horizontal faces on the exterior
            #CM In other words, surface and bottom faces
            if len(shp) == 3:
                field3D_ed[i_3d][:, :, segs_h[i_3d][2]:segs_h[i_3d][3]:2] += \
                    mul * field2D[i]
                field3D_ed[i_3d][:, :, (segs_h[i_3d][2] + 1):segs_h[i_3d][3]:2] += \
                    mul * field2D[i][mv[i], :, :]
            elif len(shp) == 4:
                # CM First add the surface (top) face values
                field3D_ed[i_3d][:, :shp[1], :, segs_h[i_3d][2]:segs_h[i_3d][3]:2] \
                    += mul * field2D[i][:, :shp[1], :, :]
                #CM Then add the bottom face values
                #CM BUG HERE??
#                print "CM: Inside Add2Dto3D_ed.__call__():"
#                print "Checking bottom edge values"
#                beids2 = [27, 28, 35, 36]
#                beids3 = [1655, 1657, 1671, 1673]
#                print "field3D_ed = ", field3D_ed[i_3d][:, 1, 0, beids3]
#                print "field2D    = ", field2D[i][:, 1, 0, beids2]

                field3D_ed[i_3d][:, :shp[1], :, (segs_h[i_3d][2] + 1):segs_h[i_3d][3]:2] \
                    += mul * field2D[i][mv[i], :shp[1], :, :]
                #CM Original code was:
                #field3D_ed[i_3d][mv[i], :shp[1], :, (segs_h[i_3d][2] + 1):segs_h[i_3d][3]:2] \
                #    += mul * field2D[i][:, :shp[1], :, :]

        return field3D_ed

def get_bcs_from_field_ed(sol, field_ed, bcids=None, field_ed_bcs=None, \
        negbcids=None):
    """get_bcs_from_field_ed(sol, field_ed, bcids=None, field_ed_bcs=None)

    Starting with all the edges in a field, return only the boundary
    conditions, formatted in the approprate data-structured to be used by
    common solver functions.

    @param sol (\c object) The solution data-structure

    @param field_ed (\c float) List of numpy arrays of the edged values

    @param bcids (\c list) [optional] List of ids which should be obtained
        from field_ed. ids not included in the list will return a value of
        '0' instead.

    @param field_ed_bcs (\c float) [optional] An already-formatted boundary
        condition output data structured. This is a list of numpy arrays.
        The data in this array will be overwritten for the id's specified.

    @param negbcids (\c list) List of boundary ids for which the boundary
        condition values should be negated. This happens, for example, for
        Neumann bcs at the bottom of the ocean where nz = -1

    @retval field_ed_bcs (\c float) Boundary conditions obtained from field_ed,
        and formatted in the approprate data-structured to be used by
        common solver functions.
    """
    if bcids == None:
        if len(field_ed[0].shape) == 3:
            field_ed_bcs = [fe[:, :, ids] \
                for fe, ids in zip(field_ed, sol.ids_exterior_ed)]
        elif len(field_ed[0].shape) == 4:
            field_ed_bcs = [fe[:, :, :, ids] \
                for fe, ids in zip(field_ed, sol.ids_exterior_ed)]
    else:
        shp = [sp.array(fld.shape) for fld in field_ed]
        if field_ed_bcs == None: #Initialize the outputs
            for s, ids in zip(shp, sol.ids_exterior_ed):
                s[-1] = ids.sum()
            field_ed_bcs = [sp.zeros(s) for s in shp]

        for i in range(len(shp)):
            ids_ex = sol.ids_exterior_ed[i]
            for bcid in bcids:
                ids2 = sp.zeros_like(ids_ex)
                ids1 = sol.bcid[i] == bcid
                ids2[ids_ex] = ids1
                if len(shp[0]) == 3:
                    field_ed_bcs[i][:, :, ids1] = field_ed[i][:, :, ids2]
                elif len(shp[0]) == 4:
                    field_ed_bcs[i][:, :, :, ids1] = field_ed[i][:, :, :, ids2]

    if negbcids != None:
        shp = [sp.array(fld.shape) for fld in field_ed]
        for i in range(len(shp)):
            ids_ex = sol.ids_exterior_ed[i]
            for bcid in negbcids:
                ids1 = sol.bcid[i] == bcid
                field_ed_bcs[i][:, :, ids1] = -field_ed_bcs[i][:, :, ids1]

    return field_ed_bcs

def get_bcs_from_field(sol, field, bcids=None, field_ed_bcs=None, negbcids=None):
    """get_bcs_from_field(sol, field, bcids=None, field_ed_bcs=None)

    Starting with all the volume elements in a field, return only the boundary
    conditions, formatted in the approprate data-structured to be used by
    common solver functions.

    @param sol (\c object) The solution data-structure

    @param field (\c float) List of numpy arrays of the volume values

    @param bcids (\c list) [optional] List of ids which should be obtained
        from field_ed. ids not included in the list will return a value of
        '0' instead.

    @param field_ed_bcs (\c float) [optional] An already-formatted boundary
        condition output data structured. This is a list of numpy arrays.
        The data in this array will be overwritten for the id's specified.

    @param negbcids (\c list) List of boundary ids for which the boundary
        condition values should be negated. This happens, for example, for
        Neumann bcs at the bottom of the ocean where nz = -1

    @retval field_ed_bcs (\c float) Boundary conditions obtained from field_ed,
        and formatted in the approprate data-structured to be used by
        common solver functions.

    @see get_bcs_from_field_ed
    """
    #Grab the edge values on the left only:
    field_ed = [fed[0] for fed in sol.get_elm2ed_array(field)]
    return get_bcs_from_field_ed(sol, field_ed, bcids, field_ed_bcs, negbcids)

def master_grad_elm(sol, field):
    """This function takes the weak gradient of the provided tracer field in the
    master element, and returns those. This is primarily a helper function for
    the grad and div functions. In other words, this takes the operation
    \f$(\mathbf q, \theta) = -(u, \nabla \cdot \theta)\f$
    (so, only the volume terms)

    @param sol (\c object) The solution data-structure

    @param field (\c float) List of numpy arrays of the field on the element
        (that is \f$u\f$). Note, this function only deals with tracer fields,
        that is len(field[i].shape) = 3.

    @retval field_grad (\c float) List of numpy arrays of the gradient of the
        field. Note, this function returns the gradient as:
        field_grad[i].shape = (nb, dim, n_fld, n_elm)

    @note The derivatives are taken without quadrature, that is, this is a
        quadrature-free implementation.
    """

#    print 'FIELD DATA:'
#    print field[0].shape
#    print field[1].shape

    #===================================#
    #     setup
    #===================================#
    #First, build the operators we need
    dim = sol.dim
    n_fld = field[0].shape[1]

    ##Element Derivative matrices (weak form)
    Ds = sol.Ds

    #===================================#
    #     computation
    #===================================#
    #initialize outputs
    grad = [sp.zeros((fld.shape[0], dim, n_fld, fld.shape[2])) for fld in field]
    ### DO THE Master INTEGRATIONS ###
    for i in range(len(sol.n_elm_type)):
        shapv = field[i].shape
        for j in range(dim):
            #The next line should be dominatingly computationally expensive:
            #Calculate the derivatives in the master element:
            grad[i][:, j, :, :] = (\
                sp.dot(Ds[i][j], field[i]\
                .reshape(shapv[0], n_fld * shapv[2], order='F'))\
                ).reshape(shapv[0], n_fld, shapv[2], order='F')

    return grad

def mk_ed_flux(sol, field, field_ed, grdim=None, gradflux=None):
    """Make the edge fluxes in x-y-z space for the gradient and divergence
    operatos.

    @param sol (\c object) The solution data-structure

    @param field (\c float) List of numpy arrays of the field on the element
        (that is \f$u\f$).

    @param field_ed (\c float) List of numpy arrays of the field on the edges
        of the elements (that is \f$\hat u\f$).

    @param grdim (\c list) This is an optional list of integers that contain the
        dimensions for which to take the gradient. That is, if grdim = [0, 2],
        only the x and z gradients will be returned. By default, all gradients
        are returned.

    @param gradflux (\c bool) Flag to choose between making fluxes for the
        gradient: F_ed[i][:, j, :, :] = \f$ \phi \hat n_{x_j} J_ed \f$
        or divergence: F_ed[i][:, :, :] = \f$\sum_j \phi_j \hat n_j J_ed\f$
        If not specified, gradient fluxes will be used for tracers and
        divergence fluxes will be used for vectors
    """
    #First do some input parsing
    if grdim is None:
        grdim = range(sol.dim)
    shape = [f.shape for f in field_ed]
    if gradflux == None:
        if len(shape[0]) == 3: #gradients
            gradflux = True
        elif len(shape[0]) == 4: #divergences
            gradflux = False
        else:
            print "Error in util.operators.mk_ed_flux, the flux is neither a"\
                + " vector nor a scalar."
    #error checking
    if gradflux and len(shape[0]) != 3:
       print "Error in util.operators.mk_ed_flux, the flux is not a scalar, "\
            + "but a gradient flux is requested."
    if (not gradflux) and (len(shape[0]) != 4):
       print "Error in util.operators.mk_ed_flux, the flux is not a vector, "\
            + "but a divergence flux is requested."

    #First, make the edge_flux matrix
    nrml_on_elm = sol.get_ed2elm_array(sol.nrml, negate_right=True)
    jac_ed_on_elm = sol.get_ed2elm_array(sol.jac_ed)
    f_ed = sol.get_ed2elm_array(field_ed)
    shape = [fed.shape for fed in f_ed]

    L = [m.L for m in sol.master]

    if gradflux:
        F_ed = [sp.zeros((s[0], len(grdim), s[1], s[2])) for s in shape]
        for i in range(len(F_ed)):
            for j in range(len(grdim)):
                for k in range(s[1]):
                    F_ed[i][:, j, k, :] = jac_ed_on_elm[i] \
                        * nrml_on_elm[i][:, grdim[j], :] \
                        * (f_ed[i][:, k, :] \
                            - sp.dot(L[i].T, field[i][:, k, :]))
    else:
        F_ed = [sp.zeros((s[0], s[2], s[3])) for s in shape]
        for i in range(len(F_ed)):
            for j in grdim:
                for k in range(s[2]):
                    F_ed[i][:, k, :] += jac_ed_on_elm[i] \
                        * nrml_on_elm[i][:, j, :] \
                        * (f_ed[i][:, j, k, :] \
                            - sp.dot(L[i].T, field[i][:, j, k, :]))

    return F_ed

def grad(sol, field, field_ed, F_ed=None, grdim=None,
         edgeflag=True, elmflag=True):
    """This function takes the strong gradient of the provided field in real
    space.
    \f$(\mathbf{q}, \theta) = (\nabla u, \theta)
        + \langle(\hat{u}-u)\mathbf{n},\theta\rangle\f$

    @param sol (\c object) The solution data-structure

    @param field (\c float) List of numpy arrays of the field on the element
        (that is \f$u\f$).

    @param field_ed (\c float) List of numpy arrays of the field on the edges
        of the elements (that is \f$\hat u\f$).

    @param F_ed (\c float) List of numpy arrays of the edge flux on the
        ed2elm space. This is basically jac_ed * nrml * (\hat u - u)
        @see mk_ed_flux. If not specified this array will be created.

    @param grdim (\c list) This is an optional list of integers that contain the
        dimensions for which to take the gradient. That is, if grdim = [0, 2],
        only the x and z gradients will be returned. By default, all gradients
        are returned.

    @param edgeflag (\c bool) Flag that includes (True) or excludes (False) the
        edge integral terms

    @param elmflag (\c bool) Flag that includes (True) or excludes (False) the
        element integral terms

    @retval field_grad (\c float) List of numpy arrays of the gradient of the
        field. Note, this function returns the gradient as:
        field_grad[i].shape = (nb, dim, n_fld, n_elm) if the provided field is a
        tracer. Or
        field_grad[i].shape = (nb, dim, dim, n_fld, n_elm) if the provided field
        is a vector

    @note The derivatives are taken without quadrature, that is, this is a
        quadrature-free implementation.
    """
    #First do some input parsing
    if grdim == None:
        grdim = range(sol.dim)

    #Now get shorter names
    lift = sol.lift

    #Now reshape the field and edges if necessary
    isvector = False
    if len(field[0].shape) == 4:
        isvector = True
        n_fld = field[0].shape[2]
        field = [f.reshape(f.shape[0], f.shape[1] * f.shape[2], f.shape[3], \
            order='F') for f in field]
        if edgeflag:
            field_ed = [f.reshape(f.shape[0], f.shape[1] * f.shape[2], f.shape[3], \
                order='F') for f in field_ed]
        v_fld = n_fld
        n_fld = field[0].shape[1]
    else:
        n_fld = field[0].shape[1]

    #Now find the weak volume derivatives in the master element
    if elmflag:
        grad_rst = master_grad_elm(sol, field)

    #And make the fluxes
    if edgeflag:
        if F_ed == None:
            F_ed = mk_ed_flux(sol, field, field_ed, grdim=grdim)
        F_ed = [sp.dot(lift, fed\
            .reshape(fed.shape[0], fed.shape[1] * fed.shape[2] * fed.shape[3]))\
            .reshape(lift.shape[0], fed.shape[1], fed.shape[2], fed.shape[3])\
            for lift, fed in zip(sol.lift, F_ed)]

    #combine these gradients using the appropriate factors to find the gradients
    #in real space
    #Initialize outputs
    grad_xy = [sp.zeros((f.shape[0], len(grdim), n_fld, f.shape[2]))\
        for f in field]
    for i in range(len(field)):
        jj = -1
        for j in grdim:
            jj += 1
            for l in range(n_fld):
                #Add the edge terms:
                if edgeflag:
                    grad_xy[i][:, jj, l, :] += F_ed[i][:, jj, l, :] / sol.jac[i]
                if elmflag:
                  for k in range(sol.dim):
                    #combine volume terms
                    grad_xy[i][:, jj, l, :] += \
                        grad_rst[i][:, k, l, :] * sol.jac_factor[i][:, k, j, :]

    #reshape outputs if necessary
    if isvector:
        grad_xy = [f.reshape(f.shape[0], len(grdim), sol.dim, v_fld, f.shape[3],\
            order='F')\
            for f in grad_xy]

    return grad_xy

def div(sol, field, field_ed, field_ed2elm=None, grdim=None, edgeflag=True,
        nozflag=False, interiorflag=True):
    """This function takes the strong divergence of the provided field in real
    space.
    \f$(q,\theta)=(\nabla\cdot\mathbf{u},\theta)
        +\langle(\hat{\mathbf{u}}-\mathbf{u})\cdot\mathbf{n},\theta\rangle\f$

    @param sol (\c object) The solution data-structure

    @param field (\c float) List of numpy arrays of the field on the element
        (that is \f$u\f$). Note, this HAS to be a vector field.

    @param field_ed (\c float) List of numpy arrays of the field on the edges
        of the elements (that is \f$\hat u\f$). This HAS to be a vector field.

    @param field_ed2elm (\c float) List of numpy arrays of the field edge
        values formatted on the element (that is \f$\hat u\f$ in the ed2elm
        space). If not specified, this array will be created from field_ed.
        If specified, field_ed will not be used.
        @see sol_nodal.Sol_Nodal.get_ed2elm_array

    @param grdim (\c list) This is an optional list of integers that contain the
        dimensions for which to take the divergence. That is, if grdim = [0, 2],
        only the sum of the x and z derivatives will be returned. By default,
        the full divergence is returned.

    @param edgeflag (\c bool) Flag to indicate whether or not to include the
        edge-penalty terms. True by default.

    @param nozflag (\c bool) Flag to indicate that no edge-penalty should
        occur for faces with non-zero z normal.

    @retval field_div (\c float) List of numpy arrays of the gradient of the
        field. Note, this function returns the divergence as:
        field_grad[i].shape = (nb, n_fld, n_elm)

    @note The derivatives are taken without quadrature, that is, this is a
        quadrature-free implementation.
    """

    #CM DEBUG
#    print "CM: div() parameter check:"
#    print "field_ed2elm?", field_ed2elm is None
#    print "grdim = ", grdim
#    print "edgeflag = ", edgeflag
#    print "nozflag = ", nozflag

    #First do some input parsing
    if grdim == None:
        grdim = range(sol.dim)
    grdim = sp.array(grdim)
    #Now check if this is a vector field
    if len(field[0].shape) != 4:
        raise Exception("ERROR: Cannot take divergence of a non-vector field in operators.div")

    #Make the edge-fluxes if they're not provided:
    if field_ed2elm == None and edgeflag:
        if len(field_ed[0].shape) != 4:
            raise Exception("ERROR: Cannot take divergence of a non-vector field_ed in operators.div")
        F_ed = mk_ed_flux(sol, field, field_ed, grdim=grdim, gradflux=False)
    elif edgeflag:
        if len(field_ed2elm[0].shape) != 3:
            raise Exception("ERROR: When providing the field_ed2elm array, the edge "+\
                "vector edge flux should be dotted with the normal, and "+\
                "therefore be a tracer field, in operators.div")
        F_ed = field_ed2elm


    #Reshape the field (to take the gradients in master space)
    n_fld = field[0].shape[2]
    if len(grdim) == field[0].shape[1]:
        field = [f[:, :, :, :].reshape(f.shape[0],\
            len(grdim) * f.shape[2], f.shape[3], order='F') for f in field]
    else:
        field = [f[:, grdim, :, :].reshape(f.shape[0],\
            len(grdim) * f.shape[2], f.shape[3], order='F') for f in field]



    #Find the derivatives in the master element
    grad_rst = master_grad_elm(sol, field)

    #Reshape the derivatives
    grad_rst = [grst.reshape(\
        grst.shape[0], sol.dim, len(grdim), n_fld, grst.shape[3], order='F')
        for grst in grad_rst]

    #CM DEBUG
    #print "grad_rst shape = ", grad_rst[0].shape
#    if grad_rst[0].shape[0] > sol.dim**2:
#        print "K_j(F_i) = "
#        print grad_rst[0][0,:,:,0,0]
#        print grad_rst[0][1,:,:,0,0]
#        print grad_rst[0][2,:,:,0,0]
#        print grad_rst[0][3,:,:,0,0]
#        print grad_rst[0][4,:,:,0,0]
#        print grad_rst[0][5,:,:,0,0]
#        print grad_rst[0][6,:,:,0,0]
#        print grad_rst[0][7,:,:,0,0]
#        print "J = "
#        print sol.jac_factor[0][5,:,:,0]

    #combine these gradients using the appropriate factors to find the gradients
    #in real space
    #Initialize outputs
    div_xy = [sp.zeros((f.shape[0], n_fld, f.shape[4]))\
        for f in grad_rst]
    if interiorflag:
        for i in range(len(grad_rst)):
            jj = 0
            for j in grdim:
                for k in range(sol.dim):
                    for l in range(n_fld):
                        div_xy[i][:, l, :] += \
                            grad_rst[i][:, k, jj, l, :] * sol.jac_factor[i][:, k, j, :]
                jj = jj + 1

    #Add the edge terms
    if edgeflag:
        if nozflag:
            nrml_on_elm = sol.get_ed2elm_array(sol.nrml, negate_right=True)
            TOL = 1e-12
        for i in range(len(grad_rst)):
            for l in range(n_fld):
                if nozflag:
                    noz = sp.absolute(nrml_on_elm[i][:, sol.dim -1, :]) < TOL
                    div_xy[i][:, l, :] += \
                        sp.dot(sol.lift[i], noz * F_ed[i][:, l, :]) / sol.jac[i]
                else:
                    div_xy[i][:, l, :] += \
                        sp.dot(sol.lift[i], F_ed[i][:, l, :]) / sol.jac[i]

    return div_xy


