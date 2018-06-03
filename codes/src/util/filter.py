# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:28:39 2012

@author: empeeu
"""
import numpy as np
import scipy as sp
import src.master.mk_basis_nodal as mkbn
import src.master.mk_master as mkm
import copy
import os
import pdb


def mk_filter(master, forder=2, degcut=0, alpha=None):
    """
    Create the filtering matrix of the desired order.

    @param master (\c object) The master data-structure or list of master
        datastructures

    @param forder (\c int) The order of the filter

    @param degcut (\c int) The cut-off order of the filter, below which the
        polynomials are untouched

    @param alpha (\c float) The master.n+1 degree polynomial will have this
        value (that is, the polynomial one order higher than the current
        basis)

    @retval filt (\c list) List of numpy arrays of the filter matrices for
        each master element in the input list

    @retval vand (\c list) List of numpy arrays of the modal basis
        vandermonde matrices for each master element in the input list

    @retval ivand (\c list) Same as vand, but the inverse instead

    @note The exponential filter takes the from:
        \f$\exp\left(-\alpha \left(\frac{N - degcut}{N_{max} - degcut + 1}
            \right) ^ {forder} \right), \forall N > degcut\f$,
        where N is the degree of the polynomial

    @note To retrieve the modal coefficients from the nodal basis use:
        np.dot(ivand[i], u[i])
    """

    if alpha == None:
        alpha = -np.log(np.finfo(np.float).eps)

    if type(master) != list:
        master = [master]

    ivand = []
    vand = []
    filt = []
    N = []
    i = 0
    for mast in master:
        #Create the modal basis, then form the Vandermonde matrix and it's
        #inverse
        basis_modal = mkbn.Basis_tensor_modal(mast.n, mast.dim, mast.element)
        pts = mast.basis.nodal_pts
        vand.append(basis_modal.eval_at_pts(pts))
        ivand.append(np.linalg.inv(basis_modal.eval_at_pts(pts)))

        #Create the exponential matrix diagonal (the filter, essentially)
        R = np.ones(basis_modal.nb)
        ids = np.array(basis_modal.basis_degree) > degcut
        N.append(np.array(basis_modal.basis_degree))
        R[ids] = np.exp(-alpha * \
            (1.0 * (N[i][ids] + 1.0 - degcut) / \
            (N[i][ids].max() - degcut + 2.0)) ** forder)
        filt.append( np.dot(vand[i], np.dot(np.diag(R), ivand[-1])))
        i += 1
    return filt, ivand, vand, N

class Filter:
    """This is the filter-class. It will calculate a right-hand side forcing
    which, when added to an equation, will filter the solution.
    __init__(self, sol, sf, forder, degcut, alpha, alphatop, directionfilt)

    This class implements a selective filter, which only filters when the
    solution is deemed 'non-smooth'. Smoothness is defined as the rate at
    which the derivatives in the taylor-series expansion decay, or the size
    of the coefficients in front of the modal bases.

    sf controls the selective part of the filter. sf = 0 filters completely
    and at every timestep. sf = 1 filters with a linear strenght between
    alpha and alphatop. sf = 2 filters with a quadratic strength... etc.

    forder determines the strength of the filter
    degcut determines the cutoff polynomial, below which the coefficients are
            untouched

    directionfilt gives the option of using a directional filter this
        filters only the modes which are too large, so it does not touch
        everything (at least, this is the theory) -- experimental
    """
    def __init__(self, sol, sf=2, forder=2, degcut=0,\
            alpha=0.8, alphatop=0.05, directionfilt=False):
        """
        @param sol (\c object) The solution data-structure

        @param sf (\c float) The selective filter falloff strength
            (see class definition for notes)

        @param forder (\c float) The filter order or strength. Higher order
            is a weaker filter

        @param degcut (\c int) The polynomial degree below which coefficients
            are unmodified

        @param alpha (\c int) The lower modal decay threshold. Above this
            decay rate (i.e. slower decay) the modes will be filtered.
            node, alpha is the slope of the line log(u), where u are the
            modal coefficients

        @param alphatop (\c float) The upper modal decay threshold. Above this
            decay rate (i.e. slower decay) the modes will be fully filtered.

        @param directionfilt (\c bool) Flag to indicate that the experimental
            directional filter should be used.
        """
        if sf == None:
            sf = 2

        self.filt, self.ivand, self.vand, self.basis_degree = \
            mk_filter(sol.master, forder, degcut)

        #Make the decay spectrum
        self.alpha = alpha
        self.spec = [np.exp(-self.alpha * n).reshape(n.shape[0], 1, 1) \
            for n in self.basis_degree]
        norm = [copy.copy(spec[1]) for spec in self.spec]
        for i in range(len(self.spec)):
            self.spec[i][1:] /= norm[i]
            self.spec[i][0] = 0 #never touch the mean

        self.alphatop = alphatop
        self.spectop = [np.exp(-self.alphatop * n).reshape(n.shape[0], 1, 1) \
            for n in self.basis_degree]
        norm = [copy.copy(spec[1]) for spec in self.spectop]
        for i in range(len(self.spectop)):
            self.spectop[i][1:] /= norm[i]
            self.spectop[i][0] = 0 #never touch the mean

        #compare the spectrums (Some silliness to avoid divide by zero )
        self.speccomp = self.spectop
        for i in range(len(self.speccomp)):
            self.speccomp[i][0, :, :] = 1
        self.speccomp = [spec / spectop for spec, spectop in \
            zip(self.spec, self.speccomp)]
        for i in range(len(self.speccomp)):
            self.speccomp[i][0, :, :] = 1

        self.speccompmul = [-np.log(s.prod()) for s in self.speccomp]

        #Set the normalizing tolerance
        self.TOL = 1e-6 #somewhat arbitrary

        #Setting the falloff factor
        self.sf = sf

        #Record the dimension
        self.dim = sol.dim

        #Set the filter type
        self.directionfilt = directionfilt

    def __call__(self, field, rhs=None, sf=None):
        """
        @param field (\c float) List of numpy arrays of the field to be
            filtered.

        @param rhs (\c float) List of numpy arrays of the rate of change of
            the field to be filtered (Currently not used)

        @param sf (\c float) The selective filter decay rate. As in the
            __init__ of the class, but for convenience, included here also.

        """
        if sf == None:
            sf = self.sf

        isvec = False
        if len(field[0].shape) == 4:
            isvec = True
            vecshape = [f.shape for f in field]
            field = [f.reshape(v[0], v[1]*v[2], v[3]) \
                for f, v in zip(field, vecshape)]
        shape = [f.shape for f in field]
        filterrhs = [np.repeat(np.repeat(spec, s[1], 1), s[2], 2)\
            for spec, s in zip(self.spec, shape)]

        for i in range(len(field)):
            #determine if we should filter this field
            tmp = np.dot(self.ivand[i], field[i].reshape(\
                shape[i][0], shape[i][1] * shape[i][2])).reshape(*shape)

            tmpmean = np.abs(tmp[1:self.dim + 1, :, :]).max(0)\
                .reshape(1, shape[i][1], shape[i][2])

            tmpmean2 = \
                np.abs(tmp[self.dim + 1:, :, :]).max(0)\
                .reshape(1, shape[i][1], shape[i][2])

            tmpmean[tmpmean < self.TOL] = tmpmean2[tmpmean < self.TOL]

            #For debugging output
            #tmp1 = copy.copy(tmp)

            #normalize and compare
            #tmp[1:, :, :] /= (np.repeat(tmpmean, shape[i][0] - 1, 0)\
            #    + self.TOL)
            #now compare this to the spectrum
            filterrhs[i][1:, :, :] = filterrhs[i][1:, :, :] \
                / np.abs(tmp[1:, :, :]) \
                * (np.repeat(tmpmean, shape[i][0] - 1, 0) + self.TOL)

            if self.directionfilt:
                #Now, to add this as a forcing term unto the equations we have
                #new = F/r * r + r - r
                #new = r + (F/r - 1) * r
                #new = r (for untouched modes)
                #    = F (for touched modes)
                #So, F/r - 1 = 0 for untouched modes
                 #if greater than 1, the coefficient is already smaller than the
                #spectrum that we have defined
                ids = np.abs(filterrhs[i]) <= 1
                filterrhs[i][ids == False] = 0

                #Also, finally FINALLY we scale the forcing based on how badly
                #or how far away we are from the ideal spectrum (intead of doing
                #a order 1 update for values that are close, we just penalize
                #it a little bit)
                tmp2 = filterrhs[i] / \
                    np.repeat(np.repeat(self.speccomp[i], s[1], 1), s[2], 2)

                tmp2[tmp2 > 1] = 1
                s = shape[i]
                #For touched modes, we have to subtract 1 ,
                filterrhs[i][ids] = filterrhs[i][ids] * tmp2[ids] ** sf - 1

                #now we finish the multiplication
                filterrhs[i][ids] *= tmp[ids]
                filterrhs[i][0, :, :] = 0

                #And we transform it back to the nodal space
                filterrhs[i] = np.dot(self.vand[i], filterrhs[i].reshape(\
                    shape[i][0], shape[i][1] * shape[i][2])).reshape(*shape)

                #tmpnew = field[i] + filterrhs[i]
                #tmpforce =filterrhs[i]
            else:
                #Try2 (shorter name)
                try2 = filterrhs[i]
                try2[try2 > 1] = 1
                try2mean = - (np.log(try2[1:, :, :].prod(0)))
                try2mean /= self.speccompmul[i]
                try2mean[try2mean > 1] = 1
                try2 = np.repeat(try2mean.reshape(1, s[1], s[2]), s[0], 0)
                filterrhs[i] = (np.dot(self.filt[i], field[i].reshape(\
                    shape[i][0], shape[i][1] * shape[i][2])).reshape(*shape)\
                    - field[i]) * try2 ** sf

        if isvec: #reshape outputs
            filterrhs = [f.reshape(v) for f, v in zip (filterrhs, vecshape)]

        return filterrhs

class Nodal_TVD(Filter):
    """TODO
    """
    def __init__(self, sol, sf=None, forder=2, degcut=0,\
            alpha=8, alphatop=5, directionfilt=False):
        """
        @param sol (\c object) The solution data-structure

        @param sf (\c float) The selective filter falloff strength
            (see class definition for notes)

        @param forder (\c float) The filter order or strength. Higher order
            is a weaker filter

        @param degcut (\c int) The polynomial degree below which coefficients
            are unmodified

        @param alpha (\c int) The lower modal decay threshold. Above this
            decay rate (i.e. slower decay) the modes will be filtered.
            node, alpha is the slope of the line log(u), where u are the
            modal coefficients

        @param alphatop (\c float) The upper modal decay threshold. Above this
            decay rate (i.e. slower decay) the modes will be fully filtered.

        @param directionfilt (\c bool) Flag to indicate that the experimental
            directional filter should be used.
        """
        self.filt, self.ivand, self.vand, self.basis_degree = \
            mk_filter(sol.master, forder, degcut)

        self.n = sol.n

        #Make "Projection" matrix for high-order degrees only
        self.PH = [(bdeg >= self.n).reshape(bdeg.shape[0], 1, 1) \
            for bdeg in self.basis_degree]

        #Make the decay spectrum
        self.alpha = alpha
        #self.spec = [np.exp(-self.alpha * n).reshape(n.shape[0], 1, 1) \
        #    for n in self.basis_degree]
        self.spec = [((n+1.0) ** (-alpha)).reshape(n.shape[0], 1, 1) \
            for n in self.basis_degree]
#        norm = [copy.copy(spec[1]) for spec in self.spec]
#        for i in range(len(self.spec)):
#            self.spec[i][1:] /= norm[i]
#            self.spec[i][0] = 0 #never touch the mean

        self.alphatop = alphatop
#        self.spectop = [np.exp(-self.alphatop * n).reshape(n.shape[0], 1, 1) \
#            for n in self.basis_degree]
        self.spectop = [((n+1.0) ** (-self.alphatop)).reshape(n.shape[0], 1, 1) \
            for n in self.basis_degree]

#        norm = [copy.copy(spec[1]) for spec in self.spectop]
#        for i in range(len(self.spectop)):
#            self.spectop[i][1:] /= norm[i]
#            self.spectop[i][0] = 0 #never touch the mean

        #compare the spectrums (some silliness to avoid divide by zero erorr)
        self.speccomp = self.spectop
        for i in range(len(self.speccomp)):
            self.speccomp[i][0, :, :] = 1
        self.speccomp = [spec / spectop for spec, spectop in \
            zip(self.spec, self.speccomp)]
        for i in range(len(self.speccomp)):
            self.speccomp[i][0, :, :] = 1

        self.speccompmul = [-np.log(s.prod()) for s in self.speccomp]

        #Set the normalizing tolerance
        self.TOL = 1e-10#somewhat arbitrary
        self.TOL2 = 1e-14#somewhat arbitrary


        #Setting the falloff factor
        self.sf = sf

        #Record the dimension
        self.dim = sol.dim

        #Set the filter type
        self.directionfilt = directionfilt

        #Build the indexing arrays for finding the TVD of neighboring
        #elements
        ids = [np.arange(et).reshape(et, 1)\
            .repeat(sol.mesh.elm2elm.shape[1], 1) for et in sol.n_elm_type]

        for i in range(len(ids)):
            elm_type_ids = sol.mesh.elm_type == i
            repids = sol.mesh.elm2elm[elm_type_ids, :]
            ids[i][repids > 0] =  repids[repids > 0]
        self.neig_ids = ids

        #CM added for debugging purposes
        self.sol = sol


    def __call__(self, field, rhs, dt, sf=1, mmfact=None):
        """
        @param field (\c float) List of numpy arrays of the field to be
            filtered.

        @param rhs (\c float) List of numpy arrays of the rate of change of
            the field to be filtered
        @param dt (\c float) The time step size

        @param sf (\c float) The selective filter decay rate. As in the
            __init__ of the class, but for convenience, included here also.

        @param mmfact (\c float) List of numpy arrays of the moving-mesh
            correction factor.

        @retval (\c float) List of numpy arrays containing the forcing that
            corrects the variation of the solution.

        """
        if self.sf == None:
            self.sf = sf
        if mmfact == None:
            mmfact = [np.array([1]) for i in range(len(field))]

        isvec = False
        if len(field[0].shape) == 4:
            isvec = True
            vecshape = [f.shape for f in field]
            field = [f.reshape(v[0], v[1]*v[2], v[3]) \
                for f, v in zip(field, vecshape)]
            rhs = [f.reshape(v[0], v[1]*v[2], v[3]) \
                for f, v in zip(rhs, vecshape)]

        shape = [f.shape for f in field]
        n_elm = 0
        for s in shape:
            n_elm += s[2]
        tvdrhs = [np.zeros(s) for s in shape]
        maxfield = np.zeros((shape[0][1], n_elm))
        minfield = np.zeros((shape[0][1], n_elm))
        elstart = 0
        for i in range(len(field)):
            elend = elstart + shape[i][2]
            #STEP 1: Determine the total variation of this field
            minfield[:, elstart:elend] = field[i].min(0)
            maxfield[:, elstart:elend] = field[i].max(0)
            elstart = elend

        for i in range(len(field)):
            s = shape[i]
            minfield2 = minfield[:, self.neig_ids[i]].min(2) \
                .reshape(1, s[1], s[2])
            maxfield2 = maxfield[:, self.neig_ids[i]].max(2) \
                .reshape(1, s[1], s[2])

            #Now we have the min's and maxes for all elements
            #STEP 2: Create the first-pass TVD correction
            #That is, compute phibar = phi + dt*F
            #Create the first-pass TVD correction

            tmp = (field[i] + dt * rhs[i])
            for j in range(shape[i][1]):
#                print j, tmp[: j, :].shape, mmfact[i].shape
                #CM: mmfact is always 1; see above!
                tmp[:, j, :] = tmp[:, j, :] * mmfact[i]

            tmp1 = tmp - maxfield2.repeat(s[0], 0)
            top_ids = tmp1 > self.TOL
            tvdrhs[i][top_ids] = - tmp1[top_ids] / dt
            tmp1 = tmp - minfield2.repeat(s[0], 0)
            bot_ids = tmp1 < self.TOL
            tvdrhs[i][bot_ids] = - tmp1[bot_ids] / dt

            #Okay, this correction will affect the mean in a cell, but we
            #do not want to touch the mean... so let's figure out what the
            #change in the mean is for this correction
            meanchange = np.dot(self.ivand[i][0, :], tvdrhs[i].reshape(\
                s[0], s[1] * s[2])).reshape(1, s[1], s[2])

            #Keep track of which elements have been limited
            lim_elm_ids = top_ids.any(0) | bot_ids.any(0)

            #Create the max_up_allowed change array, which gives the maximium
            #value by which we may increase the field
            #With the indexing, it's simpler to do one field at a time
            for j in range(s[1]):
                le_ids = lim_elm_ids[j, :]
                #First look at nodes where the mean has gone down
                ids = (meanchange[0, j, :] > self.TOL) & (le_ids)
                if any(ids):
                    max_adj = minfield2[:, j, ids].repeat(s[0], 0) \
                        - (field[i][:, j, ids] + dt * (\
                        rhs[i][:, j, ids] + tvdrhs[i][:, j, ids]))
                    nodes = np.dot(self.ivand[i], max_adj) / dt

                    ids4 = np.zeros((ids.shape), dtype=bool)
                    ids3 = np.abs(nodes[0, :]) > np.abs(meanchange[0, j, ids])
                    ids2 = (np.abs(nodes[0, :]) > self.TOL) & ids3
                    ids4[ids] = ids2
                    nodes[:, ids2] = - nodes[:, ids2] / nodes[0:1, ids2]\
                        .repeat(s[0], 0) \
                        * (meanchange[:, j, ids4].repeat(s[0], 0))
                    #nodes[:, ids2 == False] = 0
                    tvdrhs[i][:, j, ids] += np.dot(self.vand[i], nodes)
                    meanchange[0, j, ids] = 0

                ids = (meanchange[0, j, :] < -self.TOL) & (le_ids)
                if any(ids):
                    max_adj = maxfield2[:, j, ids].repeat(s[0], 0) \
                        - (field[i][:, j, ids] + dt * (\
                        rhs[i][:, j, ids] + tvdrhs[i][:, j, ids]))
                    nodes = np.dot(self.ivand[i], max_adj) / dt

                    ids4 = np.zeros((ids.shape), dtype=bool)
                    ids3 = np.abs(nodes[0, :]) > np.abs(meanchange[0, j, ids])
                    ids2 = (np.abs(nodes[0, :]) > self.TOL) & ids3
                    ids4[ids] =ids2
                    nodes[:, ids2] = - nodes[:, ids2] / nodes[0:1, ids2]\
                        .repeat(s[0], 0) \
                        * (meanchange[:, j, ids4].repeat(s[0], 0))
                    #nodes[:, ids2 == False] = 0
                    tvdrhs[i][:, j, ids] += np.dot(self.vand[i], nodes)
                    meanchange[0, j, ids] = 0

            #3. For mean-changes that still remain, we simply subtract out
            #the mean from the tvdrhs correction
            meanchange = np.dot(self.ivand[i][0, :], tvdrhs[i].reshape(\
                s[0], s[1] * s[2])).reshape(1, s[1], s[2])
            mc_ids = np.abs(meanchange) > self.TOL * 10
            if np.any(mc_ids):
                pass
#                print "NODAL_TVD limiter has not properly adjusted the",\
#                    " means for", mc_ids.sum(), "elements."
            for j in range(s[1]):
                ids = mc_ids[0, j, :]
                #if ids.sum() > 0:
                #    iiidddsss = ids.nonzero()[0][0]
                #    print tvdrhs[i][:, j, iiidddsss], \
                #        meanchange[:, j, iiidddsss],\
                #        minfield2[0, j, iiidddsss],\
                #        maxfield2[0, j, iiidddsss], '\n'
                tvdrhs[i][:, j, ids] += -self.vand[i][0, 0] \
                    * meanchange[:, j, ids].repeat(s[0], 0)

        #If selectively limiting depending on smoothness of solution

        #CM 02/14/2018: Bug here?  Should not depend on p being greater than 1; commented out.
        #if sf != None and self.n > 1:
        if sf is not None:
            for i in range(len(field)):
                s = shape[i]
                #Calculate function smoothness
                tmpmean = self.calc_selectivity(field[i]  + dt * rhs[i], i)
                #Scale the limiting forcing appropriately
                #pdb.set_trace()
                if __debug__ and np.mod(self.sol.k, self.sol.debuginterval) == 0:
                #if np.mod(self.sol.k, self.sol.debuginterval) == 0:
                    #CM: Using TVD limiter (for advection) implies problem is time-dependent
                    outfile = (self.sol.output_dir + os.sep
                           + 'TVD_k' + str(self.sol.k)
                           + 's' + str(self.sol.stage)
                           + 'p' + str(self.sol.n) + '.mat')
                    if isvec:
                        tvdOutDict = dict()
                        tvdOutDict['alpha_u1'] = tmpmean
                    else:
                        tvdOutDict = sp.io.loadmat(outfile)
                        tvdOutDict['alpha_T'] = tmpmean
                    tvdOutDict['x'] = self.sol.dgnodes
                    sp.io.savemat(outfile, tvdOutDict)
                    #pdb.set_trace()

                tvdrhs[i] = tvdrhs[i] * np.repeat(tmpmean, s[0], 0) ** sf

        if isvec: #reshape outputs
            tvdrhs = [f.reshape(v) for f, v in zip (tvdrhs, vecshape)]

        return tvdrhs

    def calc_smoothness(self, field, i):
        s = field.shape
         #Create the reference fall-off filter
        filterrhs = \
            np.repeat(np.repeat(self.PH[i], s[1], 1), s[2], 2)
        #Do a modal decomposition of field
        tmp = np.dot(self.ivand[i], (field).reshape(\
            s[0], s[1] * s[2])).reshape(*s)

        # Find the value of the detector for the field
        tmpmean = np.log(((tmp * filterrhs + self.TOL2)**2).sum(0)\
            / ((tmp[1:,:,:]**2 + self.TOL2).sum(0))).reshape(1, s[1], s[2])

        return tmpmean

    def calc_smoothness_type(self, field):
        tmpmean = []
        for i in range(len(field)):
            tmpmean.append(np.repeat(self.calc_smoothness(field[i], i), \
                field[i].shape[0], 0))
        return tmpmean

    def calc_selectivity(self, field, i):
        """Computes the selectivity index alpha for a field on a given element type

        Helper function for calc_selectivity_type
        """
        #Get the smoothness of the field
        tmpmean = self.calc_smoothness(field, i)
        #Compare this to the reference spectrums
        botref = np.log(((self.spec[i] * self.PH[i]) ** 2).sum(0) \
            / ((self.spec[i][1:,:,:]) ** 2 + self.TOL2).sum(0))[0][0]
        topref = np.log(((self.spectop[i] * self.PH[i]) ** 2).sum(0) \
            / ((self.spectop[i][1:,:,:]) ** 2 + self.TOL2).sum(0))[0][0]

        tmpmean = (tmpmean - botref) / (topref - botref - self.TOL2)
        tmpmean[tmpmean < 0] = 0
        tmpmean[tmpmean > 1] = 1

        return tmpmean

    def calc_selectivity_type(self, field):
        """Computes the selectivity index alpha for each element type"""
        tmpmean = []
        for i in range(len(field)):
            tmpmean.append(np.repeat(self.calc_selectivity(field[i], i), \
                field[i].shape[0], 0))
        return tmpmean

#==============================================================================
#     The next two function were used to hueristically tune the modal decay
#       spectrums of various polynomial order basis. Turn out, using an
#       exponential decay with coefficient -1 is a pretty good rule of thumb
#       These functions are left for posterity .
#==============================================================================
def tune_sin(n, recurse=16, addstart=0.0625, debug=False):
    """
    Uses a sin function to tune/find the decay spectrum of the
        least smooth function that does not cause overshoots above the
        defined tolerance.
    @param n (\c int) The order of basis to be tuned

    @param recurse (\c int) The number of refinements/recursions to find the
        least smooth function that does not cause overshoots

    @param addstart (\c float) The resolution at which the least smooth
        function is search for.

    @param debug (\c bool) The debug flag. If True, additional outputs are
        given for debug purposes.

    @retval spectrum (\c float) Numpy array of the decay spectrum of the
        least-smooth function

    @retval coeff (\c float) The slope of the line such that e^(coeff * N)
        gives the decay spectrum.

    These are only outputted when debug == True

    @retval u (\c float) The nodal values

    @retval sol (\c float) The low-resolution actual solution at the nodes

    @retval sol_full (\c float) The high-resolution interpolated nodal values

    @retval pts (\c float) Locations of the nodal points

    @retval sol_real (\c float) The high-resolution actual solution
    """
    master = [mkm.Master_nodal(n, 1, 0)]
    filt, ivand, vand, N = mk_filter(master)
    filt = filt[0]
    ivand = ivand[0]
    vand = vand[0]
    tol = 1e-3
    #We want to know what wavenumber of sin we can approximate without overshooting
    master = master[0]

    pts = master.basis.nodal_pts
    pts_full = np.linspace(-1, 1, n ** 2 * 10)
    pts_full = pts_full.reshape(pts_full.shape[0], 1)

    vand_full = master.basis.eval_at_pts(pts_full)
    minfreq = 1e15
    PHI = np.linspace(0, 2, 200)
    minphi = 0
    for phi in PHI:
        notfailed = True
        for i in range(recurse):
            if i == 0:
                freq = 0
                add = addstart
                ii = 0
            else:
                freq -= add
                add /= 2
            while notfailed:
                ii += 1
                freq += add
                u = np.sin(np.pi * freq * pts + phi)
                u_full = np.dot(vand_full, u)
                if np.abs(u_full).max() > 1 + tol or ii == 100:
                    notfailed = False
                    print "phi=", phi, "i=",i, "freq=", freq, ": ", np.abs(u_full).max()
            notfailed = True
        if freq < minfreq:
            minfreq = freq
            minphi = phi
        minfreq = min(minfreq, freq)

    print "Min_frequency:", minfreq, "At phase of:", minphi
    sol = np.sin(np.pi * minfreq * pts + minphi)
    u = np.dot(ivand, sol)
    sol_full = np.dot(vand_full, sol)
    sol_real = np.sin(np.pi * minfreq * pts_full + minphi)

    u = np.abs(u / u[1])
    coeff = np.polyfit(np.arange(n + 1), np.log(u), 1)[0]
    spectrum = np.exp(coeff * np.arange(n+1))
    spectrum[1:] /= spectrum[1]

    if debug == True:
        return spectrum, coeff, u, pts, sol, pts_full, sol_full, sol_real
    else:
        return spectrum, coeff

def tune_tanh(n, recurse=16, addstart=0.0625, debug=False):
    """ Uses a hyperbolic tangent to tune/find the decay spectrum of the
        least smooth function that does not cause overshoots above the
        defined tolerance.

    @param n (\c int) The order of basis to be tuned

    @param recurse (\c int) The number of refinements/recursions to find the
        least smooth function that does not cause overshoots

    @param addstart (\c float) The resolution at which the least smooth
        function is search for.

    @param debug (\c bool) The debug flag. If True, additional outputs are
        given for debug purposes.

    @retval spectrum (\c float) Numpy array of the decay spectrum of the
        least-smooth function

    @retval coeff (\c float) The slope of the line such that e^(coeff * N)
        gives the decay spectrum.

    These are only outputted when debug == True

    @retval u (\c float) The nodal values

    @retval sol (\c float) The low-resolution actual solution at the nodes

    @retval sol_full (\c float) The high-resolution interpolated nodal values

    @retval pts (\c float) Locations of the nodal points

    @retval sol_real (\c float) The high-resolution actual solution
    """
    master = [mkm.Master_nodal(n, 1, 0)]
    filt, ivand, vand, degrees = mk_filter(master)
    filt = filt[0]
    ivand = ivand[0]
    vand = vand[0]

    #We want to know what wavenumber of sin we can approximate without overshooting
    master = master[0]

    pts = master.basis.nodal_pts
    pts_full = np.linspace(-1, 1, n ** 2 * 10)
    pts_full = pts_full.reshape(pts_full.shape[0], 1)
    tol = 1e-15
    vand_full = master.basis.eval_at_pts(pts_full)
    minfreq = 1e15
    PHI = [0]#np.linspace(-0.05, 0.05, 100)
    minphi = 0
    for phi in PHI:
        notfailed = True
        for i in range(recurse):
            if i == 0:
                freq = 0
                add = addstart
                ii = 0
            else:
                freq -= add
                add /= 2
            while notfailed:
                ii += 1
                freq += add
                u = np.tanh(freq * pts + phi)
                u_full = np.dot(vand_full, u)
                u_real = np.tanh(freq * pts + phi)
                #if np.abs(u_full - np.tanh(freq * pts_full + phi)).max() > tol\
                if np.abs(u_full).max() > np.abs(u_real).max() + tol\
                        or ii == 30:
                    notfailed = False
                    print "phi=", phi, "i=",i, "freq=", freq, ": ",\
                        np.abs(u_full).max(), np.abs(u_real).max()
            notfailed = True
        freq -= add
        if freq < minfreq:
            minfreq = freq
            minphi = phi
        minfreq = min(minfreq, freq)

    print "Min_frequency:", minfreq, "At phase of:", minphi
    sol = np.tanh(minfreq * pts + minphi)
    u = np.dot(ivand, sol)
    return u, sol, pts
#    sol_full = np.dot(vand_full, sol)
#    sol_real = np.tanh(minfreq * pts_full + minphi)
#
#    u = np.abs(u / u[1])
#    coeff = 0#np.polyfit(np.arange(n + 1), np.log(u), 1)[0]
#    spectrum = np.exp(coeff * np.arange(n+1))
#    spectrum[1:] /= spectrum[1]

    if debug == True:
        return spectrum, coeff, u, sol, sol_full, pts, sol_real
    else:
        return spectrum, coeff
