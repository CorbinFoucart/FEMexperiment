#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def L2_error_1D_domain(u, uExFn, dofh, _map, master):
    """ compute the L2 error over the entire 1D domain
    u.T M_K u , where mass matrices are scaled by detJ
    @param uExFn  function which can compute the exact solution
    @param dofh  DOFHandler instance
    """
    uEx = uExFn(dofh.dgnodes)
    diff = np.abs(u - uEx)
    detJ = _map._detJ[0]
    L2Err = np.dot(diff.T, np.dot(master.M, diff*detJ))
    return L2Err

def _1D_L2_domain_integral(u, dofh, _map, master):
    """ computes the L2 integral of u over the domain (\int_{\Omega} |u|^p dx)^(1/p)
    @param u  dgnodes-formatted array u over the domain
    """
    detJ = _map._detJ[0]
    Mku = np.dot(master.M, detJ * np.abs(u))
    intg = 0.
    for k in range(dofh.nElm):
        intg += np.dot(np.abs(u[:, k]), Mku[:, k])
    return np.sqrt(intg)
