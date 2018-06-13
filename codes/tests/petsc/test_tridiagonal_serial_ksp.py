#!/usr/bin/env/python
# -*- coding: utf-8 -*-
"""
Solves a linear tridiagonal system with KSP

inspired by the PETSc tutorial:
http://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex1.c.html
"""

import numpy as np
import petsc4py
from petsc4py import PETSc


def test_tridiagonal_serial_ksp():
    # problem size
    n = 100

    ################################################################################
    # declaration and assembly of A
    ################################################################################
    # create the tridiagonal matrix A
    A = PETSc.Mat().createAIJ([n, n], nnz=3)

    # first row
    A.setValue(row=0, col=0, value=  2.0)
    A.setValue(row=0, col=1, value= -1.0)

    # inner rows
    for i in range(1, n-1):
        A.setValue(row=i, col=i-1, value= -1. )
        A.setValue(row=i, col=i  , value=  2.0)
        A.setValue(row=i, col=i+1, value= -1. )

    # last row
    A.setValue(row=n-1, col=n-2, value= -1.0)
    A.setValue(row=n-1, col=n-1, value=  2.0)

    # assembly complete, make matrix usable
    A.assemblyBegin()
    A.assemblyEnd()

    ################################################################################
    # create exact soln u, solution vector x, RHS vector b
    ################################################################################
    # create solution vector, exact solution, and right hand side
    x = PETSc.Vec().createSeq(n)
    u = PETSc.Vec().createSeq(n)
    b = PETSc.Vec().createSeq(n)

    # set all entries of u to 1.0
    u.set(1.)
    A.mult(u, b)

    ################################################################################
    # create the KSP solver
    ################################################################################
    # create the solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    ################################################################################
    # solve the system, check iteration number and error
    ################################################################################
    # call the solver
    ksp.solve(b, x)

    # check number of iterations
    ksp.getIterationNumber()

    # check the error
    alpha = -1.0
    x.axpy(alpha, u)
    norm = x.norm(norm_type=PETSc.NormType.NORM_2)
    assert np.isclose(norm, 0.)

    ################################################################################
    # clean up
    ################################################################################
    del A, u, x, b, ksp
