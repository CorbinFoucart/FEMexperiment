# -*- coding: utf-8 -*-
""" @package src.pyutil
A collection of functions specifically for python coding usage, and not related
to FEM. Some of these are borrowed/available/similar to MATLAB functions.


@author: Matt Ueckermann

Created on Sat Feb 26 18:30:11 2011
"""

import numpy as np
from scipy import ndimage, array
from scipy.misc import factorial
from scipy.linalg import solve as sp_solve
import copy
import os, pdb

def find(thelist, item=True, ret_all=False):
    """Find a specified function in a list.

    @param thelist (\c list) The list which will be searched through

    @param item (\c ??) The item which will be searched for.
                        Default value is True

    @param ret_all (\c bool) Flag to indicate whether all instances should be
                            searched for, or only the first one

    @retval index (\c int) List of indices of where the item was found

    @note http://dev.ionous.net/2009/01/python-find-item-in-list.html
    """
    if ret_all:
        return [i for i in xrange(len(thelist)) if thelist[i] == item]
    else:
        return next((i for i in xrange(len(thelist)) if thelist[i] == item), \
            None)

###############################################################################
def parse(argv, string):
    """ This function parses the inputs for function when called from the
    command line.

    @param    argv    The arguments from the command-line. [Accessed in a
                      function using: import sys, sys.argv]
    @param    string  Tuple of strings that give the flags that should be
                      parsed. Given in return order.
    @return   Each element of returned list corresponds to the option
              value as specified.

    @code
    #Usage:
    >>> (N,dim,element) = parse(sys.argv, ("-N", "-dim", "-element"))
    @endcode

    @author: Matt Ueckermann
    """
    out = [0 for i in range(len(string))]
    while len(argv) > 1:
        option = argv[1]
        del argv[1]
        found = False
        for i in xrange(len(string)):
            if option == string[i]:
                out[i] = argv[1]
                del argv[1]
                found = True
        if found == False:
            print("Option ", option, "not found. Please specify one of", string)
            del argv[1]
    print(out)
    return out

###############################################################################
def sortrows(a, i=0, index_out=False, recurse=True):
    """ Sorts array "a" by columns i
    @author Matt Ueckermann

    @param a (\c array) array to be sorted
    (OPTIONAL)
    @param i (\c int) column to be sorted by, taken as 0 by default
    @param index_out (\c bool) return the index I such that a(I) = sortrows(a,i)
    @param recurse  (\c bool) recursively sort by each of the columns. i.e.
                    once column i is sort, we sort the smallest column number
                    etc. True by default.

    @retval a: The array 'a' sorted in descending order by column i
    @retval I: The index such that a(I) = sortrows(a,i)

    @code
    #Example usage:

    >>> a = array([[1, 2],
                   [3, 1],
                   [2, 3]])
    >>> b = util.sortrows(a,0)
    >>> b
    array([[1, 2],
           [2, 3],
           [3, 1]])
    c, I = util.sortrows(a,1,True)

    >>> c
    array([[3, 1],
           [1, 2],
           [2, 3]])
    >>> I
    array([1, 0, 2])
    >>> a[I,:] - c
    array([[0, 0],
           [0, 0],
           [0, 0]])
    @endcode

    @see unique
    """
    I = np.argsort(a[:, i])
    a = a[I, :]
    #We recursively call sortrows to make sure it is sorted best by every
    #column
    if recurse & (len(a[0]) > i + 1):
        for b in np.unique(a[:, i]):
            ids = a[:, i] == b
            colids = list(range(i)) + list(range(i+1, len(a[0])))
            a[np.ix_(ids, colids)], I2 = sortrows(a[np.ix_(ids, colids)], 0, True, True)
            I[ids] = I[np.nonzero(ids)[0][I2]]

    if index_out:
        return a, I
    else:
        return a

###############################################################################
def unique(a, by='', flag=False):
    """Finds and returns unique elements, row, or columns in a 2D numpy array
    @author Matt Ueckermann

    @param a (\c array) array to be sorted
    @param by (\c str): Takes arguments 'cols' to find unique colums
        or 'rows to find unique rows. If unspecified or only unique elements
        are returned
    @param flag (\c bool): Set to True if the index vector 'I' of the
        unique rows/columns and the reverse index map 'J' should be returned

    @retval e: The array containing only unique elements
    @retval I: The index such that a[I,:] = e (if by='rows') or a[:,I] = e (if by='cols')
        That is, I is the unique, possibly permuted, row/column numbers of a.
    @retval J: The index such that e[J,:] = a (if by='rows') or e[:,J] = a (if by='cols')
        That is, J is the rows/columns of e needed to get back a.

    @code
    #Example usage:
    >>> import pyutil as util
    >>> import numpy as np
    >>> a = np.array([[2,1,2],
                      [2,1,2],
                      [4,3,4],
                      [2,1,2],
                      [4,3,4]])
    >>> b = util.unique(a)
    >>> b
    array([1, 2, 3, 4])
    >>> c = util.unique(a, 'rows')
    >>> c
    array([[2, 1, 2],
           [4, 3, 4]])
    >>> d, I, J = util.unique(a, by='cols', flag=True)
    >>> d
    array([[1, 2],
           [1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])
    >>> I
    array([1, 0])
    >>> J
    array([1, 0, 1])
    @endcode

    @see util.sortrows
    """
    if by == 'rows':
        rows = a.shape[0]
        b, I = sortrows(a, 0, True, True)
        c = b[0:(rows - 1), :] != b[1:rows, :]
        c = c.transpose()
        d = np.ones((rows), dtype=bool)
        d[1:rows] = c.any(0)
        e = b[d, :]
        J = np.cumsum(d, dtype=int) - 1
        j = np.cumsum(d, dtype=int) - 1
        J[I] = j
        I = I[d]
    elif by == 'cols':
        e, I, J = unique(a.transpose(), 'rows', True)
        e = e.transpose()
    else:
        e, I, J = np.unique(a, True, True)
    if flag:
        return e, I, J
    else:
        return e

###############################################################################
class Bi_linear_interp:
    """An object that can be used as a function that bi-linearly interpolates
    the provided regularly-spaced data to desired points.

    __init__(coords, fld, dim) -- class initialization funtion
    @param coords: (\c float) Numpy array of coordinates created using np.mgrid

    @param fld: (\c float) Numpy array of field values evaluated at the points
                           specified by coords.

    @param dim: (\c int) Dimension of the coordinates (1D, 2D, or 3D)

    __call__(newcoords)
    @param newcoords (\c float) Numpy array of shape = (n_pts, dim) where the
                     provided field should be interpolated to.

    @code
    >>> # Usage example:
    >>> import numpy as np
    >>> import pyutil
    >>> coords = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]
    >>> fld = coords[0]*coords[1]*coords[2]
    >>> f = pyutil.Bi_linear_interp(coords, fld, 3)
    >>> f(array([[0.5,0.5,0.5],[0,0,0],[0.8,0.5,-0.8]]))

    array([  1.26836754e-01,  -3.84341200e-20,  -3.27067853e-01])

    @endcode
    """
    def __init__(self, coords, fld, dim, order=1):
        x0 = [None] * dim
        dx = [None] * dim
        for i in range(dim):
            if dim > 1:
                coord = coords[i]
                coord2 = coords[i]
            else:
                coord = coords
                coord2 = coords
            for j in range(dim):
                coord = coord[0]
                coord2 = coord2[np.int(i == j)]
            x0[i] = coord
            dx[i] = coord2 - coord
        self.x0 = x0
        self.dx = dx
        if order > 1:
            self.coeffs = ndimage.spline_filter(fld, order=order)
        else:
            self.coeffs = fld
        self.dim = dim
        self.order = order

    def __call__(self, newcoords):

        coords = array([[None] * self.dim ] * len(newcoords))
        if self.dim > 1:
            coords = [(newcoords[:, i] - self.x0[i]) / self.dx[i] \
                for i in xrange(self.dim)]
        else:
            coords = array([(newcoords[:] - self.x0[0]) / self.dx[0]])

        return ndimage.map_coordinates(self.coeffs, coords, prefilter=True,
                                       order=self.order)
###############################################################################
def permute_options(option_list):
    """ Permutes a set of options and returns full list with each option
    repeated as much as necessary for a full permutation.

    @param option_list (\c list) A list of lists containing the options that
        need to be permuted.

    @retval permuted_list (\c list) A list of lists containing repetitions
        of options to give a full permutation of the options.

    @note len(permuted_list[i]) = len(options_list[0]) * len(options_list[1])
        * ... * len(options_list[len(options_list)])

    @note For example:
    >>> option_list = [[1, 2, 3], [-1, 0], ['a', 'b', 'c', 'd']]
    >>> permuted_list = permute_options(option_list)
    >>> permuted_list
    [[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    , [-1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1,
       -1, 0, 0, 0, 0], ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b',
    'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']]
    """
    lengths = [len(option) for option in option_list]
    total_len = 1
    for l in lengths:
        total_len *= l
    outputs = [[0 for l in range(total_len)] for length in lengths]
    repeats = copy.copy(total_len)
    for i in range(len(option_list)):
        n = 0
        repeats = int(repeats / lengths[i])
        rounds = int(total_len / lengths[i] / repeats)
        for j in range(rounds):
            for opt in option_list[i]:
                for k in range(repeats):
                    outputs[i][n] = opt
                    n += 1
    return outputs
###############################################################################
def write_result(filename, header, data):
    """This utility appends data to a file, but locks the file to make sure
    that two processes do not try to write to the file at the same time.
    This is used for collecting outputs from multiple parallel runs in a
    single file.

    @param filename (\c string) The name of the file. Note, the extension
        ".res" will be automatically added to the end of the filename. The
        lock-file will have ".lck" added to the end of the filename.

    @param header (\c string) Any header data that should be written to the
        top of the file if it's being written for the first time.

    @param data (\c string) A string of data to be written to the file.

    @note this function has not outputs, but does create the file
        filename + '.res'
    """
    locked = True
    while locked:
        try:
            lkfile = open(filename + '.lck')
        except:
            locked = False
            lkfile = open(filename + '.lck', 'w')
            try:
                outfile = open(filename + '.res')
                outfile.close()
            except:
                #Write the header
                outfile = open(filename + '.res', 'a')
                outfile.write(header + '\n\n')
                outfile.close()
            outfile = open(filename + '.res', 'a')
            outfile.write(data + '\n')
            outfile.close()
            lkfile.close()
            os.remove(filename + '.lck')

###############################################################################
def taylor(dx, deriv, pade=None, out=False):
    """
     [C P] = taylor(dx,deriv, pade)
     CALCULATES THE COEFFICIENTS FOR TAYLOR EXPANSIONS TO APPROXIMATE
     DERIVATIVES. INCLUDES PADE SCHEME EXPANSIONS

     INPUTS
       dx:       Vector containing values of (x-dx)
       deriv:      Which derivative to approximate: d^deriv/dx^deriv
       pade:       Optional input. 2D array, each column containing
                   [{where to evaluate derivite}, {which derivative to use}]
       out:        If true, scheme is output to the screen
     OUTPUTS
       C:          Coefficients in front of f(x) terms. Ordered from large to
                       small
       P:          Pade coefficients in the order inputted
       STDOUT - the scheme is printed to standard output if out==true

     Example usage:
    1. To recover 2nd order central difference for first order derivatives:
     >> C=taylor(np.array([-1, 1]),1)

      1/(2h^1) *[1 f(1*dx) + -0 f(0*dx) + -1 f(-1*dx)]
      		 + h^2 * 0.166667 d^3f^(3)/dx^3

     C =

         0.5000
              0
        -0.5000

    2. To recover 2nd order forward difference for second order derivatives:
    >>  C=taylor(np.array([3, 2, 1]),2)

      1/(1h^2) *[-1 f(3*dx) + 4 f(2*dx) + -5 f(1*dx) + 2 f(0*dx)]
      		 + h^2 * -0.916667 d^4f^(4)/dx^4

     C =

        -1.0000
         4.0000
        -5.0000
         2.0000

    3. To recover 4th order central difference pade scheme for first order derivatives:
     >> [C P] = taylor(np.array([-1, 1]),1,np.array([[-1, 1], [1, 1]]))

      1/(1.33333h^1) *[1 f(1*dx) + -0 f(0*dx) + -1 f(-1*dx)] +
     	(h^0) *-0.25 f^(1)(-1*dx) + (h^0) *-0.25 f^(1)(1*dx)
      		 + h^4 * -0.00833333 d^5f^(5)/dx^5

     C =

         0.7500
              0
        -0.7500


     P =

        -0.2500
        -0.2500
    3. To recover 4th order central difference pade scheme for second order derivatives:
    >> [C P] = taylor(np.array([-1, 1]),2,np.array([[-1, 2],[1, 2]]))

      1/(0.833333h^2) *[1 f(1*dx) + -2 f(0*dx) + 1 f(-1*dx)] +
     	(h^0) *-0.1 f^(2)(-1*dx) + (h^0) *-0.1 f^(2)(1*dx)
      		 + h^4 * -0.005 d^6f^(6)/dx^6

     C =

         1.2000
        -2.4000
         1.2000


     P =

        -0.1000
        -0.1000

    Author: Matt Ueckermann for MIT course 2.29 -- transcribed to python
    """
    #check inputs
    if pade == None:
        pade=np.array([])

    #Set RHS of system. Derivative to be solved for has value 1, all else is
    #zero.
    rhs = np.zeros(len(dx)+pade.shape[0])
    rhs[deriv - 1] = 1
    #Calculate the coefficients (term) of the Taylor series expansion
    term = np.zeros((len(dx) + 3 + pade.shape[0], len(dx) + pade.shape[0]))
    for j  in range(len(dx)):
        dx2=dx[j]
        for i in range(len(dx) + 3 + pade.shape[0]):
            term[i, j] = dx2 ** (i + 1) / factorial(i + 1)

    #Calculate the coefficients (term) of the Taylor series expansion for
    #derivatives (Pade scheme)
    for j in range(len(dx), len(dx) + pade.shape[0]):
        dx2 = pade[j - len(dx), 0]
        for i in range(len(dx) + 3 - pade[j - len(dx), 1] + pade.shape[0]):
            #Need to be careful with offsets here
            term[i + pade[j - len(dx), 1] - 1, j] = dx2 ** (i) / factorial(i)

    #Error checking
    minij = min(term.shape)
    if minij < len(rhs):
        print('Not enough terms included to solve for derivative\n Add more terms to dx')
    minij = len(dx)+pade.shape[0]
    coeffs = sp_solve(term[:minij,:minij], rhs)

    # find remainder term:
    Rem = np.dot(term, coeffs)
    Rem[deriv - 1] = 0
    Rem = np.abs(Rem)
    dR = Rem.nonzero()[0][0]

    #Separate into Pade and non-pade
    divide = len(dx)
    C = coeffs[:divide]
    P = coeffs[divide:]
    #Normalize coefficients for output to screen
    denom = 1. / (np.min(np.abs(C)))
    C = C * denom
    C = np.concatenate((-np.array([np.sum(C)]), C))
    dx = np.concatenate((np.array([0]), dx))
    ind = np.argsort(dx)[::-1] #Sort
    dx = dx[ind]
    C = C[ind]
    #Print coefficients to screen
    if out:
        print('\n 1/(%gh^%g) *['.format(denom, deriv))
        for j in range(len(C)):
            print('%g f(%g*dx)'.format(C[j],dx[j]))
            if j != len(C) - 1:
                print(' + ')

    C = C / denom
    if out:
        print(']')
        if pade.shape[0] > 0:
            #Print pade coefficients to screen
            print(' + \n\t')
            for j in range(len(P)):
                print('(h^%g) *'.format(pade[j, 1] - deriv))
                print('%g f^(%g)(%g*dx)'.format(P[j], pade[j,1],pade[j,0]))
                if j != len(P) - 1:
                    print(' + ')

    #Print remainder to screen
    print('\n \t\t + h^%g * %g d^%gf^(%g)/dx^%g\n'.format(dR - deriv + 1, Rem[dR], dR + 1, dR + 1, dR + 1))

    if P.shape[0] > 0:
        return C, P
    else:
        return C

def extrapcoefs(N, x=None):
    """
     function [D] = extrapcoefs(N, x)
     Creates the coefficients used to extrapolate a value from a polynomial
     function.

     INPUTS:
     @param N (\c int)      The degree of the polynomial
     @param x (\c float[]) (optional) the points where the polynomial is
             evaluated.
               x[0]: x value where extrapolation is desired
               x[1 : N + 1]: x values of existing data
             Default value x = [-1, 0, 1, ..., N-1]

     OUTPUTS:
     @retval D (\c float[])  Extrapolation coefficients, such that:
               F(x(1)) ~ D * F(x(2 : N + 1))

     Examples:
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

     Written by: Matt Ueckermann
    """
    #Creating the coefficients is pretty simple:
    # Eventually we want to know the y-value at x*, which can be evaluated
    # using:
    # [x*^0 x*^1 x*^2 ... x*^n] * [a1;a2;...;an] = y*
    #
    # But for that we need to solve for coefficients of polynomial:
    # [x1^0 x1^1 x1^2 ... x1^n;
    #  x2^0 x2^1 x2^2 ... x2^n;
    #  ...   ...  ...  ... ...;
    #  xn^0 xn^1 xn^2 ... xn^n;] * [a1;a2;...;an] = [y1;y2;...;yn]
    #
    # A = X^-1 * Y
    # (X*) * A = (X*) * X^-1 * Y;
    #          = D * Y;

    #Create the Vandermonde matrix:
    if x == None:
        x = np.arange(-1, N + 1)

    V = np.fliplr(np.vander(x[1:]))

    x = np.fliplr(np.vander(np.repeat(x[0], N + 1)))
    D = sp_solve(V.T, x[0, :])
    return D

