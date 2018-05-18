import sympy as sym
from sympy import Symbol, symbols, Matrix, diff
from sympy.tensor.array import Array, tensorproduct, derive_by_array
import pdb

class VectorField(object):
    """ representation of vector field; simple interface to the sympy.physics.vector functionality
    All VectorField objects are internally 3D, although the user could pass a 2D field in. These
    vector fields are either 2D or 3D spatially.
    """

    def __init__(self, v):
        """
        @param v  a list of sympy expressions for each component of the field. Each expression
        should be dependent on only the Symbols x, y, z, t, and the length of v should be either 2
        or 3.
        """
        self.dim, syms = len(v), symbols('x y z')
        self.basis = syms[0:self.dim]
        self.v = v

    def gradient(self):
        """ returns a sympy matrix containing Dij = dvj/dxi (vj by col, d/dxi by row) """
        v = Array(self.v)
        gradv = derive_by_array(v, self.basis)
        return gradv.tomatrix()

    def divergence(self):
        div = 0
        for idx, component in enumerate(self.v):
            dvidxi = diff(component, self.basis[idx])
            div += dvidxi
        return div

    def advection(self):
        """ returns a sympy matrix containing (v \cdot \nabla)v """
        gradv = self.gradient()
        v = Matrix(self.v)
        adv = v.T * gradv
        return adv.T

    def time_derivative(self):
        """ returns a sympy Matrix (col vector) of the time derivative of self.v """
        time_derivative = list()
        for component in self.v:
            time_derivative.append(diff(component, Symbol('t')))
        return Matrix(time_derivative)

    def laplacian(self):
        """ returns the vector laplacian of self.v as a sympy Matrix (column vector)
        NB: the vector laplacian is the laplacian of each component
        """
        vector_laplacian = list()
        for component in self.v:
            scalar_laplacian = self._scalar_laplacian(component)
            vector_laplacian.append(scalar_laplacian)
        return Matrix(vector_laplacian)

    def component_gradient(self, component):
        """ since gradient organized by Dij = dvj/dxi, return column j """
        return self.gradient()[:, component]

    def _scalar_laplacian(self, phi):
        """ returns the scalar laplacian d/dxi d/dxi phi = sum_i d^2/dxi^2 phi of a scalar sympy
        expression phi """
        scalar_laplacian = 0
        for idx, varsymbol in enumerate(self.basis):
            scalar_laplacian += diff(phi, varsymbol, 2)
        return scalar_laplacian

class ScalarField(object):
    """ representation of a scalar field
    similar to VectorField class but with simplified functionality
    """

    def __init__(self, phi, dim, *args, **kwargs):
        """ note that the dimension must be specified; it can not be inferred """
        self.basis = symbols('x y z')[0:dim]
        self.phi = phi

    def gradient(self):
        """ returns the gradient of phi as a sympy Matrix (col vector) """
        phi = Array([self.phi])
        gradphi = derive_by_array(phi, self.basis)
        return gradphi.tomatrix()

    def scalar_laplacian(self, kappa):
        """ Returns scalar laplacian term for the field $\nabla \cdot (\kappa \nabla \phi)$
        @param kappa  list of sympy expressions indexed by problem dimension
        """
        scalar_laplacian = 0
        for idx, varsymbol in enumerate(self.basis):
            scalar_laplacian += diff(kappa[idx] * diff(self.phi, varsymbol), varsymbol)
        return scalar_laplacian

    def advection(self, v):
        """ computes the advection term $\nabla\cdot(\phi \vec v) phi advected by symbolic velocity
        field v, given as a list of numpy expressions
        @param v  list of sympy expressions for each velocity component, indexed by dim
        """
        advection = 0
        for idx, varsymbol in enumerate(self.basis):
            advection += diff(self.phi * v[idx], varsymbol)
        return advection

    def time_derivative(self):
        return diff(self.phi, Symbol('t'))
