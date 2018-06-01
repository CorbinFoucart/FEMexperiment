Contains routines for creation of 1D and 2D master elements

## 1D

Newly written functionality for 1D master elements.
- `polynomials_1D.py` contains library for dealing with 1D polynomials and vandermonde matrices
- `nodal_basis_1D` contains class definition for 1D nodal basis
- `master_1D.py` contains class definition for 1D master element, built on a nodal basis

## 2D, 3D

The rest of the functionality is from the MPU authored `mk_basis` and `mk_master`, which I still
have to clean out and re-write in a more modular way.
