# -*- coding: utf-8 -*-
"""
@package src.msh.mesh
Created on Sun Apr 24 20:51:08 2011
Contains the Mesh classes

@author: Matt Ueckermann
@author: Corbin Foucart
@note While the elements are certainly numbered CCW, the edges may not be. The
      edge numbering comes from mk_basis.int_el_pqr, and is done in the frame
      of references of the master element, which may not be the same as how
      the element is numbered in real space. Tricky...

"""
import numpy as np
import src.pyutil as util
import code, pdb
import src.msh.util as mshu

import logging
logger = logging.getLogger(__name__)

from src.msh.util import fix, simpvol2D, connect_elm
#import msh.plot as plt
import copy

class Mesh():
    """
    Parent class of Mesh2D and Mesh3D for shared functionality.

    Aim is to prevent duplication of code common to both Mesh2D and Mesh3D classes, since they
    otherwise would not share a parent.
    """

    def get_global_vertex_numbers(self, globEdNum=None):
        """ returns the global vertex numbers given a global edge nubmer of the mesh

        Recall that ed2ed has the following structure for each row (corresponding to a single edge
        in the mesh:

        [(+) elm #, (+) side local edge #, (-) elm#, (-) elm loc edge #, vertex 1, vertex 2...]
                (0:2)                            (2:4)               (4:)
        """
        return self.ed2ed[globEdNum, 4:].ravel()

    def get_adjacent_elm(self, globalEdgeNum, side):
        """
        returns the adjacent element number on the specified side of the given global edge number
        @param globalEdgeNum - the global edge number
        @param side - either 'LEFT' or 'RIGHT' where in ed2ed the LEFT element is given first, then
        the right element.

        # to get the left adjacent element of glboal edge 23
        mesh.get_adjacent_elm(23, 'LEFT')
        """
        # assign the index based on the structure of the ed2ed rows
        if side == 'LEFT':
            elmIdx = 0
        elif side == 'RIGHT':
            elmIdx = 2
        else:
            raise ValueError("Invalid edge side specification")
        return self.ed2ed[globalEdgeNum, elmIdx].ravel()[0]

    def get_adjacent_elm_local_edge(self, globalEdgeNum, side):
        """
        For a global edge, returns the corresponding local edge number for the adjacent element on
        the specified side.

        @param globalEdgeNum - the global edge number
        @param side - either 'LEFT' or 'RIGHT' where in ed2ed the LEFT element is given first, then
        the right element.

        Each global edge has a left element, and possibly a right element (not in the case of a
        boundary). In the case where the adjacent element exists, the global edge is spatially
        corresponds to one of the local edges on the adjacent element. This information is contained
        in the ed2ed connectivity array, and this helper function represents a way for the user of
        the mesh class to retrieve this information without knowing the internals of the mesh data
        structure.

        NOTE: if the element to the right hand side does not exist (boundary case), then the return
        value will be -1.
        """
        # assign the index based on the structure of the ed2ed rows
        if side == 'LEFT':
            elmIdx = 0
        elif side == 'RIGHT':
            elmIdx = 2
        else:
            raise ValueError("Invalid edge side specification")
        return self.ed2ed[globalEdgeNum, elmIdx + 1].ravel()[0]

class Mesh2D(Mesh):
    def __init__(self, elm, vert):
        '''
        We are assuming only two types of elements here. 2D triangles and 2D
        quadrilaterals.

        @param elm (\c int) Numpy array that defines the elements. Each row
                    is an element, and each column indicates a vertex of the
                    element. That is elm[0, 0] gives the global vertex number
                    of vertex 0 of element 0.
        @param vert (\c float) Numpy array that gives the spatial coordinates
                    of the global vertices.

        @note The elm and vert inputs may be inputted as a tuple, for
              convenience.
        @code
        >>> import msh
        >>> t,p = msh.mk.struct2D()
        >>> mesh = msh.Mesh2D(t,p)
        >>> #OR equivalently
        >>> mesh = msh.Mesh2D(msh.mk.struct2D())
        @endcode

        @author Matt Ueckermann
        '''
        dim = 2
        elm, vert = fix(elm, vert)           # CCW ordering of element nodes
        elm = mshu.sort_by_element_type(elm) # Tri first, then quads

        n_elm = len(elm)
        n_tri = (elm[:,3] < 0).sum()

        n_elm_type = [n_tri, n_elm - n_tri]
        if n_tri == 0:     n_elm_type = [n_elm] # all quads
        if n_elm == n_tri: n_elm_type = [n_elm] # all tris

        #Now sort by vertex number
        if n_tri > 0: elm[0:n_tri, :] = mshu.sort_by_vertex_number(elm[0:n_tri, :])
        if n_tri < n_elm: elm[n_tri:n_elm, :] = mshu.sort_by_vertex_number(elm[n_tri:n_elm, :])

        elm_type = np.zeros(n_elm, dtype=int)
        if n_tri == n_elm: u_elm_type = [0]
        elif n_tri == 0:   u_elm_type = [1]
        else:
            u_elm_type = [0, 1]
            elm_type[n_tri:n_elm] = 1

        #create the connectivity matrixes (Which needs the element enumerated type)
        elm2elm, ed2ed = connect_elm(elm, np.array(u_elm_type)[elm_type], dim, u_elm_type)

        ## The element connectivity matrix. elm2elm[i, j] gives the element
        # number which is connected to element i, through edge j of element i.
        self.elm2elm = elm2elm

        ## The edge connectivity matrix.
        # ed2ed[i, 0:2] gives the [element #, local edge #] of the plus-side element.
        # ed2ed[i, 2:4] gives the [element #, local edge #] of the minus-side element.
        # ed2ed[i, 4:] gives the vertices that make up the edge.
        # numbered CCW with outward-point normal (according to Right-hand rule)
        # CF: This seems to indicate that left is plus, right is minus
        self.ed2ed = ed2ed

        ##A boolean array used to select the interior edges only
        self.ids_interior_ed = (self.ed2ed[:, 2] >= 0).ravel()

        ##A boolean array used to select the exterior edges only
        self.ids_exterior_ed = (ed2ed[:, 2] < 0).ravel()

        ## The triangulation matrix that defined each element.
        # elm[i, :] gives the global vertex numbers that make up the element.
        # This matrix is ordered such that the first num2Dtri elements are
        # triangular elements, while the remaining ones are quadrilaterals.
        self.elm = elm

        ## The different unique types of elements in the triangulation.
        # TODO: CF: this is differently abled -- if only there were some type of ... dictionary in
        # python which could take a readable name as a key...
        self.u_elm_type = np.array(u_elm_type, dtype = int)

        ## The element type. u_elm_type[elm_type[i]] gives the type of element
        #for global element number i.
        self.elm_type = elm_type

        ## The edge element type. u_ed_type[elm_type[i]] gives the type of
        #edge element for global edge element number i. For 2D, there is only
        #one edge type -- lines
        self.ed_type = np.zeros(len(self.ed2ed), dtype=int)

        ## The different unique types of edges in the triangulation. for 2D
        # there is only the one type -- lines
        self.u_ed_type = np.array([0], dtype = int)

        ## Gives the total number of elements in the triangulation.
        # The number of triangles is given by n_tri, and the number of
        # quads can be calculated using n_elm-n_tri
        self.n_elm = n_elm

        ## Gives the number of elements of a particular type in the triangulation
        self.n_elm_type = n_elm_type

        ## Gives the total number of edges in the triangulation.
        self.n_ed = len(self.ed2ed)

        ## Gives the number of edge elements of a particular type
        self.n_ed_type = [len(self.ed2ed)]

        ## Array giving the x-y coordinates of the global vertices in the triangulation.
        self.vert = vert

        ##The dimension of the mesh, dim=2, since this Mesh2D is exclusively for 2D meshes.
        self.dim = dim

        ##Vertex map, maps the vertex number from one periodic edge to the
        #other. This map is needed when comparing the orientation of the edge
        #on the element to the orientation of the periodic edge. The element on
        #the right will not have matching vertex numbers, because it's edge
        #used to be a boundary edge, but has disappeared because of the
        #periodicity.
        # EG. in 1D:
        # [0] a1--(A)--a0 1 b0--(B)--b1 [2] ==> 0 --(A)-- 1 --(B)-- 0
        # ed2ed = [A a0  B  b0 1                   ed2ed = [A a0 B b0 1
        #          A a1 -1 -1  0            ==>             A a1 B b1 0]
        #          B b1 -1 -1  2]
        # elm = [0 1                        ==>      elm = [0 1
        #        1 2]                                       1 2]
        #
        #This array is populated in the msh.mk.periodic function
        self.vertmap = None

        #Next we have to build a rather annoying structure -- the elements have
        #global numbers -- however the data is stored/organized according to the
        #element type. So within the element type, the element will have a
        #different number/location. The next structure figures out what that
        #element number is. The same goes for the edge types
        ## The "global element number" to "element number within a type"
        # conversion array. For example, the data for global element number i
        # is stored in field[elm_type[i]][:, :, glob2type_elm[i]].
        self.glob2type_elm = np.zeros(self.n_elm, dtype=int)
        sumtype = [0] * len(self.u_elm_type)
        for i in range(self.n_elm):
            elm_type = self.elm_type[i]
            self.glob2type_elm[i] = sumtype[elm_type]
            sumtype[elm_type] += 1

        ## The "global edge number" to "edge number within a type"
        # conversion array. For example, the data for global edge number i
        # is stored in field_ed[ed_type[i]][:, :, glob2type_ed[i]].
        self.glob2type_ed = np.zeros(self.n_ed, dtype=int)
        sumtype = [0] * len(self.u_ed_type)
        for i in range(self.n_ed):
            ed_type = self.ed_type[i]
            self.glob2type_ed[i] = sumtype[ed_type]
            sumtype[ed_type] += 1

        ##A list of boolian arrays used to select the interior edges only
        self.ids_interior_ed_by_type = [self.ids_interior_ed[self.ed_type == i] \
            for i in range(len(self.n_ed_type))]

        ##A list of boolian arrays used to select the exterior edges only
        self.ids_exterior_ed_by_type = [self.ids_exterior_ed[self.ed_type == i] \
            for i in range(len(self.n_ed_type))]

        ##Index mapping array from ed_type edge id number to ed_bc_type id
        #number. Basically, in the solver we will refer to, for e.g. the
        #data field_ed[i][:, :, j], where j refers to a boundary edge, numbered
        #according to the element-type local id number. The boundary condition
        #data is stored in an array smaller that field_ed, that is, field_ed_bc
        #contains ONLY the boundary condition information, so calling
        #field_ed_bc[i][:, :, j] will exceed the array bounds. Instead we call
        #field_ed_bc[i][:, :, in2ex_bcid[j]].
        #TODO: Determine if this array is actually still needed
        #   (Indexing has been improved since the below was implemented)
        self.in2ex_bd_id = [ex.cumsum()-1 for ex in self.ids_exterior_ed_by_type]

    def fix(self):
        ''' Function that ensures the the elements are properly numbered in
        a counter-clockwise fashion, with no crosses. This function updates the
        elm and vert data members.
        @see msh.util.fix
        '''
        self.elm, self.vert = fix(self.elm, self.vert)

    def vol(self, ids=None):
        return simpvol2D(self.elm, self.vert) if ids is None else simpvol2D(self.elm[ids,:], self.vert)

    def set_bc_ids(self, bc_id_lambda):
        """To change the default id number for boundary conditions, you can
        use this function

        @param bc_id_lambda (\c lambda function) List of lambda functions. The
                            id of the list determines the id of the boundary.
               bc_id_lambda = lambda (p): f(p)
               where p is a numpy array with p.shape = (n_ext_ed, dim) with the
               centroids of the edges. bc_id_lambda[i](p) should evaluate to
               True if that edge should have the id '-i'.

        CF: Note that the list of boundaries are traversed in the order they
        occur in the list bc_id_lambda, so the final ID is the LAST index of
        the containing the lambda function which returns true when called on
        the edge centroid.
        """
        #Find edge centroids
        ids = (self.ids_interior_ed == False).nonzero()[0]
        vts = self.ed2ed[ids, 4:]

        p = np.array([coord[vts].mean(1) for coord in self.vert[:].T]).T

        for i in range(len(bc_id_lambda)):
            self.ed2ed[ids[bc_id_lambda[i](p)], 2:3] = -i - 1

        #Boundary condition information
        tot_bc_ids = -min(self.ed2ed[self.ids_exterior_ed, 2])
        ##Total number of different bc_ids
        self.n_bc_id = tot_bc_ids

    def write_mesh_to_vtk(self, filename):
        """
        author: CF
        write the 2D mesh out to VTK file so that it can be viewed in Paraview
        or some similar software
        """

        pts, conn = self.vert, self.elm
        Points, Cells = vtk.vtkPoints(), vtk.vtkCellArray()

        # add node / connectivity information to VTK object
        for pt in pts:
            Points.InsertNextPoint(pt)

        for cn in conn:
            cell = vtk.vtkTriangle() if cn[-1] == -1 else vtk.vtkQuad()
            for idx, pt in enumerate(cn):
                if pt != -1:
                    cell.GetPointIds().SetId(idx, pt)
            Cells.InsertNextCell(cell)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(Points)
        polydata.SetPolys(Cells)

        # write VTK object to file
        polydata.Modified()
        if vtk.VTK_MAJOR_VERSION <= 5:
            polydata.Update()

        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName(filename);
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()

###############################################################################

class Extrude2D:
    """ This is the base class for translating 2D data to 3D data. This class
    should probably never be actually used.
    """

    def __init__(self, data_2D, data_extrude):
        #set Default length
        self._len = len(data_2D)

    def __len__(self):
        return self._len

    def _key2globids(self, key, length):
        """ By default this class uses numpy arrays for indexing. If the user
        inputs slice data, we need to turn this into a list of ids in numpy
        array format.
        """
        start = key.start
        stop = key.stop
        step = key.step

        #Debugging check
        #print "start:", start, "  stop:", stop, "  step:", step
        if key.stop == None:
            stop = length
        elif key.stop < 0:
            stop = length + key.stop + 1
        #If I don't add this line, I get this strange error where, if I call
        #v3d[:], stop is equal to a very large number...
        #Uncomment the following lines to NOT get the error
        elif stop > length:
            stop = length
        if key.start == None:
            start = 0
        elif key.start < 0:
            start = length + key.start
        if key.step == None:
            step = 1

        #global ids are the indices in the 3D frame of reference (global frame)
        globids = np.array(range(start, stop, step), dtype=int)
        return globids

    def __getitem__(self, key):
        """
        This makes 2D data act like 3D data, but is accessed the same way as
        a normal numpy array. The main purpose of this function is to do the
        user input parsing, which will be inherited.
        """
        #Do some input parsing

        #By default, the x, y, and z-coordinates are returned
        xyz_key = np.array(range(self._dim))

        #Now parse which vertices get returned. There are a number of allowed
        #cases. For the first set, the user only inputs one set of indices:
        #1. User inputs a slice (i.e. Extrude_2D[start:stop:step])
        #2. User inputs a single integer (i.e. Extrude_2D[7])
        #3. User inputs a list (i.e. Extrude_2D[[1,2,3,4,6]])
        #4. User inputs a numpy array (i.e. Extrude_2D[np.array([1,2,3,4,6])])

        #Also allowed is a tuple or list of indices. This tuple should allow
        #for any combination of the above four cases.
        #For example, Extrude_2D(:, [0,2]), Extrude_2D(1, :), and
        # Extrude_2D[[1, [0,2]]] are all allowed.

        #First handle the single-input cases that are not lists or numpy arrays
        if type(key) == slice:
            globids = self._key2globids(key, self._len)
        elif type(key) == int:
            if key < 0:
                globids = np.array([key + self._len], dtype=int)
            else:
                globids = np.array([key], dtype=int)
        #Next do tuples
        elif type(key) == tuple:
            #First index of the tuple
            if type(key[0]) == slice:
                globids = self._key2globids(key[0], self._len)
            elif type(key[0]) == list:
                globids = np.array(key[0], dtype=int)
            elif type(key[0]) == int:
                if key[0] < 0:
                    globids = np.array([key[0] + self._len], dtype=int)
                else:
                    globids = np.array([key[0]], dtype=int)
            else: #numpy array assumed if nothing else
                globids = key[0]
            #second index of the tuple
            if type(key[1]) == slice:
                xyz_key = self._key2globids(key[1], self._dim)
            elif type(key[1]) == list:
                xyz_key = np.array(key[1], dtype=int)
            elif type(key[1]) == int:
                if key[1] < 0:
                    xyz_key = np.array([key[1] + self._dim], dtype=int)
                else:
                    xyz_key = np.array([key[1]], dtype=int)
            else:
                xyz_key = key[1]
        #Now handle the case of the list
        elif type(key) == list:
            if len(key) == 1:
                globids = np.array(key, dtype=int)
            elif (type(key[0]) != list) and (type(key[1]) != list):
                globids = np.array(key, dtype=int)
            else:
                if type(key[0]) == list:
                    globids = np.array(key[0], dtype=int)
                elif type(key[0]) == int:
                    if key[0] < 0:
                         globids = np.array([key[0] + self._len], dtype=int)
                    else:
                        globids = np.array([key[0]], dtype=int)
                else:
                    globids = key[0]
                if type(key[1]) == list:
                    xyz_key = np.array(key[1], dtype=int)
                elif type(key[1]) == int:
                    if key[1] < 0:
                        xyz_key = np.array([key[1] + self._dim], dtype=int)
                    else:
                        xyz_key = np.array([key[1]], dtype=int)
                else:
                    xyz_key = key[1]
        else: #Finally, assumed a single numpy array if nothing else
            globids = np.array(key, dtype=int)

        #Now return the 3D data, using the _return_rule function to figure out
        #exactly what that means.
        return self._return_rule(globids, xyz_key)

    def _return_rule(self, globids, xyz_key):
        """This is the function that determines what is returned and how.
        """
        pass

#Now define the first class that inherits from Extrude_2D
class Vert2D_extrude(Extrude2D):
    """ Let's users interact with a data-set of extrude rules
    and 2D points as though it was a data-set of 3D points.
    """
    def __init__(self, vert, dz):
        """The constructor for this object assigns the required data
        structures.
        @param vert (\c float) Numpy array of 2D vertices
        @param dz   (\c float) Numpy array of \f$\Delta z\f$'s associated with
                               each vertex. dz.shape = (len(vert), nlevels)

        @author Matt Ueckermann
        """
        ##Private copy of the 2D vertices
        self._vert2D = vert
        ##private copy of the \f$\Delta z\f$'s transformed to z-coordinate
        #points
        self._z = np.column_stack((np.zeros((len(vert),1)), -np.cumsum(dz, 1)))
        ##Private copy of the number of z-levels
        self._zlevels = len(dz[0]) + 1
        ##The defined length for this class, in this case the number of vertices
        #times the number of zlevels
        self._len = len(vert) * self._zlevels
        ##The dimension of the problem, in this case clearly 3D
        self._dim = 3
        ##To mimick numpy arrays, we define the shape variable
        self.shape = (self._len, self._dim)

    def _return_rule(self, globids, xyz_key):
        """This routine defines the return rule based on a global id number.
        Basically how to combine the 2D information and extrude information
        to get 3D information
        """
        #Debugging print statements
        #print globids, xyz_key
        vert2D_ids = globids / self._zlevels
        z_ids = globids % self._zlevels

        if np.any(xyz_key == 2):
            #This is a little ugly, but it basically takes columns of 2D
            #vertices and attaches or stacks to that colums of the z-coordinate.
            #THEN it selects the second index that the user wanted.
            return np.column_stack((self._vert2D[vert2D_ids, 0:self._dim-1], \
                self._z[vert2D_ids, z_ids]))[:, xyz_key]
        else:
            return self._vert2D[np.ix_(vert2D_ids, xyz_key)]

#Next define the class that does the 3D vertices.
class Elm2D_extrude(Extrude2D):
    """Let's users interact with a 2D triangulation and the number of extruded
    elements as if it was a 3D triangulation
    """
    def __init__(self, elm, levels):
        """The constructor for this object assigns the required data
        structures.
        @param elm  (\c int) Numpy array of 2D triangulation
        @param levels   (\c int) Number of vertices in the vertical. Note,
                         zlevels = levels + 1, where levels is the number of
                         ELEMENTS in the vertical

        @author Matt Ueckermann
        """
        ## The base 2D triangulation using the global vertex numbers
        self._elm2D = elm * (levels + 1)
        ##Private copy of the number of levels
        self._levels = levels

        ##The defined length for this class, in this case the number of elements
        #times the number of levels
        self._len = len(elm) * self._levels
        ##The dimension of the problem, in this case, means the maximum number
        #of vertices in a 3D element.
        self._dim = 8 #for extruded quads

        ##To mimick numpy arrays, we define the shape variable
        self.shape = (self._len, self._dim)

    def _return_rule(self, globids, xyz_key):
        """This routine defines the return rule based on a global id number.
        Basically how to combine the 2D information and extrude information
        to get 3D information
        """
        #Debugging print statements
        #print globids, xyz_key
        elm2D_ids = globids / self._levels
        #slightly tricky, we multiplied elm2D by zlevels, but since we only have
        #levels element in each column, we divide by levels, not zlevels when
        #figuring out the element ids
        z_ids = np.tile(globids % self._levels, (4, 1)).T

        tmp =  np.column_stack((self._elm2D[elm2D_ids, :] + 1 + z_ids, \
            self._elm2D[elm2D_ids, :] + z_ids))
        tmp[tmp < 0] = -1
        tmp[tmp[:, 7] < 0, 0:7] = \
            tmp[np.ix_(tmp[:, 7] < 0, [0, 1, 2, 4, 5, 6, 7])]
        return tmp[:, xyz_key]

#Now we do the element types
class Elm_type2D_extrude(Extrude2D):
    """Let's users interact with a 2D element connectivity and the # of extruded
    elements as if it was a 3D element connectivity matrix
    """
    def __init__(self, elm_type, levels):
        """The constructor for this object assigns the required data
        structures.
        @param elm_type (\c int) Numpy array of element types for a
                        2D triangulation
        @param levels   (\c int) Number of vertices in the vertical. Note,
                         zlevels = levels + 1, where levels is the number of
                         ELEMENTS in the vertical and zlevels the number of
                         vertices.

        @author Matt Ueckermann
        """
        ## The base 2D triangulation using the element numbers
        self._elm_type2D = elm_type

        ##Private copy of the number of levels
        self._levels = levels

        ##The defined length for this class, in this case the number of elements
        #times the number of levels
        self._len = len(elm_type) * self._levels
        ##The dimension of the problem, in this case, means the maximum number
        #of vertices in a 3D element.
        self._dim = 1 #for extruded quads

        ##To mimick numpy arrays, we define the shape variable
        self.shape = (self._len, self._dim)

    def _return_rule(self, globids, xyz_key):
        """This routine defines the return rule based on a global id number.
        Basically how to combine the 2D information and extrude information
        to get 3D information
        """
        #This one is pretty simply. The elements in the column will have the
        #same type
        elm2D_ids = globids / self._levels

        return self._elm_type2D[elm2D_ids]

#Now we start with the connectivity matrices
class Elm2Elm2D_extrude(Extrude2D):
    """Let's users interact with a 2D element connectivity and the # of extruded
    elements as if it was a 3D element connectivity matrix
    """
    def __init__(self, elm2elm, levels, topbcid=-5, botbcid=-6):
        """The constructor for this object assigns the required data
        structures.
        @param elm2elm (\c float) Numpy array of 2D triangulation
        @param levels   (\c int) Number of vertices in the vertical. Note,
                         levels = levels + 1, where levels is the number of
                         ELEMENTS in the vertical
        @param topbcid (\c int) Negative integer giving the id of the top
                        boundary condition. Default = -1
        @param botbcid (\c int) Negative integer giving the id of the bottom
                        boundary condition. Default = -2

        @note If botbcid == topbcid and > 0, then assume periodic

        @author Matt Ueckermann
        """
        ## The base 2D triangulation using the element numbers
        self._elm2elm2D = elm2elm * (levels)
        #Fix the boundary conditions
        self._elm2elm2D[elm2elm < 0] = elm2elm[elm2elm < 0]

        ##Private copy of the number of levels
        self._levels = levels

        ##The defined length for this class, in this case the number of elements
        #times the number of levels
        self._len = len(elm2elm) * self._levels
        ##The dimension of the problem, in this case, means the maximum number
        #of vertices in a 3D element.
        self._dim = 6 #for extruded quads

        ##To mimick numpy arrays, we define the shape variable
        self.shape = (self._len, self._dim)

        ##The bottom boundary condition id.
        self._botbc = botbcid

        ##The top boundary condition id.
        self._topbc = topbcid

    def _return_rule(self, globids, xyz_key):
        """This routine defines the return rule based on a global id number.
        Basically how to combine the 2D information and extrude information
        to get 3D information
        """
        #Debugging print statements
        #print globids, xyz_key
        elm2D_ids = globids / self._levels
        z_ids = globids % self._levels

        #This is a few-step process to get the boundary conditions correct
        tmp = self._elm2elm2D[elm2D_ids, :]
        #Only affect the non-negative ids
        ids = tmp < 0
        negtemp = tmp[ids]
        tmp = tmp + np.tile(z_ids, (4, 1)).T
        tmp[ids] = negtemp

        tmp =  np.column_stack((globids + 1, globids - 1, tmp))

        #fix top and bottom boundary conditions
        if (self._botbc == self._topbc) and (self._topbc > 0):
            tmp[z_ids == self._levels - 1, 0] = \
                globids[z_ids == self._levels - 1]
            tmp[z_ids == 0, 1] = \
                globids[z_ids == 0] + self._levels - 1
        else:
            tmp[z_ids == self._levels - 1, 0] = self._botbc
            tmp[z_ids == 0, 1] = self._topbc

        return tmp[:, xyz_key]

#Then the edge connectivity matrix
class Ed2Ed2D_extrude(Extrude2D):
    """Let's users interact with a 2D element connectivity and the # of extruded
    elements as if it was a 3D element connectivity matrix
    """
    def __init__(self, ed2ed, elm, levels, topbcid=-5, botbcid=-6):
        """The constructor for this object assigns the required data
        structures.
        @param ed2ed (\c int) Numpy array of 2D triangulation
        @param elm   (\c int) Numpy array defining element in terms of global
                     vertex numbers.
        @param levels   (\c int) Number of vertices in the vertical. Note,
                         zlevels = levels + 1, where levels is the number of
                         ELEMENTS in the vertical
        @param topbcid (\c int) Negative integer giving the id of the top
                        boundary condition. Default = -1
        @param botbcid (\c int) Negative integer giving the id of the bottom
                        boundary condition. Default = -2

        @note If botbcid == topbcid and > 0, then assume periodic

        @author Matt Ueckermann
        """
        ## The base 2D triangulation using the element numbers
        self._ed2ed2D = ed2ed.copy()
        #Do the multiplications to use the global element numbers for the
        #surface mesh
        #first for the element numbers
        ids = ed2ed[:, 0] >= 0
        self._ed2ed2D[ids, 0] = self._ed2ed2D[ids, 0] * levels
        ids = ed2ed[:, 2] >= 0
        self._ed2ed2D[ids, 2] = self._ed2ed2D[ids, 2] * levels
        #then for the vertex numbers (which requires zlevels instead)
        #also add by 1 because it is convenient later. The faces have to
        #be labelled counter clockwise, and start numbering from the bottom
        #vertices. :)
        self._ed2ed2D[:, 4:] = self._ed2ed2D[:, 4:] * (levels + 1) + 1

        ##Private copy of the number of interior edges (not boundary) for the
        # vertical faces
        self._n_ed_in = np.sum(ed2ed[:,2] >= 0)

        ##Private copy of the element triangulation matrix, used for the top
        #and bottom edge connectivity
        self._elm2D = elm * (levels + 1)

        ##Private copy of the number of levels
        self._levels = levels

        ##The defined length for this class, in this case the number of elements
        #times the number of levels
        self._len = len(ed2ed) * self._levels + len(elm) * (self._levels + 1)
        ##The number of columns in the ed2ed matrix. This is equal to 4 + the
        #number of vertices that make up a face. That will be a max of 8 in 3D
        self._dim = 8 #for extruded quads

        ##To mimick numpy arrays, we define the shape variable
        self.shape = (self._len, self._dim)

        ##The bottom boundary condition id.
        self._botbc = botbcid

        ##The top boundary condition id.
        self._topbc = topbcid

    def _return_rule(self, globids, xyz_key):
        """This routine defines the return rule based on a global id number.
        Basically how to combine the 2D information and extrude information
        to get 3D information
        """
        #intialize tmp
        tmp = []

        #The edge ids are divided into 4 major categories:
        #1. Interior vertical edges (0:n_ed_in * levels)
        #2. Interior horizontal edges
        #   <start> : len(_elm2D) * (levels - 1) + <start>
        #3. Boundary vertical edges
        #   <start> : len(ed2ed) - n_ed_in + <start>
        #4. Boundary horizontal edges
        #   <start> : len(_elm2D) * 2 + <start>
        #
        # and each category should be dealt with separately

        #Do the first category
        ids_done = np.array(len(globids) * [False], dtype=bool)
        ids = globids < (self._n_ed_in * self._levels)

        if np.any(ids):
            #print 'first'
            elm2D_ids = globids[ids] / self._levels
            z_ids = globids[ids] % self._levels

            #Note the colstack is needed to connect the two vertices (at different
            # zlevels). The bottom two vertices are already included in the
            #ed2ed matrices
            tmp1 = np.column_stack((self._ed2ed2D[elm2D_ids, :4],
                                    self._ed2ed2D[elm2D_ids, 4] + z_ids,
                                    self._ed2ed2D[elm2D_ids, 5] + z_ids,
                                    self._ed2ed2D[elm2D_ids, 5] - 1 + z_ids,
                                    self._ed2ed2D[elm2D_ids, 4] - 1 + z_ids))
            #modify the element ids to be correct
            ids = tmp1[:, 0] >= 0
            tmp1[ids, 0] = tmp1[ids, 0] + z_ids[ids]
            ids = tmp1[:, 2] >= 0
            tmp1[ids, 2] = tmp1[ids, 2] + z_ids[ids]
            #the local edge ids also need correction
            tmp1[:, 1] = tmp1[:, 1] + 2
            tmp1[:, 3] = tmp1[:, 3] + 2
            #Record which globids have been handled already

            # CF: OLD MPU CODE DOES THIS:
            # ids_done[ids] = True
            #
            # where he applies a boolean mask that's shorter than the actual array. Is this is a
            # shortcut or a bug? Unclear, but explicitly padding to fix the warning.
            padLength = len(ids_done) - len(ids)
            pad = np.zeros(padLength, dtype=bool)
            paddedIds = np.append(ids, pad)
            ids_done[paddedIds] = True
            tmp = tmp1

        #Second category
        start = (self._n_ed_in * self._levels)
        ids = globids < (start + len(self._elm2D) * (self._levels - 1))
        #Don't want to include the ids already done, so remove them
        ids[ids_done] = False

        if np.any(ids):
            #print 'second'
            #This is illusively tricky. There are only levels-1 internal
            #horizontal faces, so when the global face ids are converted to
            #local 2D ids, we have to divide by levels-1, however...
            elm2D_ids = (globids[ids] - start) / (self._levels - 1)
            z_ids = (globids[ids] - start) % (self._levels - 1)

            #... when calculating the global element number we have to multiply
            #the local 2D element number by the TOTAL number of levels.
            tmp2 = np.column_stack((elm2D_ids * self._levels + z_ids + 1,
                                    np.ones_like(z_ids),
                                    elm2D_ids * self._levels + z_ids,
                                    np.zeros_like(z_ids),
                                    self._elm2D[elm2D_ids, 0] + z_ids + 1,
                                    self._elm2D[elm2D_ids, 1] + z_ids + 1,
                                    self._elm2D[elm2D_ids, 2] + z_ids + 1,
                                    self._elm2D[elm2D_ids, 3] + z_ids + 1))

            ids_done[ids] = True


            #correct the final column of tmp2
            ids = self._elm2D[elm2D_ids, 3] < 0
            tmp2[ids, -1] = -1
            if len(tmp):
                tmp = np.concatenate((tmp, tmp2))
            else:
                tmp = tmp2

        #Third category
        start = start + len(self._elm2D) * (self._levels - 1)
        ids = globids < ( start + \
            (len(self._ed2ed2D) - self._n_ed_in) * self._levels )
        #Don't want to include the ids already done, so remove them
        ids[ids_done] = False

        if np.any(ids):
            #print 'third'
            #Record which globids have been handled already
            ids_done[ids] = True
            #There is an offset here because for the vertical faces, we only
            #deal with the boundary elements of the 2D mesh
            elm2D_ids = (globids[ids] - start) / self._levels + self._n_ed_in
            z_ids = (globids[ids] - start) % self._levels

            #Note the colstack is needed to connect the two vertices (at different
            # zlevels). The bottom two vertices are already included in the
            #ed2ed matrices
            tmp1 = np.column_stack((self._ed2ed2D[elm2D_ids, :4], \
                self._ed2ed2D[elm2D_ids, 4] + z_ids, \
                self._ed2ed2D[elm2D_ids, 5] + z_ids, \
                self._ed2ed2D[elm2D_ids, 5] - 1 + z_ids, \
                self._ed2ed2D[elm2D_ids, 4] - 1 + z_ids))
            #modify the element ids to be correct
            ids = tmp1[:, 0] >= 0 #should really be all of them
            tmp1[ids, 0] = tmp1[ids, 0] + z_ids[ids]
            #ids = tmp1[:, 2] >= 0
            #tmp1[ids, 2] = tmp1[ids, 2] + z_ids[ids]
            #the local edge ids also need correction
            tmp1[:, 1] = tmp1[:, 1] + 2
            #tmp1[:, 3] = tmp1[:, 3] + 2

            if len(tmp):
                tmp = np.concatenate((tmp, tmp1))
            else:
                tmp = tmp1

        #Fourth category
        start = start + (len(self._ed2ed2D) - self._n_ed_in) * self._levels
        ids = globids < (start + len(self._elm2D) * (2))
        #Don't want to include the ids already done, so remove them
        ids[ids_done] = False

        if np.any(ids):
            #print 'fourth'
            #This is illusively tricky. There are only levels-1 internal
            #horizontal faces, so when the global face ids are converted to
            #local 2D ids, we have to divide by levels-1, however...
            elm2D_ids = (globids[ids] - start) / (2)
            z_ids = (globids[ids] - start) % (2)

            #... when calculating the global element number we have to multiply
            #the local 2D element number by the TOTAL number of levels.
            tmp2 = np.column_stack((\
                elm2D_ids * self._levels + z_ids * (self._levels - 1), \
                np.zeros_like(z_ids), \
                np.ones((len(z_ids), 2), dtype=int), \
                self._elm2D[elm2D_ids, 0] + z_ids * (self._levels), \
                self._elm2D[elm2D_ids, 1] + z_ids * (self._levels), \
                self._elm2D[elm2D_ids, 2] + z_ids * (self._levels), \
                self._elm2D[elm2D_ids, 3] + z_ids * (self._levels)))

            ids_done[ids] = True

            #now fix the boundary conditions and local face numbering
            ids = (z_ids == 0)
            tmp2[ids, 2] = self._topbc
            tmp2[ids, 1] = 1
            ids = (z_ids == 1)
            tmp2[ids, 2] = self._botbc
            #The bottom boundaries also need to have the vertex ordering
            #re-order, so that the bottom normal will be OUTWARD pointing
            swtch= np.array([[6],[4]])
            swtch2= np.array([[4],[6]])
            tmp2[ids, swtch] = tmp2[ids, swtch2]

            if len(tmp):
                tmp = np.concatenate((tmp, tmp2))
            else:
                tmp = tmp2

        return tmp[:, xyz_key]

#===============================================================================
# ##############################################################################
#===============================================================================
class Mesh3D(Mesh):
    """This is the base class for 3D meshes"""
    def __init__(self):
        pass

    def __len__(self):
        return self.n_elm

    def set_bc_ids(self, bc_id_lambda):
        """To change the default id number for boundary conditions, you can
        use this function

        @param bc_id_lambda (\c lambda function) List of lambda functions. The
                            id of the list determines the id of the boundary.
               bc_id_lambda = lambda (p): f(p)
               where p is a numpy array with p.shape = (n_ext_ed, dim) with the
               centroids of the edges. bc_id_lambda[i](p) should evaluate to
               True if that edge should have the id '-i'.

        """
        if bc_id_lambda != None:
            #Find edge centroids
            ids = (self.ids_interior_ed == False).nonzero()[0]
            vts = self.ed2ed[ids, 4:]

            p = np.array([coord[vts].mean(1) for coord in self.vert[:].T]).T
            ids_tri = self.ed_type[ids] == False
            p[ids_tri, :] = np.array([coord[vts[ids_tri, :3]].mean(1)\
                for coord in self.vert[:].T]).T
            for i in range(len(bc_id_lambda)):
                self.ed2ed[ids[bc_id_lambda[i](p)], 2:3] = -i - 1
        else:
            print("Input function to set_bcs_ids was 'None'." + \
                " New boundary conditions id's were not set.")

#===============================================================================
# Mesh3D_Extrude
#===============================================================================
class Mesh3D_extrude(Mesh3D):
    """This class acts as a container for a 3D mesh defined by a 2D surface
    mesh that is extruded down cumulatively by dz, defined at each grid-point
    """
    def __init__(self, mesh2D, dz, topbcid=-5, botbcid=-6):
        '''
        This initializes the extruded mesh object, and can henceforth be used
        as though it was a 3D mesh object.

        @param mesh2D (\c Mesh2D object) An instance of the Mesh2D class.
        @param dz (\c int) Array that keeps track of the \f$\Delta z\f$'s
                           defined at each vertex in the mesh.
        @param topbcid (\c int) Negative integer giving the id of the top
                        boundary condition. Default = -1
        @param botbcid (\c int) Negative integer giving the id of the bottom
                        boundary condition. Default = -2

        @author Matt Ueckermann
        '''
        #Validate input
        assert(topbcid < 0 and botbcid < 0)

        ##A copy of the 2D mesh from which the 3D mesh is created
        self.mesh2D = mesh2D

        ## The number of vertically structured levels
        self.levels = len(dz[0])

        ##Negative integer giving the id of the top boundary condition.
        #Default = -1
        self.topbcid = topbcid

        ##Negative integer giving the id of the bottom
        # boundary condition. Default = -2
        self.botbcid = botbcid

        ## The element connectivity matrix. elm2elm[i, j] gives the element
        # number which is connected to element i, through edge j of element i.
        self.elm2elm = Elm2Elm2D_extrude(mesh2D.elm2elm, self.levels, \
                                         self.topbcid, self.botbcid)
        #Now make a real actual copy of this array
        self.elm2elm = self.elm2elm[:]

        #Make an class that makes the 3D array of ed2ed
        ed2ed = Ed2Ed2D_extrude(mesh2D.ed2ed, mesh2D.elm, self.levels, \
                                     self.topbcid, self.botbcid)
        ## The edge connectivity matrix.
        # ed2ed[i, 0:2] gives the [element #, local edge #] of the plus-side
        # element.
        # ed2ed[i, 2:4] gives the [element #, local edge #] of the minus-side
        # element.
        # ed2ed[i, 4:] gives the vertices that make up the edge. These are
        #numbered count-clockwise with outward-point normal (according to
        #Right-hand rule)
        self.ed2ed = ed2ed[:] #Make an actual in-memory array

        ##A boolian array used to select the interior edges only
        self.ids_interior_ed = (ed2ed[:, 2] >= 0).ravel()

        ##A boolian array used to select the exterior edges only
        self.ids_exterior_ed = (ed2ed[:, 2] < 0).ravel()

        ## The triangulation matrix that defined each element.
        # elm[i, :] gives the global vertex numbers that make up the element.
        # This matrix is ordered such that the first num2Dtri elements are
        # triangular elements, while the remaining ones are quadrilaterals.
        self.elm = Elm2D_extrude(mesh2D.elm, self.levels)

        ## The different or unique types of elements in the triangulation.
        self.u_elm_type = copy.copy(mesh2D.u_elm_type)
        #Triangles become prisms
        self.u_elm_type[self.u_elm_type == 0] = 2

        ## The different or unique types of edge elements in the triangulation.
        self.u_ed_type = []
        for npe in self.u_elm_type:
            if npe == 0:
                self.u_ed_type.append(0)
            elif npe == 1:
                self.u_ed_type.append(1)
            elif npe == 2:
                self.u_ed_type.append(1)
                self.u_ed_type.append(0)
        self.u_ed_type = util.unique(self.u_ed_type)

        #Instead of creating a 2D version and extruding -- it's simpler just
        #to make the whole array -- plus it's a reasonably small array
        #in any case
        ## The edge element type. elm_type[i] gives the type of edge element
        #for global edge element number i. -- where the real type is given in
        #u_ed_type
        self.ed_type = np.array((self.ed2ed[:, -1] >= 0).ravel(), dtype=bool)
        #If there are only rectangles, the relative edge type should be zero!
        if all(self.u_ed_type == 1):
            self.ed_type[:] = 0

        #make the elm_type 2D-3D class
        elm_type = Elm_type2D_extrude(mesh2D.elm_type, self.levels)
        ## The element type. elm_type[i] gives the type of element for global
        # element number i, where the real type is given in u_elm_type.
        self.elm_type = elm_type[:] #actual in-memory copy of array

        ## Gives the total number of elements in the triangulation.
        # The number of triangles is given by num2Dtri, and the number of
        # quads can be calculated using n_elm-num2Dtri
        self.n_elm = len(self.elm_type)

        ## Gives the number of 2D triangular elements in the triangulation.
        self.n_elm_type = [tp * self.levels for tp in mesh2D.n_elm_type]

        ## Gives the total number of edges in the triangulation.
        self.n_ed = len(self.ed2ed)

        ## Gives the number of edges in the triangulation of a particular type.
        self.n_ed_type = [sum(self.ed_type == i) for i in range(len(self.u_ed_type))]

        ## Array giving the x-y-z coordinates of the global vertices in the
        # triangulation.
        self.vert = Vert2D_extrude(mesh2D.vert, dz)

        ##The height-map for the 2D vertices (or bathymetry)
        self.h_map = self.vert._z[:, -1].reshape(len(self.vert._z), 1)

        #Convert vertices to a real numpy array
        self.vert = self.vert[:]

        ##The dimension of the mesh, dim=2, since this Mesh2D is exclusively
        # for 2D meshes.
        self.dim = mesh2D.dim + 1

        ##Vertex map, maps the vertex number from one periodic edge to the
        #other. This map is needed when comparing the orientation of the edge
        #on the element to the orientation of the periodic edge. The element on
        #the right will not have matching vertex numbers, because it's edge
        #used to be a boundary edge, but has disappeared because of the
        #periodicity.
        # EG. in 1D:
        # [0] a1--(A)--a0 1 b0--(B)--b1 [2] ==> 0 --(A)-- 1 --(B)-- 0
        # ed2ed = [A a0  B  b0 1                   ed2ed = [A a0 B b0 1
        #          A a1 -1 -1  0            ==>             A a1 B b1 0]
        #          B b1 -1 -1  2]
        # elm = [0 1                        ==>      elm = [0 1
        #        1 2]                                       1 2]
        #
        #This array is populated in the msh.mk.periodic function
        self.vertmap = None

    def write_mesh_to_vtk(self, filename):
        """
        @author foucartc
        write the 3D mesh out to VTK file so that it can be viewed in Paraview
        or some similar software
        """
        pts, conn = self.vert, self.elm[:]
        Points, Cells = vtk.vtkPoints(), vtk.vtkCellArray()

        # add node / connectivity information to VTK object
        for pt in pts:
            Points.InsertNextPoint(pt)

        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(Points)

        for idx, cn in enumerate(conn):
            if cn[-1] == -1:
                cell = vtk.vtkWedge()
                cnRef = cn[0:6]
                for idx,pt in enumerate(cnRef):
                    cell.GetPointIds().SetId(idx,pt)

            else:
                cell = vtk.vtkHexahedron()
                for idx, pt in enumerate(cn):
                    cell.GetPointIds().SetId(idx, pt)
            grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())


        writer = vtk.vtkXMLUnstructuredGridWriter();
        writer.SetInputData(grid)
        writer.SetFileName(filename);
        writer.Write()
