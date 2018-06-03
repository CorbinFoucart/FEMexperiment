'''
Created on Jul 9, 2010

@author: Matt Ueckermann
'''
import numpy as np
from msh.util import connect_elm2ed
from master.mk_basis import int_el_pqr
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
try:
    from enthought.tvtk.api import tvtk
except:
    from tvtk.api import tvtk
try:
    from enthought.mayavi import mlab, sources
except:
    from mayavi import mlab, sources

def meshplot(mesh, labels=None, vert=False, elm=False, ed=False, \
    scale=[1,1,1], elm_ids=[], offscreen=False, savefig=None, is2D=False):
        '''Function to plot the mesh, with various labelling options

        @param mesh  A mesh object
        @param labels   One interface fo labeling. labels is a list of size 3
                        of type bool. Then entries in labels corresponds to
                        labels = [vert, elm, ed] (See below)
        @param vert (\c bool) Flags wether or not to plot vertex labels. False
                               by default.
        @param elm  (\c bool) Flags wether or not to plot element labels. False
                               by default.
        @param ed   (\c bool) Flags wether or not to plot edge labels. False
                               by default.
        @param scale (\c int) List indicating the scaling of the vertices --
                    this is used to scale the z-direction for a thin mesh. By
                    default, there is no scaling, i.e. scale = [1, 1, 1]

        @param elm_ids Numpy array of elements that will be plotted. If left
                    empty, all elements are plotted. This is closely used with
                    the labels inputs -- labels are expensive (memory-wise) to
                    plot in 3D, so only a limited number of elements can be
                    comfortably plotted with labels in 3D.
        @param offscreen ?
        @param savefig ?
        @param is2D ?
        '''
        if offscreen:
            mlab.options.offscreen = True
            if  savefig == None:
                savefig = 'out.png'

        if labels==None:
            labels = [vert, elm, ed]
        if mesh.dim == 2:
            if any(labels) or is2D:
                mshplot2D(mesh.elm, mesh.vert, mesh.ed2ed, labels)
            else:
                mshplot3D(mesh)
        elif mesh.dim == 3:
            mshplot3D(mesh, scale, elm_ids, labels)

        if savefig != None:
            mlab.savefig(savefig)

def mshplot2D(t, p, ed=[], labels=[0, 0, 0], dgnodes=None, savename=None):
    ''' Plot 2D meshes (and optionally label vertices, elements, and edges)
    @param t (\c int) Triangulation matrix
    @param p (\c float) Vertex matrix
    @param ed (\c int) Edge connectivity matrix
    @param labels (\c bool) List of three elements indicateting what should be
                           labeled. By default, nothing is labeled
                           labels[0]=True labels vertices
                           labels[1]=True labels elements
                           labels[2]=True labels edges
    @param dgnodes A list of node locations in space
    @param savename Name of the file in which to save the figure
    @see msh.mesh.Mesh2D

    @author Matt Ueckermann
    '''
    plt.figure()
    plt.gca().set_aspect('equal')
    ax = plt.gca()

    ids2Dquad = (t[:, 3] >= 0)
    if not all(ids2Dquad):
        plt.triplot(p[:, 0], p[:, 1], t[ids2Dquad == False, 0:3], 'g-')

    #Plotting the quads takes a little more effort:
    if any(ids2Dquad):
        verts = p[t[ids2Dquad, :], :2]

        codes = [Path.MOVETO]
        vertices = [0, 0]
        for i in xrange(len(verts)):
            #These codes directly tell the patch object what to do with each
            #of the vertices. len(codes) = len(vertices)
            codes += [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
            vertices = np.vstack([vertices, np.vstack([verts[i], [0, 0]])])
            #Note, the final [0,0] vertex is just a dummy vertex for the
            #command CLOSEPOLY

        #Make sure vertices is a numpy array, then create the compound path
        #object
        vertices = np.array(vertices, float)
        path = Path(vertices, codes)

        #Now make the pathpatch object that will actually be plotted
        pathpatch = PathPatch(path, facecolor='None', edgecolor='blue')

        ax.add_patch(pathpatch)

    if labels[0]:
        for i in xrange(len(p)):
            ax.text(p[i, 0], p[i, 1], i, color=(0, 0, 0))
    if labels[1]:
        for i in xrange(len(t)):
            if ids2Dquad[i]:
                x = np.mean(p[t[i, :], 0])
                y = np.mean(p[t[i, :], 1])
                col = 'blue'
            else:
                x = np.mean(p[t[i, 0:3], 0])
                y = np.mean(p[t[i, 0:3], 1])
                col = 'green'
            ax.text(x, y, i, color=col)
    if labels[2]:
        for i in xrange(len(ed)):
            x = np.mean(p[ed[i, 4:], 0])
            y = np.mean(p[ed[i, 4:], 1])
            col = 'red'

            ax.text(x, y, i, color=col)

    if dgnodes!=None:
        decor = ['g.', 'b.']
        i =0
        for dgn in dgnodes:
            plt.plot(dgn[:,0,:].ravel(), dgn[:, 1, :].ravel(), decor[i])
            i+=1
    plt.axis('tight')
    plt.title('2D Mesh')
    plt.gca().set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    if savename == None:
        plt.show()
    else:
        plt.savefig(savename)

def connect_check_plot2D(mesh, elm=True, ed=True):
    ''' Plot 2D meshes (and optionally label vertices, elements, and edges)
    @param mesh (\c msh.mesh.mesh2D) Mesh object
    @param elm (\c bool) Check/plot element connectivity
    @param ed (\c bool) Check/plot edge connectivity

    @author Matt Ueckermann
    '''
    plt.gcf()
    plt.gca().set_aspect('equal')

    if elm:
        #element connectivity check
        for i in xrange(len(mesh.elm2elm)):
            for j in range(len(mesh.elm2elm[0])):
                if mesh.elm2elm[i, j] >= 0:
                    ne = mesh.elm_type[i] + 3
                    x1 = np.mean(mesh.vert[mesh.elm[i, 0:ne], 0])
                    y1 = np.mean(mesh.vert[mesh.elm[i, 0:ne], 1])
                    ii = mesh.elm2elm[i, j]
                    ne = mesh.elm_type[ii] + 3
                    x2 = np.mean(mesh.vert[mesh.elm[ii, 0:ne], 0])
                    y2 = np.mean(mesh.vert[mesh.elm[ii, 0:ne], 1])
                    plt.plot([x1, x2], [y1, y2], 'g--', linewidth=3)
                    plt.plot(x1, y1, 'xb')
                    plt.plot(x2, y2, 'og')
    if ed:
        #edge connectivity check
        for j in xrange(len(mesh.ed2ed)):
            i = mesh.ed2ed[j, 0]
            ne = mesh.elm_type[i] + 3
            x1 = np.mean(mesh.vert[mesh.elm[i, 0:ne], 0])
            y1 = np.mean(mesh.vert[mesh.elm[i, 0:ne], 1])
            if mesh.ed2ed[j, 2] >= 0:
                ii = mesh.ed2ed[j, 2]
                ne = mesh.elm_type[ii] + 3
                x2 = np.mean(mesh.vert[mesh.elm[ii, 0:ne], 0])
                y2 = np.mean(mesh.vert[mesh.elm[ii, 0:ne], 1])
            else:
                x2 = np.mean(mesh.vert[mesh.ed2ed[j, 4:], 0])
                y2 = np.mean(mesh.vert[mesh.ed2ed[j, 4:], 1])

            plt.plot([x1, x2], [y1, y2], 'r-')
            plt.plot(x1, y1, 'xr')
            plt.plot(x2, y2, 'og')

    plt.show()

def mshplot3D(mesh, scale=[1, 1, 1], elm_ids=[], labels=[0, 0, 0]):
    ''' Plot 3D meshes (and optionally label vertices, elements, and edges)

    @param mesh A 3D mesh object

    @param elm_ids Numpy array of elements that will be plotted. If left empty,
                   all elements are plotted.

    @param labels (\c bool) List of three elements indicateting what should be
                           labeled. By default, nothing is labeled
                           labels[0]=True labels vertices
                           labels[1]=True labels elements
                           labels[2]=True labels edges

    @param scale (\c int) List indicating the scaling of the vertices -- this
                          is used to scale the z-direction for a thin mesh. By
                          default, there is no scaling, i.e. scale = [1, 1, 1]

    @see msh.mesh.Mesh3D

    @author Matt Ueckermann
    '''
    if type(elm_ids) is not int:
        if len(elm_ids) == 0:
            elm_ids = np.arange(len(mesh.elm))
    #Figure out how many vertices are in each element type
    n_vert_in_type = [len(int_el_pqr(element=element, dim=3)[0]) \
        for element in mesh.u_elm_type]

    #Now create the TVTK data-structures for the 'cells' and 'offsets'
    #cells give [#verts, vert1id, vert2id...,vertnid, #verts ...]
    #offsets gives the id's in the cells list where #verts are listed. So above
    # it is [0, n+1]
    cells = np.array([], dtype=int)
    offset = np.array([], dtype=int)
    for elm, elm_type in zip(mesh.elm[elm_ids, :], mesh.elm_type[elm_ids]):
        n_vt = n_vert_in_type[elm_type]
        offset = np.append(offset, len(cells))
        cells = np.append(cells, [n_vt] + elm[:n_vt].tolist())

    #Also have to create a list of element-types -- to do that we have to
    #convert from my numbering system to the TVTK numbering system
    if mesh.dim == 3:
        type_convert = np.array([tvtk.Tetra().cell_type,\
            tvtk.Hexahedron().cell_type, tvtk.Wedge().cell_type])
    elif mesh.dim == 2:
        type_convert = np.array([tvtk.Triangle().cell_type,\
            tvtk.Quad().cell_type])
    cell_types = mesh.u_elm_type[mesh.elm_type[elm_ids]]
    cell_types = type_convert[cell_types]

    #To help visualize the mesh, we color it be the distance from the origin
    if mesh.dim == 3:
        x, y, z = mesh.vert[:].T
    elif mesh.dim == 2:
        x, y = mesh.vert[:,:2].T
        z = np.zeros_like(x)
    dist = np.sqrt(x**2 + y**2 + z**2)
    #and we scale the x, y, z, coordinates according the the assigned 'scale'
    points = np.column_stack((x * scale[0], y * scale[1], z * scale[2]))

    #Now we create the data-structures
    cell_array = tvtk.CellArray()
    cell_array.set_cells(len(offset), cells)

    ug = tvtk.UnstructuredGrid(points=points)
    ug.set_cells(cell_types, offset, cell_array)
    ug.point_data.scalars = dist.ravel()
    ug.point_data.scalars.name = 'Distance from Origin'

    #Next we set the new data-structures as a mayavi source
    src = sources.vtk_data_source.VTKDataSource(data=ug)

    #Extract the edges from the grid
    edges = mlab.pipeline.extract_edges(src)

    #Use a shorter name for the elements
    elm = mesh.elm[elm_ids, :]

    if any(labels) and len(elm) > 20:
        string = "WARNING:\nAre you sure you want to label more than 20" + \
            "elements in 3D? -- Labels are very memory" + \
            "(apparently -- who would have guessed?) \nY(es) to proceed:"
        answer = raw_input(string)

        if answer.lower() not in ['yes','y','yes']:
            labels = [0, 0, 0]


    #Next add labels if desired:
    maxdist = np.max(dist) / 2.
    if labels[0]: #Vertex labels
        for i in xrange(len(offset)):
            for vnum in elm[i, :]:
                if vnum >= 0:
                    mlab.text3d(points[vnum, 0], points[vnum, 1], \
                        points[vnum, 2], '%d' % vnum, \
                        color=(0, 0, 0), scale=0.02*maxdist)
    if labels[1]: #element labels
        for i in xrange(len(offset)):
            if elm_ids[i] < 0:
                elm_label = mesh.numelm + elm_ids[i]
            else:
                elm_label = elm_ids[i]
            x = np.mean(points[elm[i, : cells[offset[i]]], 0])
            y = np.mean(points[elm[i, : cells[offset[i]]], 1])
            z = np.mean(points[elm[i, : cells[offset[i]]], 2])
            mlab.text3d(x, y, z, '%d' % elm_label, color=(0.3, 0.3, 0.9), \
                scale=0.03*maxdist)
    if labels[2]: #edge labels
        if len(elm_ids) < len(mesh.elm):
            elm2ed = connect_elm2ed(mesh.elm2elm[:], mesh.ed2ed)
            ed_ids = np.unique(elm2ed[elm_ids])
            ed_ids = ed_ids[ed_ids >= 0]
            ed2ed = mesh.ed2ed[ed_ids, :]
        else:
            ed_ids = np.arange(len(mesh.ed2ed))
            ed2ed = mesh.ed2ed[:]

        for i in xrange(len(ed2ed)):
            n = 8
            if ed2ed[i, -1] < 0:
                n = 7
            x = np.mean(points[ed2ed[i, 4:n], 0])
            y = np.mean(points[ed2ed[i, 4:n], 1])
            z = np.mean(points[ed2ed[i, 4:n], 2])
            mlab.text3d(x, y, z, '%d' % ed_ids[i], color=(0.3, 0.9, 0.3), \
                scale=0.03*maxdist)

    #And plot them!
    mlab.pipeline.surface(edges, opacity=0.4, line_width=2)
    mlab.axes()
    mlab.title('3D mesh', size=0.5, height=0.95)
    #mlab.show()

    return cells, offset, cell_types
