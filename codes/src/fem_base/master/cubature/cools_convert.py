# -*- coding: utf-8 -*-
""" A module that is used to create the quadrature tables. Basically translates
the Cools encyclopedia tables. The main function is mk_tables() at the end of
the file.

    Cools, R. "Monomial Cubature Rules Since "Stroud": A Compilation--Part 2."
    J. Comput. Appl. Math. 112, 21-27, 1999.

    Cools, R. "Encyclopaedia of Cubature Formulas."
    http://www.cs.kuleuven.ac.be/~nines/research/ecf/ecf.html

Created on Thu Jun  2 10:02:45 2011

@author: Matt Ueckermann
"""
from pyutil import unique as uniquerows
from numpy import array, dot
from master.mk_basis import int_el_pqr as get_verts

def parse_dim3elm0(Gen_type, pts, weight):
    """ This function generates the actual points based on the generators in
    Cools encyclopedia.

    It does so for 3D tetrahedrals (dim = 3, elm = 0)
    """
    #The area of the tetrahedral is 4/3, so we need to multiply the weight by 8
    weight = weight * 8.0

    pts = pts + (1. - pts[0] - pts[1] - pts[2], )
    #Now, to account for floating point error, we have to make sure that
    #if the last (calculated) point is very close to one of the other points,
    #we replace it with the other point

    if abs(pts[3] - pts[2]) < 1e-14:
        pts = (pts[0], pts[1], pts[2], pts[2])
    elif abs(pts[3] - pts[1]) < 1e-14:
        pts = (pts[0], pts[1], pts[2], pts[1])
    elif abs(pts[3] - pts[0]) < 1e-14:
        pts = (pts[0], pts[1], pts[2], pts[0])

    if Gen_type == 'Fully symmetric':

        addpts = [[pts[0], pts[1], pts[2], pts[3], weight], \
            [pts[1], pts[0], pts[2], pts[3], weight], \
            [pts[2], pts[1], pts[0], pts[3], weight], \
            [pts[1], pts[2], pts[0], pts[3], weight], \
            [pts[0], pts[2], pts[1], pts[3], weight], \
            [pts[2], pts[0], pts[1], pts[3], weight], \
            [pts[3], pts[1], pts[2], pts[0], weight], \
            [pts[1], pts[3], pts[2], pts[0], weight], \
            [pts[2], pts[1], pts[3], pts[0], weight], \
            [pts[1], pts[2], pts[3], pts[0], weight], \
            [pts[3], pts[2], pts[1], pts[0], weight], \
            [pts[2], pts[3], pts[1], pts[0], weight], \
            [pts[0], pts[3], pts[2], pts[1], weight], \
            [pts[3], pts[0], pts[2], pts[1], weight], \
            [pts[2], pts[3], pts[0], pts[1], weight], \
            [pts[3], pts[2], pts[0], pts[1], weight], \
            [pts[0], pts[2], pts[3], pts[1], weight], \
            [pts[2], pts[0], pts[3], pts[1], weight], \
            [pts[0], pts[1], pts[3], pts[2], weight], \
            [pts[1], pts[0], pts[3], pts[2], weight], \
            [pts[3], pts[1], pts[0], pts[2], weight], \
            [pts[1], pts[3], pts[0], pts[2], weight], \
            [pts[0], pts[3], pts[1], pts[2], weight], \
            [pts[3], pts[0], pts[1], pts[2], weight], \
            ]
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

        #However, currently the points are in barycentric coordinates -- which
        #we certainly do not want. We want to convert all of these to
        #coordinates within OUR master triangle.
        verts, ed_ids = get_verts(element=0, dim=3)
        addpts = [(verts[0, :] * r + verts[1, :] * s + verts[2, :] * t \
            + verts[3, :] * u).tolist() +
                   [w] for r, s, t, u, w, in addpts]
    else:
        print "Unknown gentype in parse_dim3elm0:", Gen_type
        raise RuntimeError('...')
    return addpts, n_pts

def parse_dim3elm1(Gen_type, pts, weight):
    """ This function generates the actual points based on the generators in
    Cools encyclopedia.

    It does so for 3D cubes (dim = 3, elm = 1)
    """
    if Gen_type == 'Origin':
        addpts = [[pts[0], pts[1], pts[2], weight]]
        n_pts = 1

    elif Gen_type == 'Fully symmetric':
        addpts = [[pts[0], pts[1], pts[2], weight], \
            [-pts[0], pts[1], pts[2], weight], \
            [pts[0], -pts[1], pts[2], weight], \
            [-pts[0], -pts[1], pts[2], weight], \
            [pts[1], pts[0], pts[2], weight], \
            [-pts[1], pts[0], pts[2], weight], \
            [pts[1], -pts[0], pts[2], weight], \
            [-pts[1], -pts[0], pts[2], weight], \
            [pts[0], pts[1], -pts[2], weight], \
            [-pts[0], pts[1], -pts[2], weight], \
            [pts[0], -pts[1], -pts[2], weight], \
            [-pts[0], -pts[1], -pts[2], weight], \
            [pts[1], pts[0], -pts[2], weight], \
            [-pts[1], pts[0], -pts[2], weight], \
            [pts[1], -pts[0], -pts[2], weight], \
            [-pts[1], -pts[0], -pts[2], weight], \
            [pts[2], pts[1], pts[0], weight], \
            [-pts[2], pts[1], pts[0], weight], \
            [pts[2], -pts[1], pts[0], weight], \
            [-pts[2], -pts[1], pts[0], weight], \
            [pts[1], pts[2], pts[0], weight], \
            [-pts[1], pts[2], pts[0], weight], \
            [pts[1], -pts[2], pts[0], weight], \
            [-pts[1], -pts[2], pts[0], weight], \
            [pts[2], pts[1], -pts[0], weight], \
            [-pts[2], pts[1], -pts[0], weight], \
            [pts[2], -pts[1], -pts[0], weight], \
            [-pts[2], -pts[1], -pts[0], weight], \
            [pts[1], pts[2], -pts[0], weight], \
            [-pts[1], pts[2], -pts[0], weight], \
            [pts[1], -pts[2], -pts[0], weight], \
            [-pts[1], -pts[2], -pts[0], weight], \
            [pts[0], pts[2], pts[1], weight], \
            [-pts[0], pts[2], pts[1], weight], \
            [pts[0], -pts[2], pts[1], weight], \
            [-pts[0], -pts[2], pts[1], weight], \
            [pts[2], pts[0], pts[1], weight], \
            [-pts[2], pts[0], pts[1], weight], \
            [pts[2], -pts[0], pts[1], weight], \
            [-pts[2], -pts[0], pts[1], weight], \
            [pts[0], pts[2], -pts[1], weight], \
            [-pts[0], pts[2], -pts[1], weight], \
            [pts[0], -pts[2], -pts[1], weight], \
            [-pts[0], -pts[2], -pts[1], weight], \
            [pts[2], pts[0], -pts[1], weight], \
            [-pts[2], pts[0], -pts[1], weight], \
            [pts[2], -pts[0], -pts[1], weight], \
            [-pts[2], -pts[0], -pts[1], weight], \
            ]
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    elif Gen_type == 'Central symmetry':
        addpts = [[pts[0], pts[1], pts[2], weight], \
            [-pts[0], -pts[1], -pts[2], weight]]
        n_pts = 2

    elif Gen_type == 'Rectangular symmetry':
        addpts = [[pts[0], pts[1], pts[2], weight], \
            [-pts[0], -pts[1], pts[2], weight], \
            [-pts[0], pts[1], pts[2], weight], \
            [pts[0], -pts[1], pts[2], weight], \
            [pts[0], pts[1], -pts[2], weight], \
            [-pts[0], -pts[1], -pts[2], weight], \
            [-pts[0], pts[1], -pts[2], weight], \
            [pts[0], -pts[1], -pts[2], weight]]
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    elif Gen_type == 'Rotational symmetry':
        #{ I, Rxy, R2xy, R3xy, Ryz, R2yz, R3yz, Rxz, R2xz, R3xz, 10
        #Rxy Rxz, R2xy Rxz, R3xy Rxz,  13
        #Rxy R2xz, R3xy R2xz, Rxy R3xz, 16
        #R2xy R3xz, R3xy R3xz,  18
        #Ryz R2xy, Ryz Rxz, Ryz R2xy Rxz, Ryz R2xz,
        #Ryz R2xy R3xz, R3yz R2xy Rxz 24
        I = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Rxy = array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
        Ryz = array([[1, 0, 0],[0, 0, 1],[0, -1, 0]])
        Rxz = array([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])

        Rxy2 = dot(Rxy, Rxy)
        Rxy3 = dot(Rxy, Rxy2)
        Ryz2 = dot(Ryz, Ryz)
        Ryz3 = dot(Ryz, Ryz2)
        Rxz2 = dot(Rxz, Rxz)
        Rxz3 = dot(Rxz, Rxz2)


        addpts = [dot(I, pts).tolist() + [weight], \
                  dot(Rxy, pts).tolist() + [weight], \
                  dot(Rxy2, pts).tolist() + [weight], \
                  dot(Rxy3, pts).tolist() + [weight], \
                  dot(Ryz, pts).tolist() + [weight], \
                  dot(Ryz2, pts).tolist() + [weight], \
                  dot(Ryz3, pts).tolist() + [weight], \
                  dot(Rxz, pts).tolist() + [weight], \
                  dot(Rxz2, pts).tolist() + [weight], \
                  dot(Rxz3, pts).tolist() + [weight], \
                  dot(dot(Rxy, Rxz), pts).tolist() + [weight], \
                  dot(dot(Rxy2, Rxz), pts).tolist() + [weight], \
                  dot(dot(Rxy3, Rxz), pts).tolist() + [weight], \
                  dot(dot(Rxy, Rxz2), pts).tolist() + [weight], \
                  dot(dot(Rxy3, Rxz2), pts).tolist() + [weight], \
                  dot(dot(Rxy, Rxz3), pts).tolist() + [weight], \
                  dot(dot(Rxy2, Rxz3), pts).tolist() + [weight], \
                  dot(dot(Rxy3, Rxz3), pts).tolist() + [weight], \
                  dot(dot(Ryz, Rxy2), pts).tolist() + [weight], \
                  dot(dot(Ryz, Rxz), pts).tolist() + [weight], \
                  dot(dot(Ryz, dot(Rxy2, Rxz)), pts).tolist() + [weight], \
                  dot(dot(Ryz, Rxz2), pts).tolist() + [weight], \
                  dot(dot(Ryz, dot(Rxy2, Rxz3)), pts).tolist() + [weight], \
                  dot(dot(Ryz3, dot(Rxy2, Rxz)), pts).tolist() + [weight]]

        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    else:
        print "Unknown gentype in parse_dim3elm1:", Gen_type
        raise RuntimeError('...')
    return addpts, n_pts

def parse_dim2elm1(Gen_type, pts, weight):
    """ This function generates the actual points based on the generators in
    Cools encyclopedia.

    It does so for 2D squares (dim = 2, elm = 1)
    """
    if Gen_type == 'Origin':
        addpts = [[pts[0], pts[1], weight]]
        n_pts = 1

    elif Gen_type == 'Fully symmetric':
        addpts = [[pts[0], pts[1], weight], \
            [-pts[0], pts[1], weight], \
            [pts[0], -pts[1], weight], \
            [-pts[0], -pts[1], weight], \
            [pts[1], pts[0], weight], \
            [-pts[1], pts[0], weight], \
            [pts[1], -pts[0], weight], \
            [-pts[1], -pts[0], weight]]
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    elif Gen_type == 'Partial symmetry':
        addpts = [[pts[0], pts[1], weight], \
            [pts[0], -pts[1], weight]]
        #Apparently sometimes you have [x, 0], in which case you only have one
        #unique point. Should be labelled differently, but aah well.
        #To account for this case, we can again use unique...
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    elif Gen_type == 'Central symmetry':
        addpts = [[pts[0], pts[1], weight], \
            [-pts[0], -pts[1], weight]]
        n_pts = 2

    elif Gen_type == 'Rectangular symmetry':
        addpts = [[pts[0], pts[1], weight], \
            [-pts[0], -pts[1], weight], \
            [-pts[0], pts[1], weight], \
            [pts[0], -pts[1], weight]]
        dum = uniquerows(array(addpts), by='rows')
        addpts = dum.tolist()
        n_pts = len(addpts)

    elif Gen_type == 'Rotational invariant':
        #( x, y ) is a point with weight w, then ( -x, -y ), ( -y, x ) and ( y, -x )
        addpts = [[pts[0], pts[1], weight], \
            [-pts[0], -pts[1], weight], \
            [-pts[1], pts[0], weight], \
            [pts[1], -pts[0], weight]]
        n_pts = 4

    else:
        print "Unknown gentype in parse_dim2elm1:", Gen_type
        raise RuntimeError('...')
    return addpts, n_pts

def hedge_convert():
    from master.cubature.hedge_dim2elm0 import TriangleCubatureData
    cube_rule = [[[-y, x, w] for x, y, w in deg_rule] \
        for deg_rule in TriangleCubatureData]

    return cube_rule

def convert(dim, elm):
    """ This function parses the Cools tables (copied and pasted from the web
    into text files.


    Cools, R. "Monomial Cubature Rules Since "Stroud": A Compilation--Part 2."
    J. Comput. Appl. Math. 112, 21-27, 1999.

    Cools, R. "Encyclopaedia of Cubature Formulas."
    http://www.cs.kuleuven.ac.be/~nines/research/ecf/ecf.html

    @param dim Dimension to parse cubature rules
    @param elm Type of element for which cubature rules will be parsed

    @note: Legal entries are, (dim, elm) = (2, 1), (3, 1), (3, 0)
    For (dim, elm) = (2, 0) and (3, 2), we borrow the already generated tables
    from the Hedge project, and we have to create our own points for prisms.
    """
    if (dim == 2) and (elm == 0):
        cube_rule = hedge_convert()
    else:

        cube_rule = []
        with open('master/cubature/cools_dim%delm%d.txt' % (dim, elm),'r') as f:
            i = 0
            degree = 0 #Keep track of the current degree
            cur_pts = 0 #keep track of the current number of points parse
                        #for the current polynomial degree
            n_points = 0
            line = ' '
            while line != '':
                line = f.readline()
                i = i + 1 #keep track of line number
                words = line.split(" ")

                if (line[0:2] == "##") or (line[0:2] == "\n"):
                    pass # Do nothing

                elif any([word == "Dimension:" for word in words]):
                    #Just a check
                    if int(line[line.index('\n')-1]) is not dim:
                        print "Dimension should be 2, but read as:", \
                            int(line[line.index('\n')-1]),  "on line", i, \
                            "of file for degree:", degree + 1
                    ##Because this is in the header, we can
                    ##error check the PREVIOUS ruls
                    if n_points != cur_pts:
                        print "The number of expected points were not", \
                            "equal to the actual points for degree", degree, \
                            ". Expected", n_points, "but parsed", cur_pts

                        raise RuntimeError('...')

                    ##also, because this is in the header, increment the
                    #degree of the current polynomial
                    degree = degree + 1



                elif any([word == "Degree:" for word in words]):
                    #Just a check
                    if int(line[line.index('\n')-2:-1]) < degree:
                        print "Expected degree should be", degree, \
                            "but read as:", int(line[line.index('\n')-1]), \
                            "on line", i, "of file for degree:", degree

                elif any([word == "Points:" for word in words]):
                    #Not super robust
                    n_points = int(words[-1])
                    #also reset the current points parsed
                    cur_pts = 0

                    ##Since this is the last part (nominally) of the header, we
                    #do a little bit of initialization here
                    cube_rule.append([])

                elif any([word == "Generator:" for word in words]):
                    Gen_type = " ".join(\
                        words[words.index('[') + 1 : words.index(']\n')])
                    dum = f.readline()
                    i = i + 1
                    while len(dum.split(")")) < 2:
                        dum = dum + f.readline()
                        i = i + 1

                    pts = eval(dum.replace('x','*').replace('^','**'))
                    dum = f.readline()
                    i = i + 1
                    if dum != 'Corresponding weight:\n':
                        print "Expected line to be 'Corresponding weight:'", \
                    	    "but it is not. Line ", i, "."
                    dum = f.readline()
                    i = i + 1
                    #Scientific notation needs to be translated to
                    #python-speak, and trailing comma needs to be removed
                    dum = dum.replace('x','*').replace('^','**').replace(',','')
                    weight = eval(dum)

                    #Parse or 'generate' these points
                    if (dim == 2) and (elm == 1):
                        addpts, n_pts = parse_dim2elm1(Gen_type, pts, weight)
                    elif (dim == 3) and (elm == 0):
                        addpts, n_pts = parse_dim3elm0(Gen_type, pts, weight)
                    elif (dim == 3) and (elm == 1):
                        addpts, n_pts = parse_dim3elm1(Gen_type, pts, weight)
                    #keep track of the points generated for error-checking later
                    cur_pts = cur_pts + n_pts

                    #Finally, add the new points to the cubature rule for the
                    #current degree
                    for rows in addpts:
                        cube_rule[degree - 1].append(rows)

                else:
                    print "Parsed upto line", i

        print f.closed
    return cube_rule

from master.mk_cubature import gauss_quad as gauss_quad
def mk_tables():
    """
    This function generates the file cubature_tables.py. It basically tabulates
    the cubature rules for various element types.

    The file that is created will be called by src.master.get_pts_weights

    @see src.master.get_pts_weights

    @author Matt Ueckermann
    """

    f = open('master/cubature/cubature_tables.py','w')


    DIM = [2, 2, 3, 3]
    ELM = [0, 1, 0, 1]
    ELMTYPE = ['triangles', 'squares', 'tetrahedrals', 'cubes']
    for dim, elm, elmtype in zip(DIM, ELM, ELMTYPE):
        #Just for debugging
        #cube_rule = master.cubature.cools_convert.convert(dim, elm)
        cube_rule = convert(dim, elm)

        f.write('#%dD %s (dim = %d, elm = %d)\n'% (dim, elmtype, dim, elm))
        f.write('dim%d_elm%d = [ #START \n' % (dim, elm))

        for i in xrange(len(cube_rule)):
            f.write('    [  #start %d \n' % (i + 1))
            for rule in cube_rule[i]:
                f.write('        %s, \\\n' % str(rule))
            f.write('    ], #end %d \n' % (i + 1))
        f.write('] #END\n\n')

    #Now, the only tricky one that requires additional work is the prisms!
    #First, get the rule for 2D triangles:
    cube_rule = convert(2, 0)
    f.write('#%dD %s (dim = %d, elm = %d)\n'% (3, 'prism', 3, 2))
    f.write('dim%d_elm%d = [ #START \n' % (3, 2))

    for i in xrange(len(cube_rule)):
        f.write('    [  #start %d \n' % (i + 1))

        #Now we need to calculate the tensor product for this particular rule
        #First calculate the 1D gauss quadrature for the current order
        (x, w) = gauss_quad(i + 1)
        for rule in cube_rule[i]:
            #Now do the tensor product
            for pt, wght in zip(x, w):
                f.write('        [%s, %s, %s, %s], \\\n' % \
                    (str(rule[0]), str(rule[1]), str(pt), str(wght * rule[2])))
        f.write('    ], #end %d \n' % (i + 1))
    f.write('] #END\n\n')

    #Close the file, and make sure it's closed.
    f.close()
    if not f.closed:
        print "File %s did not close properly!" % f.name

