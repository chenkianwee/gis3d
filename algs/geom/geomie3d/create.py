# ==================================================================================================
#
#    Copyright (c) 2020, Chen Kian Wee (chenkianwee@gmail.com)
#
#    This file is part of geomie3d
#
#    geomie3d is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geomie3d is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py4design.  If not, see <http://www.gnu.org/licenses/>.
#
# ==================================================================================================
import numpy as np

from . import get
from . import modify
from . import calculate
from . import utility
from . import topobj
from . import geom
from . import d4pispace

from .geomdl import BSpline, utilities, construct

def box(dimx, dimy, dimz, centre_pt = [0,0,0], attributes = {}):
    """
    Constructs a box which is a solid topology where its bottom face midpt is at the origin (0,0,0).
 
    Parameters
    ----------
    dimx : float
        length of box in the x-axis.
    
    dimy : float
        length of box in the y-axis.
        
    dimz : float
        height of box.
        
    centre_pt : tuple, optional
        tuple with the xyz coordinates of the centre point of the box.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    box : solid topology
        A box of solid topology
    """
    dimx_half = dimx/2
    dimy_half = dimy/2
    dimz_half = dimz/2
    
    mnx = centre_pt[0] - dimx_half
    mny = centre_pt[1] - dimy_half
    mnz = centre_pt[2] - dimz_half
    
    mxx = centre_pt[0] + dimx_half
    mxy = centre_pt[1] + dimy_half
    mxz = centre_pt[2] + dimz_half
    
    #bottom face
    xyz_list1 = np.array([[mnx, mny, mnz],
                          [mnx, mxy, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mny, mnz]])
    
    vlist1 = vertex_list(xyz_list1)
    face1 = polygon_face_frm_verts(vlist1)
    
    #left vertical face
    xyz_list2 = np.array([[mnx, mxy, mnz],
                          [mnx, mny, mnz],
                          [mnx, mny, mxz],
                          [mnx, mxy, mxz]])
    
    vlist2 = vertex_list(xyz_list2)
    face2 = polygon_face_frm_verts(vlist2)
    
    #south vertical face
    xyz_list3 = np.array([[mnx, mny, mnz],
                          [mxx, mny, mnz],
                          [mxx, mny, mxz],
                          [mnx, mny, mxz]])
    
    vlist3 = vertex_list(xyz_list3)
    face3 = polygon_face_frm_verts(vlist3)
    
    #right vertical face
    xyz_list4 = np.array([[mxx, mny, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mxy, mxz],
                          [mxx, mny, mxz]])
    
    vlist4 = vertex_list(xyz_list4)
    face4 = polygon_face_frm_verts(vlist4)
    
    #north vertical face
    xyz_list5 = np.array([[mxx, mxy, mnz],
                          [mnx, mxy, mnz],
                          [mnx, mxy, mxz],
                          [mxx, mxy, mxz]])
    
    vlist5 = vertex_list(xyz_list5)
    face5 = polygon_face_frm_verts(vlist5)
    
    #top cap face
    xyz_list6 = np.array([[mnx, mny, mxz],
                          [mxx, mny, mxz],
                          [mxx, mxy, mxz],
                          [mnx, mxy, mxz]])
    
    vlist6 = vertex_list(xyz_list6)
    face6 = polygon_face_frm_verts(vlist6)
    shell = topobj.Shell(np.array([face1, face2, face3, face4, face5, face6]))
    solid = topobj.Solid(shell, attributes = attributes)
    return solid

def boxes_frm_bboxes(bbox_ls: list[utility.Bbox]) -> list[topobj.Topology]:
    """
    Constructs boxes from bounding boxes.
 
    Parameters
    ----------
    bbox_ls : list[utility.Bbox]
        list of bboxes to be converted to boxes.
        
    Returns
    -------
    boxes : list[topobj.Topology]
        A list of boxes of solid topology. If the bbox has zdim==0, a polygon face is returned.
    """
    bx_ls = []
    for bbox in bbox_ls:
        midxyz = calculate.bboxes_centre([bbox])[0]
        dimx = bbox.maxx - bbox.minx
        dimy = bbox.maxy - bbox.miny
        dimz = bbox.maxz - bbox.minz
        if dimz != 0:
            topo = box(dimx, dimy, dimz, centre_pt = midxyz)
        else:
            topo = polygon_face_frm_midpt(midxyz, dimx, dimy)
        bx_ls.append(topo)
    return bx_ls

def grid3d_from_bbox(bbox: utility.Bbox, nx: int, ny: int, nz: int) -> list[utility.Bbox]:
    """
    divide a bbox into multiple smaller bboxes.
 
    Parameters
    ----------
    bbox : utility.Bbox
        the bbox to divide.
    
    nx : int
        number of divisions in the x-direction

    ny : int
        number of divisions in the y-direction

    nz : int
        number of divisions in the z-direction. If nz == 0, it will produce bboxes with z-coordinate == minz.
        
    Returns
    -------
    divided_bboxes : list[utility.Bbox]
        A list of bboxes from the division
    """
    bbox_arr = bbox.bbox_arr
    xdiv = np.linspace(bbox_arr[0], bbox_arr[3], num = nx+1)
    ydiv = np.linspace(bbox_arr[1], bbox_arr[4], num = ny+1)

    xdims = xdiv[1] - xdiv[0]
    ydims = ydiv[1] - ydiv[0]

    xdiv = xdiv[:-1]
    ydiv = ydiv[:-1]
    if nz != 0:
        zdiv = np.linspace(bbox_arr[2], bbox_arr[5], num = nz+1)
        zdims = zdiv[1] - zdiv[0]
        zdiv = zdiv[:-1]
        # mesh the points
        yy, zz, xx = np.meshgrid(ydiv,zdiv,xdiv)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

    else:
        zdims = 0
        # mesh the points
        xx, yy = np.meshgrid(xdiv,ydiv)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = np.array([bbox_arr[2]])
        zz = np.repeat(zz, len(xx))

    xyzs = np.vstack([xx, yy, zz])
    xyzs = xyzs.T
    nxyzs = len(xyzs)

    xdims = np.repeat(xdims, nxyzs)
    ydims = np.repeat(ydims, nxyzs)
    zdims = np.repeat(zdims, nxyzs)

    bboxes = bboxes_frm_lwr_left_pts(xyzs, xdims, ydims, zdims)
    return bboxes

def xyzs_frm_bboxes(bbox_ls: list[utility.Bbox]) -> np.ndarray:
    """
    Constructs xyzs from bounding boxes.
 
    Parameters
    ----------
    bbox_ls : list[utility.Bbox]
        list of bboxes to be converted to xyzs.
        
    Returns
    -------
    xyzs : np.ndarray
        np.ndarray(shape(4, 3)). Bounding box is defined by 4 points.
    """
    bbox_arr_ls = np.array([bbox.bbox_arr for bbox in bbox_ls])
    
    xyzs1 = np.take(bbox_arr_ls, [0,1,2], axis=1)
    xyzs2 = np.take(bbox_arr_ls, [3,1,2], axis=1)
    xyzs3 = np.take(bbox_arr_ls, [3,4,2], axis=1)
    xyzs4 = np.take(bbox_arr_ls, [0,4,2], axis=1)

    xyzs5 = np.take(bbox_arr_ls, [0,1,5], axis=1)
    xyzs6 = np.take(bbox_arr_ls, [3,1,5], axis=1)
    xyzs7 = np.take(bbox_arr_ls, [3,4,5], axis=1)
    xyzs8 = np.take(bbox_arr_ls, [0,4,5], axis=1)

    xyzs = np.hstack([xyzs1, xyzs2, xyzs3, xyzs4, xyzs5, xyzs6, xyzs7, xyzs8])
    xyzs_shape = xyzs.shape
    xyzs = np.reshape(xyzs, (xyzs_shape[0], int(xyzs_shape[1]/3), 3))
    return xyzs

def extrude_polygon_face(face, direction, magnitude, attributes = {}):
    """
    Extrude a face according to the direction given.
 
    Parameters
    ----------
    face : face topology
        The polygon face to extrude
    
    direction : tuple
        a 3 dimension tuple defining the direction in [x,y,z].
    
    magnitude : float
        the magnitude of the extrusion.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    box : solid topology
        Solid generated from the extrusion
    """
    def vertical_faces(vs1, vs2):
        vert_faces = []
        nverts = len(vs1)
        for cnt in range(nverts):
            if cnt != nverts - 1:
                v1 = vs1[cnt]
                v2 = vs1[cnt+1]
                v3 = vs2[cnt+1]
                v4 = vs2[cnt]
            else:
                v1 = vs1[cnt]
                v2 = vs1[0]
                v3 = vs2[0]
                v4 = vs2[cnt]
            
            vert_f = polygon_face_frm_verts(np.array([v1,v2,v3,v4]))
            # utility.viz([{'topo_list': [vert_f], 'colour': 'red'}])
            vert_faces.append(vert_f)
        return vert_faces
    
    if face.surface_type == geom.SrfType.POLYGON:
        # ext_faces = []
        #get the centre point of the polygon face
        midxyz = calculate.face_midxyz(face)
        dest_xyz = calculate.move_xyzs([midxyz], [direction], [magnitude])[0]
        mv_face = modify.move_topo(face, dest_xyz, ref_xyz = midxyz)
        vs1 = get.bdry_vertices_frm_face(face)
        vs2 = get.bdry_vertices_frm_face(mv_face)
        ext_faces = vertical_faces(vs1, vs2)
        
        if len(face.hole_wire_list) != 0:
            vs3_2dls = get.hole_vertices_frm_face(face)
            vs4_2dls = get.hole_vertices_frm_face(mv_face)
            nholes = len(vs3_2dls)
            for cnt in range(nholes):
                ext_faces2 = vertical_faces(vs3_2dls[cnt], vs4_2dls[cnt])
                ext_faces.extend(ext_faces2)
        
        #check for the face normal if it is the same as the extrusion direction
        nrml = face.normal
        angle = calculate.angle_btw_2vectors(nrml, direction)

        if angle <= 90: #need to reverse the base face as it is facing its own extrusion
            face = modify.reverse_face_normal(face)
            
        elif angle > 90:
            mv_face = modify.reverse_face_normal(mv_face)
        
        ext_faces.insert(0,face)
        ext_faces.append(mv_face)
        # cmp = composite(ext_faces)
        # edges = get.edges_frm_composite(cmp)
        # from . import viz
        # for f in ext_faces:
        #     viz.viz([{'topo_list':[f], 'colour': 'blue'},
        #              {'topo_list':edges, 'colour': 'white'}])
        shell = topobj.Shell(np.array(ext_faces))
        solid = topobj.Solid(shell, attributes = attributes)
        return solid
    
    else:
        print('ERROR SURFACE IS NOT POLYGON')
    
def polygon_face_frm_midpt(centre_pt: list[float], dimx: float, dimy: float, attributes: dict = {}) -> topobj.Face:
    """
    Constructs a face from the midpt.
 
    Parameters
    ----------
    centre_pt : list[float]
        list with the xyz coordinates of the centre point of the box. A numpy array will work too.
        
    dimx : float
        length of face in the x-axis.
    
    dimy : float
        length of face in the y-axis.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    face : topobj.Face
        A face of face topology
    """
    dimx_half = dimx/2
    dimy_half = dimy/2
    
    mnx = centre_pt[0] - dimx_half
    mny = centre_pt[1] - dimy_half
    
    mxx = centre_pt[0] + dimx_half
    mxy = centre_pt[1] + dimy_half
    
    #bottom face
    xyz_list1 = np.array([[mnx, mny, centre_pt[2]],
                          [mnx, mxy, centre_pt[2]],
                          [mxx, mxy, centre_pt[2]],
                          [mxx, mny, centre_pt[2]]])
    
    xyz_list1 = np.array([[mxx, mny, centre_pt[2]],
                          [mxx, mxy, centre_pt[2]],
                          [mnx, mxy, centre_pt[2]],
                          [mnx, mny, centre_pt[2]]
                          ])
    
    vlist1 = vertex_list(xyz_list1)
    face1 = polygon_face_frm_verts(vlist1)
    
    return face1

def bbox_frm_arr(bbox_arr: np.ndarray, attributes: dict = {}) -> utility.Bbox:
    """
    Create a bounding box object
    
    Parameters
    ----------
    bbox_arr : np.ndarray
        Array specifying [minx,miny,minz,maxx,maxy,maxz].
        
    attributes : dict, optional
        dictionary of the attributes.
        
    Returns
    -------
    bbox : utility.Bbox
        A bbox object
    """
    bbox = utility.Bbox(bbox_arr, attributes = attributes)
    return bbox

def bboxes_frm_midpts(midpts: np.ndarray, xdims: np.ndarray, ydims: np.ndarray, zdims: np.ndarray, attributes_list: list[dict] = {}) -> list[utility.Bbox]:
    """
    Create bboxes based on midpts and the x,y,z dimensions.
    
    Parameters
    ----------
    midpts : np.ndarray
        np.ndarray(shape(number of midpts, 3)) xyzs specifying the xyz position.
        
    xdims : np.ndarray
        np.ndarray(shape(number of midpts)) the x dimension of the bounding boxes
    
    ydims : np.ndarray
        np.ndarray(shape(number of midpts)) the y dimension of the bounding boxes
    
    zdims : np.ndarray
        np.ndarray(shape(number of midpts)) the z dimension of the bounding boxes
    
    attributes_list : list[dict], optional
        list of attributes dictionary.
        
    Returns
    -------
    bboxes : list[utility.Bbox]
        A list of bboxes.
    """
    if type(midpts) != np.ndarray:
        midpts = np.array(midpts)
    
    if type(xdims) != np.ndarray:
        xdims = np.array(xdims)
    
    if type(ydims) != np.ndarray:
        ydims = np.array(ydims)

    if type(zdims) != np.ndarray:
        zdims = np.array(zdims)

    midptsT = midpts.T
    xs = midptsT[0]
    ys = midptsT[1]
    zs = midptsT[2]
    
    mnxs = xs - (xdims/2)
    mnys = ys - (ydims/2)
    mnzs = zs - (zdims/2)

    mxxs = xs + (xdims/2)
    mxys = ys + (ydims/2)
    mxzs = zs + (zdims/2)

    bbox_arrs = np.vstack([mnxs, mnys, mnzs, mxxs, mxys, mxzs]).T
    bboxes = [utility.Bbox(bbox_arr) for bbox_arr in bbox_arrs]
    return bboxes

def bboxes_frm_lwr_left_pts(lwr_left_pts: np.ndarray, xdims: np.ndarray, ydims: np.ndarray, zdims: np.ndarray, attributes_list: list[dict] = {}) -> list[utility.Bbox]:
    """
    Create bboxes based on lwr_left_pts and the x,y,z dimensions.
    
    Parameters
    ----------
    lwr_left_pts : np.ndarray
        np.ndarray(shape(number of lwr_left_pts, 3)) xyzs specifying the xyz position.
        
    xdims : np.ndarray
        np.ndarray(shape(number of lwr_left_pts)) the x dimension of the bounding boxes
    
    ydims : np.ndarray
        np.ndarray(shape(number of lwr_left_pts)) the y dimension of the bounding boxes
    
    zdims : np.ndarray
        np.ndarray(shape(number of lwr_left_pts)) the z dimension of the bounding boxes
    
    attributes_list : list[dict], optional
        list of attributes dictionary.
        
    Returns
    -------
    bboxes : list[utility.Bbox]
        A list of bboxes.
    """
    if type(lwr_left_pts) != np.ndarray:
        lwr_left_pts = np.array(lwr_left_pts)
    
    if type(xdims) != np.ndarray:
        xdims = np.array(xdims)
    
    if type(ydims) != np.ndarray:
        ydims = np.array(ydims)

    if type(zdims) != np.ndarray:
        zdims = np.array(zdims)


    lwr_left_ptsT = lwr_left_pts.T
    xs = lwr_left_ptsT[0]
    ys = lwr_left_ptsT[1]
    zs = lwr_left_ptsT[2]

    mxxs = xs + (xdims)
    mxys = ys + (ydims)
    mxzs = zs + (zdims)

    bbox_arrs = np.vstack([xs, ys, zs, mxxs, mxys, mxzs]).T
    bboxes = [utility.Bbox(bbox_arr) for bbox_arr in bbox_arrs]
    return bboxes

def ray(xyz_orig, xyz_dir, attributes = {}):
    """
    This function constructs a ray object.
 
    Parameters
    ----------
    xyz_orig : tuple
        tuple with the xyz coordinates of the ray origin.
        
    xyz_dir : tuple
        tuple with the xyz of the ray direction.
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    ray : ray object
        A ray object
    """
    ray = utility.Ray(xyz_orig, xyz_dir, attributes = attributes)
    return ray

def rays_d4pi_frm_verts(verts: list[topobj.Vertex], ndirs: int = 360, vert_id: bool = True) -> list[utility.Ray]:
    """
    This function constructs a 4 pi space direction of rays for each of the given vert.
 
    Parameters
    ----------
    verts : list[topobj.Vertex]
        list or verts. Each vert will be use to generate a 4 pi space direction of rays.
        
    ndirs : int, optional
        number of direction to generate for the 4 pi space.
    
    vert_id : bool, optional
        store an attribute 'vert_id' that identifies the rays generated from each vert.
 
    Returns
    -------
    rays : list[utility.Ray]
        list of all the rays generated.
    """
    unitball = d4pispace.tgDirs(ndirs)
    #create the rays for each analyse pts
    rays = []
    if vert_id == True:
        for cnt,v in enumerate(verts):
            for dix in unitball.getDirList():
                a_ray = ray(v.point.xyz, [dix.x, dix.y, dix.z])     
                rays.append(a_ray)
    else:
        for v in verts:
            for dix in unitball.getDirList():
                a_ray = ray(v.point.xyz, [dix.x, dix.y, dix.z])        
                rays.append(a_ray)
    return rays

def vertex(xyz, attributes = {}):
    """
    This function constructs a vertex topology.
 
    Parameters
    ----------
    xyz : tuple
        tuple with the xyz coordinates.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    vertex : vertex topology
        A vertex topology containing a point geometry
    """
    point = geom.Point(xyz)
    vertex = topobj.Vertex(point, attributes = attributes)
    return vertex
    
def vertex_list(xyz_list: np.ndarray, attributes_list: list[dict] = []) -> list[topobj.Vertex]:
    """
    This function constructs a list of vertex topology.
 
    Parameters
    ----------
    xyz_list : np.ndarray
        np.ndarray(shape(number of vertices, 3)).
        
    attributes_list : list[dict], optional
        Dictionary of the attributes
 
    Returns
    -------
    vertex_list : list[topobj.Vertex]
        A list of vertex topology.
    """
    nxyz = len(xyz_list)
    natts = len(attributes_list)
    
    #check if the list match
    is_atts = False
    if natts != 0:
        is_atts = True
        if natts != nxyz:
            raise NameError("Number of xyz_list and attributes_list do not match")
    
    if is_atts:
        vlist = np.array([vertex(xyz_list[cnt], attributes = attributes_list[cnt]) 
                          for cnt in range(nxyz)])
    else:
        vlist = np.array([vertex(xyz_list[cnt]) for cnt in range(nxyz)])
    
    return vlist

def vertex_list_frm_bbox(bbox: utility.Bbox) -> list[topobj.Vertex]:
    """
    This function constructs a list of vertex from bbox. The vertex_list will contain 8 vertices.
 
    Parameters
    ----------
    bbox : utility.Bbox
        utility.Bbox to be used for creating the vertex list.
 
    Returns
    -------
    vertex_list : list[topobj.Vertex]
        A list of vertices defining the bbox.
    """
    mnx = bbox.minx
    mny = bbox.miny
    mnz = bbox.minz
    mxx = bbox.maxx
    mxy = bbox.maxy
    mxz = bbox.maxz

    vert1 = vertex([mnx, mny, mnz])
    vert2 = vertex([mnx, mxy, mnz])
    vert3 = vertex([mxx, mxy, mnz])
    vert4 = vertex([mxx, mny, mnz])

    vert5 = vertex([mnx, mny, mxz])
    vert6 = vertex([mnx, mxy, mxz])
    vert7 = vertex([mxx, mxy, mxz])
    vert8 = vertex([mxx, mny, mxz])
    return [vert1, vert2, vert3, vert4, vert5, vert6, vert7, vert8]

def polygon_face_frm_verts(vertex_list, hole_vertex_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology. 
        
    hole_vertex_list : a 2d list of vertices
        A 2d list of vertex topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    bdry_wire = wire_frm_vertices(vertex_list)
    
    hole_wire_list = []
    for hole in hole_vertex_list:
        hole_wire =  wire_frm_vertices(hole)
        hole_wire_list.append(hole_wire)
    face = polygon_face_frm_wires(bdry_wire, hole_wire_list = hole_wire_list, attributes = attributes)
    return face

def polygon_face_frm_wires(bdry_wire, hole_wire_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    bdry_wire : a wire
        A wire topology that is the boundary of the face
        
    hole_wire_list : a list of wires
        A list of wire topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    face = topobj.Face(attributes = attributes)
    face.add_polygon_surface(bdry_wire, hole_wire_list = hole_wire_list)
    return face

def bspline_face_frm_ctrlpts(ctrlpts_xyz, knotvector_u, knotvector_v, deg_u, deg_v,
                             resolution = 0.06, attributes = {}):
    """
    This function creates a bspline face with control points. For a surface on the xy-plane, arrange the points from top-down-left right for a +Z normal. Refer to for more help https://github.com/orbingol/geomdl-examples/blob/master/surface/ex_surface01.py
 
    Parameters
    ----------
    ctrlpts_xyz : array of xyzs
        control points of the bspline surface. 
    
    knotvector_u : int
        knockvector_u
    
    knotvector_v : int
        knockvector_v
    
    deg_u : int
        the degree of the surface in u direction.
        
    deg_v : int
        the degree of the surface in v direction.
    
    resolution : float
        a number between 0-1. The smaller the number the higher resolution of the surface stored.
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Returns
    -------
    bspline_face : face topology
        A face topology containing a bspline surface geometry
    """
    if type(ctrlpts_xyz) == np.ndarray:
        ctrlpts_xyz = ctrlpts_xyz.tolist()
        
    # Create a BSpline surface instance
    surf = BSpline.Surface()
    # Set degrees
    surf.degree_u = deg_u
    surf.degree_v = deg_v
    kv_u = knotvector_u
    kv_v = knotvector_v
    surf.set_ctrlpts(ctrlpts_xyz, kv_u, kv_v)
    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, kv_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, kv_v)

    # Set evaluation delta
    surf.delta = resolution
    # Evaluate surface
    surf.evaluate()
    face = topobj.Face(attributes = attributes)
    face.add_bspline_surface(surf)
    return face

def bspline_face_frm_loft(bspline_edge_list,deg=1,resolution=0.06,attributes = {}):
    """
    This function creates a bspline face from bspline edges.
    
    Parameters
    ----------
    bspline_edge_list : array of Edge with bspline geometry
        array of edges with bspline geometry. 
        
    deg : int, optional
        the degree of the surface.
    
    resolution : float, optional
        a number between 0-1. The smaller the number the higher resolution of the surface stored.
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Returns
    -------
    bspline_face : face topology
        A face topology containing a bspline surface geometry
    """
    crv_ls = [e.curve for e in bspline_edge_list]
    surf = construct.construct_surface('u', *crv_ls, degree = deg)
    # Set evaluation delta
    surf.delta = resolution
    # Evaluate surface
    surf.evaluate()
    face = topobj.Face(attributes = attributes)
    face.add_bspline_surface(surf)
    return face

def grids_frm_bspline_face(face, columns, rows, polygonise = True):
    """
    This function creates a grid from a face.
 
    Parameters
    ----------
    face : face object
        face to be gridded.
        
    columns : int
        number of columns.
        
    rows : int
        number of rows.
        
    Returns
    -------
    grid : array of faces
        the gridded face, if grid_pts=True, will return empty array.
        
    """
    flist = []
    pts = grid_pts_frm_bspline_face(face,columns+1,rows+1,xyzs=True)
    ngrids = columns*rows
    if polygonise == False:
        for cnt in range(ngrids):
            id1 = cnt + int(cnt/columns)
            id2 = id1 + 1
            id3 = id1 + columns+2
            id4 = id3 - 1
            f_xyz = [pts[id4], pts[id3], pts[id1], pts[id2]]
            f = bspline_face_frm_ctrlpts(f_xyz, 2, 2, 1, 1, resolution = 0.5)
            flist.append(f)
    else:
        for cnt in range(ngrids):
            id1 = cnt + int(cnt/columns)
            id2 = id1 + 1
            id3 = id1 + columns+2
            id4 = id3 - 1
            # create polygon surface
            f_xyz = [pts[id1], pts[id2], pts[id3], pts[id4]]
            f_v = vertex_list(f_xyz)
            f = polygon_face_frm_verts(f_v)
            flist.append(f)
    return flist
    
def grid_pts_frm_bspline_face(face,columns,rows,xyzs=False):
    """
    This function creates a grid pts from a face.
 
    Parameters
    ----------
    face : face object
        face to be gridded.
        
    columns : int
        number of columns.
        
    rows : int
        number of rows.
    
    xyzs : bool, optional
        If True return the xyzs, False return vertices.
    
    Returns
    -------    
    grid_pts : array of vertices
        grid of pts. If xyzs == True, return xyzs.
    """
    if face.surface_type == geom.SrfType.BSPLINE:
        urange = [0,1,columns]
        vrange = [0,1,rows]
        params = utility.gen_gridxyz(urange, vrange)
        surface = face.surface
        pts = surface.evaluate_list(params)
        if xyzs == True:
            return pts
        else:    
            vs = vertex_list(pts)
            return vs
    else:
        print('ERROR SURFACE IS NOT BSPLINE')

def pline_edge_frm_verts(vertex_list: list[topobj.Vertex], attributes: dict = {}) -> topobj.Edge:
    """
    This function constructs a polyline edge from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    pline_edge : edge topology
        A edge topology containing a polyline geometry
    """
    edge = topobj.Edge(attributes = attributes)
    edge.add_polyline_curve(vertex_list)
    return edge

def pline_edges_frm_rays(rays: list[utility.Ray], attributes_list: list[dict] = []) -> list[topobj.Edge]:
    """
    This function constructs a polyline edges from a list of rays.
 
    Parameters
    ----------
    rays : list[utility.Ray]
        A list of rays that will be converted into edges
    
    attributes_list : list[dict], optional
        list of dictionary of the attributes.
 
    Returns
    -------
    pline_edges : list[topobj.Edge]
        list[topoobj.Edge] created from the rays.
    """
    pline_edges = []
    if len(attributes_list) != 0:
        for cnt,r in enumerate(rays):
            o = r.origin
            dirx = r.dirx
            d = calculate.move_xyzs([o], [dirx], [1])[0]
            vs = vertex_list([o, d])
            att = attributes_list[cnt]
            e = pline_edge_frm_verts(vs, attributes=att)
            pline_edges.append(e)
    else:
        for cnt,r in enumerate(rays):
            o = r.origin
            dirx = r.dirx
            d = calculate.move_xyzs([o], [dirx], [1])[0]
            vs = vertex_list([o, d])
            e = pline_edge_frm_verts(vs)
            pline_edges.append(e)
    return pline_edges

def pline_edges_frm_face_normals(faces: list[topobj.Face], magnitude: float = 1.0 , attributes_list: list[dict] = []) -> list[topobj.Edge]:
    """
    This function constructs a polyline edges from a list of list of normals extracted from the face.
 
    Parameters
    ----------
    faces : list[topobj.Face]
        A list of faces whoose normals that will be converted into edges
    
    attributes_list : list[dict], optional
        list of dictionary of the attributes.
 
    Returns
    -------
    pline_edges : list[topobj.Edge]
        list[topoobj.Edge] created from the rays.
    """
    nfaces = len(faces)
    midxyzs = []
    nrmls = []
    for f in faces:
        nrml = get.face_normal(f)
        midxyz = calculate.face_midxyz(f)
        midxyzs.append(midxyz)
        nrmls.append(nrml)

    magnitudes = np.repeat(magnitude, nfaces)
    moved_xyzs = calculate.move_xyzs(midxyzs, nrmls, magnitudes)

    mid_vs = vertex_list(midxyzs)
    moved_vs = vertex_list(moved_xyzs)

    natt = len(attributes_list)
    pline_edges = []
    for cnt in range(nfaces):
        v1 = mid_vs[cnt]
        v2 = moved_vs[cnt]
        if natt == 0:
            edge = pline_edge_frm_verts([v1, v2])
        else:
            edge = pline_edge_frm_verts([v1, v2], attributes_list[cnt])
        pline_edges.append(edge)
    
    return pline_edges

def bspline_edge_frm_xyzs(ctrlpts_xyz, degree = 2, resolution = 0.01, attributes = {}):
    """
    This function constructs a bspline edge from a list of control points.
 
    Parameters
    ----------
    ctrlpts_xyz : array of xyzs
        An array of xyzs. [[x1,y1,z1], [x2,y2,z2], [...]]
    
    degree : int, optional
        degree of the bspline curve. For straight lines use 2. For more curvy go for 3 and above
    
    resolution : float, optional 
        the resolution to store the curve. Float between 0-1. 0.1 will discretize the curve into 10 parts. 0.01 will discretize it to 100 parts. The lower the number the higher resolution of the curve stored in the edge topology. 
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    bspline_edge : edge topology
        A edge topology containing a bspline geometry
    """
    if type(ctrlpts_xyz) == np.ndarray:
        ctrlpts_xyz = ctrlpts_xyz.tolist()
        
    crv = BSpline.Curve()
    # Set degree
    crv.degree = degree
    
    # Set control points
    crv.ctrlpts = ctrlpts_xyz
    # Set knot vector
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    crv.delta = resolution
    
    edge = topobj.Edge(attributes = attributes)
    edge.add_bspline_curve(crv)
    return edge

def wire_frm_vertices(vertex_list, attributes = {}):
    """
    Constructs a wire from a list of vertices. Wire created from this will have edges containing lines (polylines with only 2 vertices, a straight line).
 
    Parameters
    ----------
    vertex_list : a list of vertices
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    edge_list = []
    n_v = len(vertex_list)
    for cnt,v in enumerate(vertex_list):
        if cnt != n_v-1:
            edge = pline_edge_frm_verts([v, vertex_list[cnt+1]])
        else:
            edge = pline_edge_frm_verts([v, vertex_list[0]])
        edge_list.append(edge)
    
    wire = wire_frm_edges(edge_list, attributes = attributes)
    
    return wire

def wire_frm_edges(edge_list: list[topobj.Edge], attributes: dict = {}):
    """
    This function constructs a wire from a list of edges.
 
    Parameters
    ----------
    edge_list : a list of edge
        A list of edge topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    wire = topobj.Wire(edge_list, attributes = attributes)
    return wire
    
def coordinate_system_frm_arrs(origin, x_dir, y_dir):
    """
    This function creates a coordinate system object.
 
    Parameters
    ----------
    origin : tuple
        xyz coordinate that defines the origin.
    
    x_dir : tuple
        The xyz of a vector defining the x-axis
        
    y_dir : tuple
        The xyz of a vector defining the y-axis  
 
    Returns
    -------
    cs : the coordinate system object
        The coordinate system object.
    """
    return utility.CoordinateSystem(origin, x_dir, y_dir)

def composite(topo_list: list[topobj.Topology], attributes: dict = {}) -> topobj.Composite:
    """
    This function add attributes to the list of topology.
 
    Parameters
    ----------
    topo_list : list[topobj.Topology]
        List of Topo objects include Vertex, Edge, Wire, Face, Shell, Solid and Composite Topology Object.
        
    attributes : dict
        Dictionary of the attributes
 
    Returns
    -------
    composite : topobj.Composite
        A composite topology containing the topo list
    """
    return topobj.Composite(topo_list, attributes = attributes)

def shell_frm_faces(face_list, attributes={}):
    """
    This function constructs a shell from a list of faces.
 
    Parameters
    ----------
    face_list : a list of faces
        A list of face topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    shell : shell topology
        A shell topology 
    """
    shell = topobj.Shell(np.array(face_list), attributes = attributes)
    return shell

def solid_frm_shell(shell, attributes = {}):
    """
    This function constructs a solid from a shell.
 
    Parameters
    ----------
    shell : shell
        Shell topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    solid : solid topology
        A solid topology 
    """
    solid = topobj.Solid(shell, attributes=attributes)
    return solid

def shell_frm_delaunay(vertex_list, tolerance = 1e-6):
    """
    This function creates a TIN from a vertex list.
 
    Parameters
    ----------
    vertex_list : face object
        the x and y dim of the vertex has to be on the same plane.
 
    Returns
    -------
    shell : shell topology
        A shell object.
    """
    pass
    # from scipy import spatial
    # #TODO:need to think about the inheritance of the vertices, edges and wires 
    # srf_type = face.surface_type
    # if srf_type == geom.SrfType.POLYGON:
    #     nrml = face.surface.normal
    #     if nrml != np.array([0,0,1]) or nrml != np.array([0,0,-1]):
    #         #it must be transformed to be flat
    #         pass
        
    #     bdry_wire = face.bdry_wire
    #     bdry_verts = get.vertices_frm_wire(bdry_wire)
    #     xyz_list = np.array([v.point.xyz for v in bdry_verts])
        
    #     hole_wire_list = face.hole_wire_list
        
    #     hole_verts = []
    #     hole_xyz_list = np.array([pt.xyz for hole in hole_list for 
    #                               pt in hole])
    
    #     #TODO transform the points from 3d to 2d
    #     xy_list = np.delete(xyz_list, 2, axis=1)
    #     hole_xy_list = np.delete(hole_xyz_list, 2, axis=1)
        
    #     d_xy_list = np.concatenate((xy_list, hole_xy_list))
    #     tri = spatial.Delaunay(d_xy_list)
        
    #     chosen = d_xy_list[tri.simplices]
    #     # print(chosen)
    #     # for indices in tri.simplices:
    #     #     fp = d_xy_list[indices]
    #     #     print(fp)
    #         # create.polygon_face_frm_verts(vertex_list)
    #         # pt1 = list(xyz[verts[0]])
    #         # pt2 = list(xyz[verts[1]])
    #         # pt3 = list(xyz[verts[2]])
    #         # occtriangle = make_polygon([pt1,pt2,pt3])
    #         # tri_area = calculate.face_area(occtriangle)
    #         # if tri_area > tolerance:
    #         #     occtriangles.append(occtriangle)