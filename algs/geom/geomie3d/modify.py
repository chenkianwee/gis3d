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
import copy

import numpy as np

from . import get
from . import create
from . import earcut
from . import calculate
from . import geom
from . import topobj
from . import utility


def fuse_vertices(vertex_list: list[topobj.Vertex], decimals: int = 6) -> list[topobj.Vertex]:
    """
    This function fuses the vertices, duplicate vertices are fuse into a single vertex.
    
    Parameters
    ----------
    vertex_list : list[topobj.Vertex]
        A list of vertex topology to be fused.
    
    decimals : int, optional
        the precision of the calculation. Default = 6
        
    Returns
    -------
    fused_vertex_list : list[topobj.Vertex]
        A numpy array of the fused vertices
    """
    xyz_list = np.array([v.point.xyz for v in vertex_list])
    xyz_list = np.round(xyz_list, decimals = decimals)
    vals, u_ids, inverse = np.unique(xyz_list, axis = 0,
                                      return_inverse=True,
                                      return_index=True)
    
    dup_ids = utility.id_dup_indices_1dlist(inverse)
    for cnt,d_id in enumerate(dup_ids):
        d = np.take(vertex_list, d_id, axis=0)
        att_list = [v.attributes for v in d]
        new_dict = {"fused_vertex"+str(x):att_list[x] 
                    for x in range(len(att_list)) if att_list[x]}
        vertex_list[d_id[0]].update_attributes(new_dict)
        
    u_ids = np.sort(u_ids)
    unique_v = np.take(vertex_list, u_ids, axis=0)
    return unique_v

def fuse_points(point_list: list[geom.Point], decimals: int = 6) -> list[geom.Point]:
    """
    This function fuses the points, duplicate points are fuse into a single point.
    
    Parameters
    ----------
    point_list : list[geom.Point]
        A list of point object to be fused.
    
    decimals : int, optional
        the precision of the calculation. Default = 6

    Returns
    -------
    fused_point_list : list[geom.Point]
        A numpy array of the fused points
    """
    
    xyz_list = np.array([point.xyz for point in point_list])
    xyz_list = np.round(xyz_list, decimals = decimals)
    vals, u_ids, inverse = np.unique(xyz_list, axis = 0,
                                      return_inverse=True,
                                      return_index=True)        
    u_ids = np.sort(u_ids)
    unique_p = point_list[u_ids]
    return unique_p

def triangulate_face(face, indices = False):
    """
    Triangulates a face.
 
    Parameters
    ----------
    face : topo object
        the face object to be triangulated.
        
    indices : bool, optional
        Specify whether to an array of [xyzs, indices]. If True will not return the points. Default = False.
 
    Returns
    -------
    list of face : list of face
        A list of triangles constructed from the meshing.
    """
    srf_type = face.surface_type
    if srf_type == geom.SrfType.POLYGON:
        bdry_verts = get.bdry_vertices_frm_face(face)
        nbdry_verts = len(bdry_verts)
        hole_verts_2dlist = get.hole_vertices_frm_face(face)
        nholes = len(hole_verts_2dlist)
        if nholes > 0:
            hole_verts_list = hole_verts_2dlist.flatten()
            all_verts = np.append(bdry_verts, hole_verts_list)
            nhole_verts_ls = [len(hole_verts) for hole_verts in hole_verts_2dlist]
            hole_idxs = [nbdry_verts]
            for cnt,nhole_verts in enumerate(nhole_verts_ls):
                if cnt != nholes-1:    
                    hole_idx = nbdry_verts + nhole_verts
                    hole_idxs.append(hole_idx)
                    # print(hole_idxs)
            
        else:
            all_verts = bdry_verts
            hole_idxs = None
        
        all_xyzs_2dlist = np.array([vert.point.xyz for vert in all_verts])
        
        #if face is not facing z rotate the xyzs here 
        nrml = get.face_normal(face)
        angle = calculate.angle_btw_2vectors([0,0,1], nrml)
        if 180 > angle > 0:
            #means it is not facing z up
            #turn it to z-up
            axis = calculate.cross_product([0,0,1], nrml)
            rot_matrice = calculate.rotate_matrice(axis, angle)
            xyz2tri_2darr = calculate.trsf_xyzs(all_xyzs_2dlist, rot_matrice)
        else:
            xyz2tri_2darr = all_xyzs_2dlist
        
        #FLATTEN ALL THE XYZS AND THEN TRIANGULATE THEM WITH EARCUT
        all_xyzs = xyz2tri_2darr.flatten().tolist()
        tri_idxs = earcut.earcut(all_xyzs, hole_idxs, 3)
        ntri_idxs = len(tri_idxs)
        if ntri_idxs > 0:
            #if it is successfully triangulated
            shape1 = int(ntri_idxs/3)
            
            #check if the vertices is ccw
            tri_idxs_arr = np.array(tri_idxs)
            tri_xyzs = np.take(all_xyzs_2dlist, tri_idxs_arr, axis=0)
            tri_xyzs2 = np.reshape(tri_xyzs,(shape1,3,3))
            tri1 = tri_xyzs2[0]
            is_ccw = calculate.is_anticlockwise(tri1, nrml)
            
            #if it is not ccw flip it to make it ccw
            tri_idxs_2darr = np.reshape(tri_idxs_arr, (shape1,3))
            if is_ccw == False:
                tri_idxs_2darr = np.flip(tri_idxs_2darr, axis = 1)
                
            if indices == False:
                #TODO figure out the inheritance structure
                #reconstruct the triangulated faces 
                tri_idxs_arr = tri_idxs_2darr.flatten()
                tri_verts_arr = np.take(all_verts, tri_idxs_arr, axis = 0)
                tri_verts_2darr = np.reshape(tri_verts_arr,(shape1,3))
                face_atts = face.attributes
                tri_faces = []
                for tri_verts in tri_verts_2darr:    
                    tri_face = create.polygon_face_frm_verts(tri_verts, attributes = face_atts)
                    tri_faces.append(tri_face)
                return tri_faces
            else:
                return [all_xyzs_2dlist, tri_idxs_2darr]
        else:
            return []
        
    if srf_type == geom.SrfType.BSPLINE:
        #check if the surface is planar
        params = utility.gen_gridxyz([0,1,6], [0,1,6])
        f_xyzs = face.surface.evaluate_list(params)
        f_vs = create.vertex_list(f_xyzs)
        is_planar = calculate.is_coplanar(f_vs)
        if is_planar:
            # the surface is planar, can triangulate it like polygon
            columns = 2
            rows = 2
            gpts = create.grid_pts_frm_bspline_face(face, columns+1, rows+1,
                                                    xyzs=True)
            
        else:
            # grid the bspline
            columns = 15
            rows = 15
            gpts = create.grid_pts_frm_bspline_face(face,columns+1,rows+1,
                                                    xyzs=True)
        ngrids = columns*rows
        if indices == False:
            #TODO figure out the inheritance structure
            flist = []
            for cnt in range(ngrids):
                id1 = cnt + int(cnt/columns)
                id2 = id1 + 1
                id3 = id1 + columns+2
                id4 = id3 - 1
                # create polygon surface
                f_xyz1 = [gpts[id1], gpts[id2], gpts[id3]]
                f_v1 = create.vertex_list(f_xyz1)
                f1 = create.polygon_face_frm_verts(f_v1)
                flist.append(f1)
                f_xyz2 = [gpts[id1], gpts[id3], gpts[id4]]
                f_v2 = create.vertex_list(f_xyz2)
                f2 = create.polygon_face_frm_verts(f_v2)
                flist.append(f2)
            return flist
            
        else:
            indx = []
            for cnt in range(ngrids):
                id1 = cnt + int(cnt/columns)
                id2 = id1 + 1
                id3 = id1 + columns+2
                id4 = id3 - 1
                # create triangles
                tri1 = [id1, id2, id3]
                tri2 = [id1, id3, id4]
                indx.append(tri1)
                indx.append(tri2)
            return [np.array(gpts), np.array(indx)]
            
def faces2mesh(face_list):
    """
    This function converts faces to a mesh for visualisation.
 
    Parameters
    ----------
    face_list : list of face object
        the list of face object to be triangulated.The face must be single facing, it
        cannot have regions that are facing itself.
 
    Returns
    -------
    mesh_dictionary : dictionary
        vertices: ndarray of shape(Npoints,3) of the vertiices
        indices: ndarray of shape(Ntriangles,3) indices of the triangles.
    """
    all_xyzs = []
    all_idxs = []
    idx_cnt = 0
    for f in face_list:
        srf_type = f.surface_type
        if srf_type == geom.SrfType.POLYGON or srf_type == geom.SrfType.BSPLINE:
            # vs = get.vertices_frm_face(f)
            # xyzs = [v.point.xyz for v in vs]
            # print(xyzs)
            xyzs_indxs = triangulate_face(f, indices = True)
            # print(xyzs_indxs)
            if len(all_xyzs) == 0:
                all_xyzs = xyzs_indxs[0]
            else:
                all_xyzs = np.append(all_xyzs, xyzs_indxs[0], axis=0)
            
            if len(all_idxs) == 0:
                all_idxs = xyzs_indxs[1]
            else:
                indxs_new =  xyzs_indxs[1] + idx_cnt
                all_idxs = np.append(all_idxs, indxs_new, axis=0)
        
            idx_cnt += len(xyzs_indxs[0])
        
    return {"vertices":all_xyzs, "indices":all_idxs}

def edges2lineedges(edge_list: list[topobj.Edge]) -> list[topobj.Edge]:
    """
    converts edges to line edges.
 
    Parameters
    ----------
    edge_list : list[topobj.Edge]
        the list of edge object to convert to line edges.
 
    Returns
    -------
    line_edges : list[topobj.Edge]
        the list of converted line edges.
    """
    line_edge_list = []
    
    for e in edge_list:
        vs = get.vertices_frm_edge(e)
        if len(vs) == 2:
            line_edge_list.append(e)
        else:
            xyzs = np.array([v.point.xyz for v in vs])
            #repeat each point and remove the first and last point after the repeat
            xyzs = np.repeat(xyzs, 2, axis=0)[1:-1]
            xyzs = np.reshape(xyzs, (int(len(xyzs)/2), 2))
            for e_xyz in xyzs:
                v2 = create.vertex_list(e_xyz)
                line_edge = create.pline_edge_frm_verts(v2)
                line_edge_list.append(line_edge)

    return line_edge_list 

def edges2lines(edge_list):
    """
    converts edges to lines for visualisation.
 
    Parameters
    ----------
    edge_list : list of edge object
        the list of edge object to convert to lines.
 
    Returns
    -------
    vertices : shape(Npoints,3) of the vertiices
        vertices of the lines
    """
    all_xyzs = []
    
    for e in edge_list:
        vs = get.vertices_frm_edge(e)
        xyzs = np.array([v.point.xyz for v in vs])
        #repeat each point and remove the first and last point after the repeat
        xyzs = np.repeat(xyzs, 2, axis=0)[1:-1]
        
        if len(all_xyzs) == 0:
            all_xyzs = xyzs
        else:
            all_xyzs = np.append(all_xyzs, xyzs, axis=0)
    
    return all_xyzs 

def move_topo(topo, target_xyz, ref_xyz = None):
    """
    move topology to the target xyz
 
    Parameters
    ----------
    topo : topo object
        the topo object to move.
        
    target_xyz: np.ndarray
        target xyz position.
    
    ref_xyz: np.ndarray
        reference xyz position, if not specified will use midpt of the topology.
 
    Returns
    -------
    mv_topo : topo
        moved topo
    """
    #TODO: implement for bspline geometries
    topotype = topo.topo_type
    if topotype == topobj.TopoType.EDGE:
        print('I am in bspline edge')
        if topo.curve_type == geom.CurveType.BSPLINE:
            print('not yet implemented for bspline curve')
            return None
    
    if topotype == topobj.TopoType.FACE:
        if topo.surface_type == geom.SrfType.BSPLINE:
            print('not yet implemented for bspline surface')
            return None

    mv_topo = copy.deepcopy(topo)
    vs = get.topo_explorer(mv_topo, topobj.TopoType.VERTEX)
    if ref_xyz is None:
        #find the midpt of the topo and use that as the ref xyz
        bbox = calculate.bbox_frm_topo(topo)
        ref_xyz = calculate.bbox_centre(bbox)
    tx = target_xyz[0] - ref_xyz[0]
    ty = target_xyz[1] - ref_xyz[1]
    tz = target_xyz[2] - ref_xyz[2]
    
    trsl_mat = calculate.translate_matrice(tx, ty, tz)
    xyzs = np.array([v.point.xyz for v in vs])
    trsf_xyzs = calculate.trsf_xyzs(xyzs, trsl_mat)
    #assigned to moved xyz back to the topo
    for cnt,v in enumerate(vs): v.point.xyz = trsf_xyzs[cnt]
    return mv_topo

def rotate_topo(topo, axis, rotation, ref_xyz = None):
    """
    rotate topology
 
    Parameters
    ----------
    topo : topo object
        the topo object to move.
        
    axis: tuple
        tuple specifying the axis to rotate along.
    
    rotation: float
        rotation in degrees, anticlockwise.
        
    ref_xyz: np.ndarray
        reference xyz position, if not specified will use midpt of the topology.
 
    Returns
    -------
    mv_topo : topo
        moved topo
    """
    #TODO: implement for bspline geometries
    topotype = topo.topo_type
    if topotype == topobj.TopoType.EDGE:
        if topo.curve_type == geom.CurveType.BSPLINE:
            print('not yet implemented for bspline curve')
            return None
    
    if topotype == topobj.TopoType.FACE:
        if topo.surface_type == geom.SrfType.BSPLINE:
            print('not yet implemented for bspline surface')
            return None
    
    rot_topo = copy.deepcopy(topo)
    if ref_xyz is None:
        #find the midpt of the topo and use that as the ref xyz
        bbox = calculate.bbox_frm_topo(topo)
        ref_xyz = calculate.bbox_centre(bbox)
    #move the topo to origin for the rotation
    tx = 0 - ref_xyz[0]
    ty = 0 - ref_xyz[1]
    tz = 0 - ref_xyz[2]
    trsl_mat = calculate.translate_matrice(tx, ty, tz)
    rot_mat = calculate.rotate_matrice(axis, rotation)
    trsl_mat2 = calculate.inverse_matrice(trsl_mat)
    
    vs = get.topo_explorer(rot_topo, topobj.TopoType.VERTEX)
    xyzs = np.array([v.point.xyz for v in vs])
    trsf_xyzs = calculate.trsf_xyzs(xyzs, trsl_mat2@rot_mat@trsl_mat)
    # trsf_xyzs = calculate.trsf_xyzs(xyzs, trsl_mat)

    for cnt,v in enumerate(vs): 
        v.point.xyz = trsf_xyzs[cnt]
    
    #get all the surfaces and update their normals
    if rot_topo.topo_type == topobj.TopoType.FACE:
        faces = [rot_topo]
    elif rot_topo.topo_type == topobj.TopoType.SHELL:
        faces = get.faces_frm_solid(rot_topo)
    elif rot_topo.topo_type == topobj.TopoType.SOLID:
        faces = get.faces_frm_solid(rot_topo)
    elif rot_topo.topo_type == topobj.TopoType.COMPOSITE:
        faces = get.faces_frm_composite(rot_topo)
    
    if len(faces) != 0:
        for f in faces:
            f.update_polygon_surface()
            
    return rot_topo
    
def reverse_face_normal(face):
    """
    reverse the normal of the face
 
    Parameters
    ----------
    face : topo object
        the face to reverse.
        
    Returns
    -------
    reversed_face : topo
        the reversed face
    """
    #check the surface type of the face
    if face.surface_type == geom.SrfType.POLYGON:
        bverts = get.bdry_vertices_frm_face(face)
        hverts = get.hole_vertices_frm_face(face)
        bxyzs = np.array([v.point.xyz for v in bverts])
        bxyzs = np.flip(bxyzs, axis=0)
        bverts = create.vertex_list(bxyzs)
        if len(hverts) == 0:    
            flip_f = create.polygon_face_frm_verts(bverts, attributes=face.attributes)
        else:
            hverts_ls = []
            for h in hverts:     
                hxyzs = np.array([v.point.xyz for v in h])
                hxyzs = np.flip(hxyzs, axis=0)
                hverts = create.vertex_list(hxyzs)    
                hverts_ls.append(hverts)
            flip_f = create.polygon_face_frm_verts(bverts, 
                                                   hole_vertex_list = hverts_ls,
                                                   attributes = face.attributes)
        return flip_f
    
    elif face.surface_type == geom.SrfType.BSPLINE:
        srf = face.surface
        cp2d = np.array(srf.ctrlpts2d)
        #flipped it to change the normals
        cp2d = np.flip(cp2d, axis=1)
        arr_shp = np.shape(cp2d)
        cp = np.reshape(cp2d, (arr_shp[0]*arr_shp[1], 3))
        cp = cp.tolist()
        kv_u = arr_shp[0]
        kv_v = arr_shp[1]
        
        deg_u = srf.degree_u
        deg_v = srf.degree_v
        flip_f = create.bspline_face_frm_ctrlpts(cp, kv_u, kv_v, deg_u, deg_v, attributes=face.attributes)
        return flip_f 

def update_topo_att(topo, attributes):
    """
    update the attributes of the topology
 
    Parameters
    ----------
    topo : topo object
        topology to update.
    
    attributes: dictionary
        the attributes
    """
    topo.update_attributes(attributes)

def overwrite_topo_att(topo, attributes):
    """
    update the attributes of the topology
 
    Parameters
    ----------
    topo : topo object
        topology to update.
    
    attributes: dictionary
        the attributes
    """
    topo.overwrite_attributes(attributes)

def xyzs2voxs(xyzs, xdim, ydim, zdim):
    """
    This function group the vertices into a 3d voxel grid.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the point.
    
    xdim : float
        x dimension of a voxel
    
    ydim : float
        y dimension of a voxel
    
    zdim : float
        z dimension of a voxel
    
    Returns
    -------
    voxel_dictionary : a dictionary of voxels
        a dictionary with key (i,j,k) for each voxel and the value as the index of the point
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
        
    xyzs_t = xyzs.T
    x = xyzs_t[0]
    y = xyzs_t[1]
    z = xyzs_t[2]
    
    mnx = np.amin(x)
    mny = np.amin(y)
    mnz = np.amin(z)
    
    i = np.fix((x - mnx)/xdim)
    j = np.fix((y - mny)/ydim)
    k = np.fix((z - mnz)/zdim)
    ijks = np.stack((i,j,k), axis=-1)
    vox_props = {'voxel_dim': [xdim, ydim, zdim]}
    voxs = {}
    for cnt,ijk in enumerate(ijks):
        ijk = ijk.tolist()
        ijk = list(map(int,ijk))
        ijk = tuple(ijk)
        if ijk in voxs:
            voxs[ijk]['idx'].append(cnt)
        else:
            mpx = ijk[0] * xdim + (xdim/2) + mnx
            mpy = ijk[1] * ydim + (ydim/2) + mny
            mpz = ijk[2] * zdim + (zdim/2) + mnz
            voxs[ijk] = {'idx':[cnt], 'midpt':[mpx,mpy,mpz]}
    vox_props['voxels'] = voxs
    return vox_props

def trsf_cs(cs1, cs2, topo):
    pass
