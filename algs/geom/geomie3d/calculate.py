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
import math
from itertools import chain
import numpy as np
from numpy.linalg import matrix_rank

from . import get
from . import modify
from . import geom
from . import create
from . import topobj
from . import utility

def dist_btw_xyzs(xyzs1, xyzs2):
    """
    This function calculates the distance between two xyz.
 
    Parameters
    ----------    
    xyzs1 : ndarray
        array defining the point.
        
    xyzs2 : ndarray
        array defining the point.
 
    Returns
    -------
    distance : float
        the distance(s) between the points
    """
    if type(xyzs1) != np.ndarray:
        xyzs1 = np.array(xyzs1)
    if type(xyzs2) != np.ndarray:
        xyzs2 = np.array(xyzs2)
    
    xyzs12 = (xyzs2-xyzs1)**2
    nshape = len(xyzs12.shape)
    if nshape > 1:    
        xyzsum = xyzs12.sum(axis=1)
    else:
        xyzsum = xyzs12.sum()
    # print(xyzsum)
    distance = np.sqrt(xyzsum)
    return distance

def reverse_vectorxyz(vector_xyzs: np.ndarray) -> np.ndarray:
    """
    This function calculates the reverse of the vectorxyz . 
 
    Parameters
    ----------
    vector_xyzs : np.ndarray
        array defining the point.
    
    Returns
    -------
    reverse_vector : np.ndarray
        The reverse of all the points.
    """
    if type(vector_xyzs) != np.ndarray:
        vector_xyzs = np.array(vector_xyzs)

    return vector_xyzs*-1
    
def xyzs_mean(xyzs):
    """
    This function calculates the mean of all points. 
 
    Parameters
    ----------
    xyzs : ndarray
        array defining the point.
    
    Returns
    -------
    midpt : xyz
        The mean of all the points.
    """
    shape = np.shape(xyzs)
    nshape = len(shape)
    if nshape == 1:
        raise ValueError("arrays have arrays of different length, this is not allowed")
    
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if nshape == 2: #only 1 set of points
        centre_pt = np.mean(xyzs, axis=0)
        return centre_pt
    
    if nshape == 3:
        centre_pt = np.mean(xyzs, axis=1)
        return centre_pt

def bbox_frm_xyzs(xyzs):
    """
    This function returns the bbox of the xyzs.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the point.

    Returns
    -------
    bbox : bbox object
        bbox object
    """
    import numpy as np
    
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)

    xyzs_t = xyzs.T
    x = xyzs_t[0]
    y = xyzs_t[1]
    z = xyzs_t[2]
    
    mnx = np.amin(x)
    mxx = np.amax(x)
    
    mny = np.amin(y)
    mxy = np.amax(y)
    
    mnz = np.amin(z)
    mxz = np.amax(z)
    
    bbox = utility.Bbox([mnx,mny,mnz,mxx,mxy,mxz])
    return bbox
    
def bboxes_centre(bboxes: list[utility.Bbox]) -> np.ndarray:
    """
    This function returns the centre point of the bbox .
    
    Parameters
    ----------
    bbox : list[utility.Bbox]
        list[utility.Bbox] to calculate the center
       
    Returns
    -------
    xyzs : np.ndarray
        np.ndarray(shape(number of bboxes, 3)) array defining the points.
    """
    bbox_arrs = np.array([bbox.bbox_arr for bbox in bboxes])
    bbox_arrsT = bbox_arrs.T
    mnxs = bbox_arrsT[0]
    mnys = bbox_arrsT[1]
    mnzs = bbox_arrsT[2]
    mxxs = bbox_arrsT[3]
    mxys = bbox_arrsT[4]
    mxzs = bbox_arrsT[5]

    midxs = mnxs + ((mxxs - mnxs)/2)
    midys = mnys + ((mxys - mnys)/2)
    midzs = mnzs + ((mxzs - mnzs)/2)
    center_pts = np.vstack([midxs, midys, midzs]).T

    return center_pts
    
def bbox_frm_bboxes(bbox_list: list[utility.Bbox]) -> utility.Bbox:
    """
    This function recalculate the bbox from a list of bboxes.
    
    Parameters
    ----------
    bbox_list : list[utility.Bbox]
        A list of bboxes    
    
    Returns
    -------
    bbox : utility.Bbox
        the calculated bbox object
    """
    bbox_arr_ls = [bbox.bbox_arr.tolist() for bbox in bbox_list]
    zip_box = list(zip(*bbox_arr_ls))
    mnx = min(zip_box[0])
    mny = min(zip_box[1])
    mnz = min(zip_box[2])
    
    mxx = max(zip_box[3])
    mxy = max(zip_box[4])
    mxz = max(zip_box[5])
    
    res_bbox = utility.Bbox([mnx,mny,mnz,mxx,mxy,mxz])
    return res_bbox

def are_bboxes1_related2_bboxes2(bboxes1: list[utility.Bbox], bboxes2: list[utility.Bbox], zdim: bool = True) -> np.ndarray:
    """
    This function calculates if bbox1 is related to bbox2. Related include any form of intersection or touching.
    
    Parameters
    ----------
    bboxes1 : list[utility.Bbox]
        the bboxes to check

    bboxes2 : list[utility.Bbox]
        the bboxes to check    
    
    zdim : bool, optional
        If True will check the z-dimension. Default = True

    Returns
    -------
    is_inside : np.ndarray
        An array of True or False indicating the relationship between the two list of bboxes. True = bboxes are related, False = bboxes are not related.
    """
    def are_bboxes_related(bboxes1, bboxes2, nbboxes, zdim):
        bbox_xyzs = create.xyzs_frm_bboxes(bboxes1)
        bboxes_shape = bbox_xyzs.shape
        bbox_xyzs1 = np.reshape(bbox_xyzs, (bboxes_shape[0]*bboxes_shape[1], bboxes_shape[2]))
        indices = match_xyzs_2_bboxes(bbox_xyzs1, bboxes2, zdim = zdim)
        are_related = np.array([False])
        are_related = np.repeat(are_related, nbboxes)
        if indices.size == 0:
            return are_related 
        else:
            xyz_indices = indices[0]
            bboxes1_indices = xyz_indices/8
            bboxes1_indices = np.floor(bboxes1_indices)
            bboxes2_indices = indices[1]
            bboxes_indices = np.array([bboxes1_indices, bboxes2_indices])
            same_indx = np.where(bboxes1_indices == bboxes2_indices)[0]
            bboxes_indicesT = bboxes_indices.T
            same = np.take(bboxes_indicesT, same_indx, axis=0)
            same = np.unique(same, axis=0)
            sameT = same.T[0].astype(int)
            np.put(are_related, sameT, True)
            return are_related

    nbboxes = len(bboxes1)
    are_related1 = are_bboxes_related(bboxes1, bboxes2, nbboxes, zdim)
    are_related2 = are_bboxes_related(bboxes2, bboxes1, nbboxes, zdim)
    are_related_res = np.logical_or(are_related1, are_related2)
    return are_related_res

def is_collinear(vertex_list):
    """
    This function checks if the list of points are collinear. 
 
    Parameters
    ----------
    vertex_list : ndarray
        array of vertices.
        
    Returns
    -------
    True or False : bool
        If True the list of points are collinear.
    """
    if type(vertex_list) != np.ndarray:
        vertex_list = np.array(vertex_list)
    
    is_2d = isinstance(vertex_list[0], np.ndarray)
    
    if not is_2d:
        xyzs = [v.point.xyz for v in vertex_list]
        
    else:
        xyzs = []
        for verts in vertex_list:
            xyz = [v.point.xyz for v in verts]
            xyzs.append(xyz)
            
    return is_collinear_xyzs(xyzs)

def is_coplanar(vertex_list):
    """
    This function checks if the list of points are coplanar. 
 
    Parameters
    ----------
    vertex_list : ndarray
        array of vertices. It can be a 2darray of vertex list
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    if type(vertex_list) != np.ndarray:
        vertex_list = np.array(vertex_list)
    
    is_2d = isinstance(vertex_list[0], np.ndarray)
    
    if not is_2d:
        xyzs = [v.point.xyz for v in vertex_list]
        
    else:
        xyzs = []
        for verts in vertex_list:
            xyz = [v.point.xyz for v in verts]
            xyzs.append(xyz)
            
    return is_coplanar_xyzs(xyzs)

def _affine_rank(xyzs):
    """
    This function calculate the affine rank of the xyzs. If having a 3d array.
    
    Parameters
    ----------
    xyzs : ndarray
        array of xyzs, [xyzs1, xyzs2, xyzs3]. each xyz is defined as [x1,y1,z1]. This function takes multiple sets of points and check their coplanarity. [point_set1, point_set2, point_setx].
        
    Returns
    -------
    affine_rank : ndarray
        affine rank of the given xyzs.
    """
    #--------------------------------------------------------------------------------------------------------------------------------
    def rank_affine(xyzs):
        centroid = xyzs_mean(xyzs)
        centered_pts = xyzs - centroid
        affine_rank = matrix_rank(centered_pts)
        return affine_rank
    #--------------------------------------------------------------------------------------------------------------------------------
    shape = np.shape(xyzs)
    nshape = len(shape)
    if nshape == 1:
        raise ValueError(
            "this is a 1d array. This sometimes happens if arrays have arrays of different length, this is not allowed")

    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)

    if nshape == 2:  # only 1 set of points
        xyzs = np.unique(xyzs, axis=0)
        affine_rank = rank_affine(xyzs)
        return affine_rank

    elif nshape == 3:
        uniqs = [np.unique(u, axis=0) for u in xyzs]
        shape1 = np.shape(uniqs)
        nshape1 = len(shape1)
        if nshape1 == 1:
            affine_rank = [rank_affine(uniq) for uniq in uniqs]
        elif nshape1 == 3:
            uniqs = np.array(uniqs)
            centroid = xyzs_mean(uniqs)
            nrepeat = np.repeat([shape1[1]], shape1[0])
            centroid = np.repeat(centroid, nrepeat, axis=0)
            centroid = np.reshape(centroid, shape1)
            centered_pts = uniqs - centroid
            affine_rank = matrix_rank(centered_pts)
            return affine_rank
    
def is_coplanar_xyzs(xyzs):
    """
    This function checks if the list of xyzs are coplanar.
    
    Parameters
    ----------
    xyzs : ndarray
        array of xyzs, [xyzs1, xyzs2, xyzs3]. each xyz is defined as [x1,y1,z1]. This function takes multiple sets of points and check their coplanarity. [point_set1, point_set2, point_setx].
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    affine_rank = _affine_rank(xyzs)
    return affine_rank <=2            

def is_collinear_xyzs(xyzs):
    """
    This function checks if the list of xyzs are collinear.
    
    Parameters
    ----------
    xyzs : ndarray
        array of xyzs, [xyzs1, xyzs2, xyzs3]. each xyz is defined as [x1,y1,z1]. This function takes multiple sets of points and check their coplanarity. [point_set1, point_set2, point_setx].
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    affine_rank = _affine_rank(xyzs)
    return affine_rank <=1 
    
def bbox_frm_topo(topo):
    """
    calculate the bbox from a topology object.
    
    Parameters
    ----------
    topo : topology object
        The topology to analyse

    Returns
    -------
    bbox : bbox object
        bbox object
    """
    verts = get.topo_explorer(topo, topobj.TopoType.VERTEX)
    xyzs = np.array([v.point.xyz for v in verts])
    bbox = bbox_frm_xyzs(xyzs)
    return bbox

def xyzs_in_bbox(xyzs: np.ndarray, bbox: utility.Bbox, zdim: bool = True, indices: bool = False) -> np.ndarray:
    """
    This function if the points are within the given boundary box.
 
    Parameters
    ----------
    xyzs : np.ndarray
        array defining the points.
    
    bbox : utility.Bbox
        bbox object
        
    zdim : bool, optional
        If True will check the z-dimension.
        
    indices : bool, optional
        Specify whether to return the indices of the points in the boundary. If True will not return the points. Default = False.
    
    Returns
    -------
    points_in_bdry : np.ndarray
        The points that is in the boundary. If indices==True, this will be the indices instead of the actual points.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    zip_xyzs = xyzs.T
    xlist = zip_xyzs[0] 
    ylist = zip_xyzs[1]
    zlist = zip_xyzs[2]
    
    bbox_arr = bbox.bbox_arr
    mnx = bbox_arr[0]
    mny = bbox_arr[1]
    mxx = bbox_arr[3]
    mxy = bbox_arr[4]
    
    x_valid = np.logical_and((mnx <= xlist),
                             (mxx >= xlist))
    
    y_valid = np.logical_and((mny <= ylist),
                             (mxy >= ylist))
    
    if zdim == True:
        mnz = bbox_arr[2]
        mxz = bbox_arr[5]
        
        z_valid = np.logical_and((mnz <= zlist),
                                 (mxz >= zlist))
        cond1 = np.logical_and(x_valid, y_valid)
        cond2 = np.logical_and(cond1, z_valid)
        index_list = np.where(cond2)[0]
    
    else:
        index_list = np.where(np.logical_and(x_valid, y_valid))[0]
    
    if indices == True:
        if index_list.size > 0:
            return index_list
        else:
            return []
    else:
        pts_in_bdry = np.take(xyzs, index_list, axis = 0)
        return pts_in_bdry

def is_xyz_in_bbox(xyz, bbox, zdim = True):
    """
    This function check if a point is in bounding box.  
 
    Parameters
    ----------
    xyz : ndarray
        array defining the point.
        
    bbox : bbox object
        bbox object
        
    zdim : bool, optional
        If True will check the z-dimension.
    
    Returns
    -------
    in_boundary : bool
        True or False, is the point in bounding box.
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    bbox_arr = bbox.bbox_arr
    mnx = bbox_arr[0]
    mny = bbox_arr[1]
    mnz = bbox_arr[2]
    mxx = bbox_arr[3]
    mxy = bbox_arr[4]
    mxz = bbox_arr[5]
    
    in_bdry = False
    if zdim == True:
        if mnx<=x<=mxx and mny<=y<=mxy and mnz<=z<=mxz:
            in_bdry = True
        
    else:
        if mnx<=x<=mxx and mny<=y<=mxy:
            in_bdry = True
    
    return in_bdry
    
def match_xyzs_2_bboxes(xyzs: np.ndarray, bbox_list: list[utility.Bbox], zdim: bool = True) -> np.ndarray:
    """
    This function returns the point indices follow by the bbox indices which it is contained in.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(nxyzs, 3)) array defining the points.
    
    bbox_list : list[utility.Bbox]
        A list of bbox
        
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    point_bbox_indices : np.ndarray
        np.ndarray(shape(2, number_of_matches)), point indices follow by the bbox indices. The two arrays corresponds to each other.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
        
    def reshape_bdry(nparray, repeat):
        arrx = np.expand_dims(nparray, axis=0)
        arrx = np.repeat(arrx, repeat, axis=0)
        return arrx
    
    npts = len(xyzs)
    
    bbox_arr_ls = np.array([bbox.bbox_arr for bbox in bbox_list])
    tbdry = bbox_arr_ls.T
    
    xmin_list = reshape_bdry(tbdry[0], npts)
    ymin_list = reshape_bdry(tbdry[1], npts)
    xmax_list = reshape_bdry(tbdry[3], npts)
    ymax_list = reshape_bdry(tbdry[4], npts)
    
    tpt = xyzs.T
    xlist = tpt[0]
    xlist.shape = (npts,1)
    
    ylist = tpt[1]
    ylist.shape = (npts,1)
    
    zlist = tpt[2]
    zlist.shape = (npts,1)
    
    x_valid = np.logical_and(xlist >= xmin_list,
                             xlist <= xmax_list)
    
    y_valid = np.logical_and(ylist >= ymin_list,
                             ylist <= ymax_list)
    
    if zdim == True:
        zmin_list = reshape_bdry(tbdry[2], npts)
        zmax_list = reshape_bdry(tbdry[5], npts)
        z_valid = np.logical_and(zlist >= zmin_list,
                                 zlist <= zmax_list)
        cond1 = np.logical_and(x_valid, y_valid)
        xyz_valid = np.logical_and(cond1, z_valid)
    else:
        xyz_valid = np.logical_and(x_valid, y_valid)
    
    index = np.where(xyz_valid)
    index = np.array(index)
    return index

def id_bboxes_contain_xyzs(bbox_list, xyzs, zdim = True):
    """
    This function returns the indices of the bbox which contains the points.
    
    Parameters
    ----------
    bbox_list : a list of bbox object
        A list of bbox
        
    xyzs : ndarray
        array defining the points.
            
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    bbox_indices : nparray
        Indices of the boundary that contains the point.
    """
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    bbox_indices = indices[1]
    bbox_indices = np.unique(bbox_indices)
    
    return bbox_indices

def id_xyzs_in_bboxes(xyzs, bbox_list, zdim = False):
    """
    This function returns the indices of the points that are contained by bboxes.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
        
   bbox_list : a list of bbox object
       A list of bbox
       
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    pt_indices : nparray
        Indices of the points in the bboxes.
    """
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    pt_indices = indices[0]
    pt_indices = np.unique(pt_indices)
    
    return pt_indices

def angle_btw_2vectors(vector1, vector2):
    """
    This function calculate the angle between two vectors.
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector(s).
        
    vector2 : ndarray
        array defining the vector(s).
 
    Returns
    -------
    angle : float
        angle between the two vectors
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
    
    nshape = len(vector1.shape)
    if nshape > 1:    
        dot = np.sum(vector1*vector2, axis=1)
        mag1 = np.sqrt(np.sum(vector1**2, axis=1))
        mag2 = np.sqrt(np.sum(vector2**2, axis=1))
    else:
        dot = np.sum(vector1*vector2)
        mag1 = np.sqrt(np.sum(vector1**2))
        mag2 = np.sqrt(np.sum(vector2**2))
        
    cos_angle = dot/(mag1*mag2)
    angle_rad = np.arccos(cos_angle)
    angle = np.degrees(angle_rad)
    return angle 

def cross_product(vector1, vector2):
    """
    This function cross product two vectors.It is a wrap of the numpy cross function
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector(s).
        
    vector2 : ndarray
        array defining the vector(s).
 
    Returns
    -------
    cross_product : ndarray
        numpy array of the cross product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
     
    cross_product = np.cross(vector1, vector2)
    return cross_product

def dot_product(vector1, vector2):
    """
    This function cross product two vectors. Wrap of numpy dot function
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector.
        
    vector2 : ndarray
        array defining the vector.
 
    Returns
    -------
    dot_product : ndarray
        numpy array of the dot product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
    
    shape = np.shape(vector1)
    nshape = len(shape)
    if nshape == 1:
        dot_product = np.dot(vector1, vector2)
        return dot_product
    elif nshape == 2:
        v1T = vector1.T
        v2T = vector2.T
        a = v1T[0]*v2T[0]
        b = v1T[1]*v2T[1]
        c = v1T[2]*v2T[2]
        dot_product = a+b+c
        return dot_product
        
def normalise_vectors(vector_list):
    """
    This function normalise the vectors.
 
    Parameters
    ----------    
    vector_list : 2d ndarray
        numpy array of the vectors.
 
    Returns
    -------
    normalise_vector : ndarray
        normalise vector.
    """
    if type(vector_list) != np.ndarray:
        vector_list = np.array(vector_list)
    naxis = len(vector_list.shape)
    return vector_list/np.linalg.norm(vector_list, ord = 2, axis = naxis-1, keepdims=True)
    
def translate_matrice(tx,ty,tz):
    """
    This function calculate a 4x4 translation matrice.
 
    Parameters
    ----------    
    tx : float
        magnitude of translation in the x-direction.
    
    ty : float
        magnitude of translation in the y-direction.
        
    tz : float
        magnitude of translation in the z-direction.
 
    Returns
    -------
    matrice : ndarray
        4x4 translation matrice.
    """
    mat = np.array([[1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]])
    
    return mat

def scale_matrice(sx,sy,sz):
    """
    This function calculate a 4x4 translation matrice.
 
    Parameters
    ----------    
    sx : float
        scaling in the x-direction.
    
    sy : float
        scaling in the y-direction.
        
    sz : float
        scaling in the z-direction.
 
    Returns
    -------
    matrice : ndarray
        4x4 scaling matrice.
    """
    
    mat = np.array([[sx, 0, 0, 0],
                    [0, sy, 0, 0],
                    [0, 0, sz, 0],
                    [0, 0, 0, 1]])
    return mat

def rotate_matrice(axis, rotation):
    """
    This function calculate a 4x4 translation matrice. The rotation in counter-clockwise.
 
    Parameters
    ----------    
    axis : tuple
        the axis for rotation, must be a normalised vector.
    
    rotation : float
        the rotation in degrees.
 
    Returns
    -------
    matrice : ndarray
        4x4 rotation matrice.
    """
    x = axis[0]
    y = axis[1]
    z = axis[2]
    #convert degree to radian
    rot_rad = math.radians(rotation)
    cos_ang = math.cos(rot_rad)
    sin_ang = math.sin(rot_rad)
    
    r1 = cos_ang + ((x**2)*(1-cos_ang))
    r2 = (x*y*(1-cos_ang)) - (z*sin_ang)
    r3 = (x*z*(1-cos_ang)) + (y*sin_ang)
    r4 = (y*x*(1-cos_ang)) + (z*sin_ang)
    r5 = cos_ang + ((y**2)*(1-cos_ang))
    r6 = (y*z*(1-cos_ang)) - (x*sin_ang)
    r7 = (z*x*(1-cos_ang)) - (y*sin_ang)
    r8 = (z*y*(1-cos_ang)) + (x*sin_ang)
    r9 = cos_ang + ((z**2)*(1-cos_ang))
    
    mat = np.array([[r1, r2, r3, 0],
                    [r4, r5, r6, 0],
                    [r7, r8, r9, 0],
                    [0, 0, 0, 1]])
    return np.round(mat, decimals = 6)

def inverse_matrice(matrice):
    """
    This function calculate a 4x4 translation matrice. The rotation in counter-clockwise.
 
    Parameters
    ----------    
    matrice : ndarray
        4x4 matrice to inverse.
    
    Returns
    -------
    matrice : ndarray
        inverted 4x4 matrice.
    """
    return np.linalg.inv(matrice)
    
def cs2cs_matrice(orig_cs, dest_cs):
    """
    This function calculate a 4x4 matrice for transforming one coordinate to
    another coordinate system.
 
    Parameters
    ----------    
    orig_cs : coordinate system object
        the original coordinate system.
    
    dest_cs : coordinate system object
        the coordinate system to transform to.
 
    Returns
    -------
    matrice : ndarray
        4x4 matrice.
    """
    # #TODO FINISH THE FUNCTION
    # #first move the origin from the original cs to the destination cs
    # orig1 = orig_cs.origin
    # orig2 = dest_cs.origin
    # trsl = orig2 - orig1
    # trsl_mat = translate_matrice(trsl[0],trsl[1],trsl[2])
    # #then get the rotation axis for the y-axis 
    # yd1 = orig_cs.y_dir
    # yd2 = dest_cs.y_dir
    # rot_axis1 = cross_product(yd1, yd2)
    # #need to get the rotation angle
    # rot_angle = angle_btw_2vectors(yd1, yd2)
    # rot_mat = rotate_matrice(rot_axis1, rot_angle)
    # # print(trsl_mat)
    pass
    
def face_midxyz(face):
    """
    Calculates the midpt of the face
 
    Parameters
    ----------    
    face : the face object
        the face object.
    
    Returns
    -------
    midxyz : ndarray
        array defining the midpt.
    """
    if face.surface_type == geom.SrfType.POLYGON:
        vertex_list = get.vertices_frm_face(face)
        xyzs = np.array([v.point.xyz for v in vertex_list])
        mid_xyz = xyzs_mean(xyzs)
        
    elif face.surface_type == geom.SrfType.BSPLINE:
        bspline_srf = face.surface
        mid_xyz = np.array(bspline_srf.evaluate_single([0.5,0.5]))
    return mid_xyz

def trsf_xyzs(xyzs, trsf_mat):
    """
    Calculates the transformed xyzs based on the given matrice
 
    Parameters
    ----------    
    xyzs : ndarray
        array defining the points.
    
    Returns
    -------
    trsf_xyzs : ndarray
        the transformed xyzs.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    #add an extra column to the points
    npos = len(xyzs)
    xyzw = np.ones((npos,4))
    xyzw[:,:-1] = xyzs
    t_xyzw = np.dot(xyzw, trsf_mat.T)
    trsf_xyzs = t_xyzw[:,:-1]
    
    return trsf_xyzs

def move_xyzs(xyzs, directions, magnitudes):
    """
    Calculates the moved xyzs based on a direction vectors and magnitudes
 
    Parameters
    ----------    
    xyzs : ndarray
        array defining the points.
    
    directions : ndarray
        array defining the directions.
    
    magnitudes : 1d array
        array defining the magnitude to move.
        
    Returns
    -------
    moved_xyzs : ndarray
        the moved xyzs.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if type(directions) != np.ndarray:
        directions = np.array(directions)
        
    if type(magnitudes) != np.ndarray:
        magnitudes = np.array(magnitudes)

    magnitudes = np.reshape(magnitudes, [len(directions),1])
    moved_xyzs = xyzs + directions*magnitudes
    return moved_xyzs 
    
def is_anticlockwise(xyzs, ref_vec):
    """
    This function checks if the list of points are arranged anticlockwise in regards to the ref_pyvec by calculating the winding number. When the number is negative they are clockwise.
    The ref_pyvec must be perpendicular to the points. 
 
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
    
    ref_vec : tuple of floats
        The reference vector must be perpendicular to the list of points. 
        A vec is a tuple that documents the xyz direction of a vector e.g. (x,y,z)
        
    Returns
    -------
    True or False : bool
        If True the list of points are arranged in anticlockwise manner, if False they are not.
    """
    shape = np.shape(xyzs)
    nshape = len(shape)
    if nshape == 1:
        raise ValueError("xyzs is a 1d array. This sometimes happens if it is a 2d arrays have arrays of different length, this is not allowed")
        
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if type(ref_vec) != np.ndarray:
        ref_vec = np.array(ref_vec)
    
    if nshape == 2: #only 1 set of points
        xyzs2 = np.roll(xyzs, -1, axis=0)
        prod = cross_product(xyzs, xyzs2)
        ttl = np.sum(prod, axis=0)
        res = dot_product(ttl, ref_vec)
        if res < 0:
            return False
        elif res > 0:
            return True
        else: #if res == 0 means the points are collinear
            return None
        
    elif nshape == 3:
        xyzs2 = np.roll(xyzs, -1, axis=1)
        #reshape both of the xyzs
        xyzs_rs = np.reshape(xyzs, (shape[0]*shape[1], shape[2]))
        xyzs2_rs = np.reshape(xyzs2, (shape[0]*shape[1], shape[2]))
        prod = cross_product(xyzs_rs, xyzs2_rs)
        prod = np.reshape(prod, shape)
        ttl = np.sum(prod, axis=1)
        res = dot_product(ttl, ref_vec)
        cond1 = np.logical_not(res <= 0) # bigger than 0 means its anticlockwise
        cond2 = np.logical_not(res != 0) #find where are the 0
        cond3 = np.where(cond2, None, cond1) #use the 0 cond rule to change the 0 to None    
        return cond3
        
def acct4obstruction(intPts, mags, rayIndxs, objIndxs):
    indx = np.arange(len(rayIndxs))
    dupIds = utility.id_dup_indices_1dlist(rayIndxs)
    dupIds_flat = list(chain(*dupIds))
    # print(dupIds)
    non_obs_indx = utility.find_xs_not_in_ys(indx, dupIds_flat)
    for di in dupIds:
        di.sort()
        mag = mags[di]
        index = np.where(mag == mag.min())[0]
        index = index.flatten()
        index = index + di[0]
        non_obs_indx = np.append(non_obs_indx, index)
    
    non_obs_indx.sort()
    nb_intPts = np.take(intPts, non_obs_indx, axis=0)
    nb_mags = np.take(mags, non_obs_indx, axis=0)
    nb_rayIndxs = np.take(rayIndxs, non_obs_indx, axis=0)
    nb_objIndxs = np.take(objIndxs, non_obs_indx, axis=0)
    return nb_intPts, nb_mags, nb_rayIndxs, nb_objIndxs

def rays_xyz_tris_intersect(rays_xyz, tris_xyz):
    """
    This function intersect multiple rays with multiple triangles. https://github.com/johnnovak/raytriangle-test/blob/master/python/perftest.py 
 
    Parameters
    ----------
    rays_xyz : ndarray
        array of rays [ray_xyz1,ray_xyz2,...]. Each ray is defined with [[origin], [direction]].
        
    tris_xyz : ndarray
        array of triangles [tri_xyz1, tri_xyz2, ...]. Each triangle is define with [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].

    Returns
    -------
    intersect_pts : ndarray
        points of intersection. If no intersection returns an empty array
        
    magnitudes : ndarray
        array of magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : ndarray
        indices of the rays. Corresponds to the intersect pts.
    
    tris_index : ndarray
        indices of the intersected triangles. Corresponds to the intersect pts.
    """
    dets_threshold = 1e-06
    precision = 6
    if type(rays_xyz) != np.ndarray:
        rays_xyz = np.array(rays_xyz)
    
    if type(tris_xyz) != np.ndarray:
        tris_xyz = np.array(tris_xyz)
    
    nrays = len(rays_xyz)
    ntris = len(tris_xyz)

    tris_xyz_tiled = np.tile(tris_xyz, (nrays,1,1))
    trisT = np.stack(tris_xyz_tiled, axis=1)
    xyzs1_0 = trisT[1] - trisT[0]
    xyzs2_0 = trisT[2] - trisT[0]
    
    rays_repeat = np.repeat(rays_xyz, ntris, axis=0)
    raysT = np.stack(rays_repeat, axis=1)
    pvecs = cross_product(raysT[1], xyzs2_0)
    dets = dot_product(xyzs1_0, pvecs)
    dets_round = np.around(dets, decimals=precision)
    # print('dets', dets)
    dets_true = np.logical_not(dets_round<dets_threshold)
    dets_index = np.where(dets_true)[0]
    # print(dets_index)
    if len(dets_index) == 0:
        #no intersections at all
        return [],[],[],[]
    #============================================
    dets = np.where(dets_true, dets, np.inf)
    invDet = np.where(dets_true, 1.0/dets, dets)
    invDet = np.where(dets_true, invDet, 0)
    dets_true_repeat = np.repeat(dets_true, 3)
    dets_true_repeat_reshape = np.reshape(dets_true_repeat, (len(raysT[0]),3))
    
    tvecs = np.where(dets_true_repeat_reshape, 
                    raysT[0] - trisT[0], 
                    np.zeros((len(raysT[0]), 3)))
    us = np.where(dets_true, 
                  dot_product(tvecs, pvecs)*invDet, 
                  np.zeros(len(dets_true)))
    
    us = np.around(us, decimals=precision)
    us_true = np.logical_and(us>=0.0-dets_threshold, us<=1.0+dets_threshold)
    us_true = np.logical_and(us_true, dets_true)
    us_index = np.where(us_true)[0]
    if len(us_index) == 0:
        #no intersections at all 
        return [],[],[],[]
    #============================================
    us_true_repeat = np.repeat(us_true, 3)
    us_true_repeat_reshape = np.reshape(us_true_repeat, (len(tvecs),3))
    
    qvecs = np.where(us_true_repeat_reshape,
                     cross_product(tvecs, xyzs1_0),
                     np.zeros([len(tvecs),3]))
    vs = np.where(us_true, 
                  dot_product(raysT[1], qvecs)*invDet,
                  np.zeros([len(us_true)]))
    vs = np.around(vs, decimals = precision)
    vs_true1 = np.logical_not(vs < 0.0-dets_threshold)
    usvs = us+vs
    usvs = np.around(usvs, decimals=precision)
    vs_true2 = np.logical_not(usvs > 1.0+dets_threshold)
    vs_true = np.logical_and(vs_true1, vs_true2)
    vs_true = np.logical_and(vs_true, us_true)
    mags = np.where(vs_true, 
                    dot_product(xyzs2_0, qvecs)*invDet,
                    np.zeros([len(vs_true)]))
    
    mags = np.around(mags, decimals=precision)
    mags_true = np.logical_not(mags < dets_threshold)
    mags = np.where(mags_true, mags, np.zeros([len(vs_true)]))
    # print('mags',mags)
    mags_reshape = np.reshape(mags,[len(mags),1])
    dir_mags = raysT[1]*mags_reshape
    intPts = raysT[0] + dir_mags
    #============================================
    #separate the intersections into ray-based groups
    #sub zeros with infinity to make processing easier later
    mags = np.where(mags_true, mags, np.full([len(vs_true)], np.inf))
    mags_ray_reshape = np.reshape(mags, (nrays,ntris))
    mags_true = np.logical_not(mags_ray_reshape==np.inf)
    mags_true_index = np.where(mags_true)
    intPts_reshape = np.reshape(intPts, (nrays,ntris,3))
    res_pts = intPts_reshape[mags_true_index]
    res_mag = mags_ray_reshape[mags_true_index]
    res_ray_indx = mags_true_index[0]
    res_tri_indx = mags_true_index[1]
    res_pts, res_mag, res_ray_indx, res_tri_indx= acct4obstruction(res_pts, 
                                                                   res_mag, 
                                                                   res_ray_indx, 
                                                                   res_tri_indx)
    
    return res_pts, res_mag, res_ray_indx, res_tri_indx

def rays_xyz_bboxes_intersect(rays_xyz, bbox_list):
    """
    This function intersect multiple rays with multiple bboxes. based on this https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection and this https://github.com/stackgl/ray-aabb-intersection/blob/master/index.js 
    
    Parameters
    ----------
    rays_xyz : ndarray
        array of rays [ray_xyz1,ray_xyz2,...]. Each ray is defined with [[origin], [direction]].
        
    bbox_list : a list of bbox object
        A list of bbox

    Returns
    -------
    intersect_pts : ndarray
        points of intersection. If no intersection returns an empty array
        
    magnitudes : ndarray
        array of magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : ndarray
        indices of the rays. Corresponds to the intersect pts.
    
    tris_index : ndarray
        indices of the intersected triangles. Corresponds to the intersect pts.
    """
    if type(rays_xyz) != np.ndarray:
        rays_xyz = np.array(rays_xyz)
    
    precision = 6
    nrays = len(rays_xyz)
    nbbox = len(bbox_list)
    bbox_arr_ls = np.array([bbox.bbox_arr for bbox in bbox_list])
    bbox_arr_ls = np.around(bbox_arr_ls, decimals=precision)
    #sort the bbox to match the number of rays, each ray to all the bboxes
    bbox_tiled = np.tile(bbox_arr_ls, (nrays,1))
    bboxT = bbox_tiled.T
    #sort the rays to match the boxs 
    rays_repeat = np.repeat(rays_xyz, nbbox, axis=0)
    rays_stacked = np.stack(rays_repeat, axis=1)
    rays_origs = np.around(rays_stacked[0], decimals=precision)
    rays_dirs = np.around(rays_stacked[1], decimals=precision)
    # rays_origs = rays_stacked[0]
    # rays_dirs = rays_stacked[1]
    rays_origsT = rays_origs.T
    rays_dirsT = rays_dirs.T
    rays_dirsT = np.where(rays_dirsT == 0, np.inf, rays_dirsT)
    #============================================
    #calculation to detect intersections
    rdirxs = rays_dirsT[0]
    x0_true = np.where(rdirxs == np.inf, True, False)
    txmin0 = bboxT[0] - rays_origsT[0]
    txmin = txmin0 / rdirxs
    txmax0 = bboxT[3] - rays_origsT[0]
    txmax = txmax0 / rdirxs
    
    txmin_neg = np.where(txmin0 < 0, True, False)
    txmax_neg = np.where(txmax0 < 0, True, False)
    
    txmin = np.where(x0_true, np.inf, txmin)
    txmin_ntrue = np.logical_and(x0_true, txmin_neg)
    txmin = np.where(txmin_ntrue, -np.inf, txmin)
    txmax = np.where(x0_true, np.inf, txmax)
    txmax_ntrue = np.logical_and(x0_true, txmax_neg)
    txmax = np.where(txmax_ntrue, -np.inf, txmax)
    
    txmin = np.around(txmin, decimals=precision)
    txmax = np.around(txmax, decimals=precision)
    
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(txmin1 > txmax1, txmax1 , txmin1)
    txmax = np.where(txmin1 > txmax1, txmin1 , txmax1)
    
    rdirys = rays_dirsT[1]
    y0_true = np.where(rdirys == np.inf, True, False)
    tymin0 = bboxT[1] - rays_origsT[1] 
    tymin = tymin0 / rdirys
    tymax0 = bboxT[4] - rays_origsT[1]  
    tymax = tymax0 / rdirys
        
    tymin_neg = np.where(tymin0 < 0, True, False)
    tymax_neg = np.where(tymax0 < 0, True, False)
    
    tymin = np.where(y0_true, np.inf, tymin)
    tymin_ntrue = np.logical_and(y0_true, tymin_neg)
    tymin = np.where(tymin_ntrue, -np.inf, tymin)
    tymax = np.where(y0_true, np.inf, tymax)
    tymax_ntrue = np.logical_and(y0_true, tymax_neg)
    tymax = np.where(tymax_ntrue, -np.inf, tymax)
    
    tymin = np.around(tymin, decimals=precision)
    tymax = np.around(tymax, decimals=precision)
    tymin1 = tymin
    tymax1 = tymax
    tymin = np.where(tymin1 > tymax1, tymax1, tymin1)
    tymax = np.where(tymin1 > tymax1, tymin1, tymax1)
    #============================================
    #make the first judgement if there is any collisions
    
    condxy = np.logical_or(txmin > tymax, tymin > txmax)
    condxy = np.logical_not(condxy)
    condxy_index = np.where(condxy)[0]
    if len(condxy_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    swapx_true1 = np.logical_and(condxy, tymin > txmin)
    swapx_true2 = np.logical_and(condxy, tymax < txmax)
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(swapx_true1, tymin , txmin1)
    txmax = np.where(swapx_true2, tymax , txmax1)
    
    #============================================
    # calcuate the z-dim
    rdirzs = rays_dirsT[2]
    z0_true = np.where(rdirzs == np.inf, True, False)
    tzmin0 = bboxT[2] - rays_origsT[2] 
    tzmin = tzmin0 / rdirzs
    tzmax0  = bboxT[5] - rays_origsT[2]
    tzmax = tzmax0 / rdirzs
    
    tzmin_neg = np.where(tzmin0 < 0, True, False)
    tzmax_neg = np.where(tzmax0 < 0, True, False)
    
    tzmin = np.where(z0_true, np.inf, tzmin)
    tzmin_ntrue = np.logical_and(z0_true, tzmin_neg)
    tzmin = np.where(tzmin_ntrue, -np.inf, tzmin)
    tzmax = np.where(z0_true, np.inf, tzmax)
    tzmax_ntrue = np.logical_and(z0_true, tzmax_neg)
    tzmax = np.where(tzmax_ntrue, -np.inf, tzmax)
    
    tzmin = np.around(tzmin, decimals=precision)
    tzmax = np.around(tzmax, decimals=precision)
    tzmin1 = tzmin
    tzmax1 = tzmax
    tzmin = np.where(tzmin1 > tzmax1, tzmax1 , tzmin1)
    tzmax = np.where(tzmin1 > tzmax1, tzmin1 , tzmax1)
    #============================================
    # make judgement on the zdim
    #make the first judgement if there is any collisions
    condxyz = np.logical_or(txmin > tzmax, tzmin > txmax)
    condxyz = np.logical_not(condxyz)
    condxyz = np.logical_and(condxy, condxyz)
    condxyz_index = np.where(condxyz)[0]
    if len(condxyz_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    swapz_true1 = np.logical_and(condxyz, tzmin > txmin)
    swapz_true2 = np.logical_and(condxyz, tzmax < txmax)
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(swapz_true1, tzmin , txmin1)
    txmax = np.where(swapz_true2, tzmax , txmax1)
    
    tx_min_true = np.logical_not(txmin > 0)
    tx_min_true = np.logical_and(tx_min_true, condxyz)
    txmin = np.where(tx_min_true, txmax, txmin)
    #check again if the magnitude and the max is positive
    tx_min_true = np.logical_not(txmin < 0)
    tx_min_true = np.logical_and(tx_min_true, condxyz)
    tx_min_true_index = np.where(tx_min_true)[0]
    if len(tx_min_true_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    mags = np.where(tx_min_true, txmin, np.zeros(len(txmin)))
    # print('mags',mags)
    mags_reshape = np.reshape(mags,[len(mags),1])
    dir_mags = rays_dirs*mags_reshape
    intPts = rays_origs + dir_mags
    
    mags = np.where(tx_min_true, mags, np.full([len(mags)], np.inf))
    mags_ray_reshape = np.reshape(mags, (nrays,nbbox))
    mags_true = np.logical_not(mags_ray_reshape==np.inf)
    mags_true_index = np.where(mags_true)
    
    intPts_reshape = np.reshape(intPts, (nrays,nbbox,3))
    res_pts = intPts_reshape[mags_true_index]
    res_mag = mags_ray_reshape[mags_true_index]
    res_ray_indx = mags_true_index[0]
    res_bbox_indx = mags_true_index[1]

    res_pts, res_mag, res_ray_indx, res_bbox_indx= acct4obstruction(res_pts, 
                                                                   res_mag, 
                                                                   res_ray_indx, 
                                                                   res_bbox_indx)
    
    return res_pts, res_mag, res_ray_indx, res_bbox_indx

def rays_faces_intersection(ray_list, face_list):
    """
    This function intersect multiple rays with multiple faces
 
    Parameters
    ----------
    ray_list : list of rays
        array of ray objects
        
    face_list : lsit of faces
        array of face objects
        
    Returns
    -------
    hit_rays : array of rays
        rays that hit faces with new attributes documenting the faces and intersections. {'rays_faces_intersection': {'intersection':[], 'hit_face':[]}}
        
    miss_rays : array of rays
        rays that did not hit any faces.
        
    hit_face : array of faces
        faces that are hit with new attributes documenting the intersection pt and the rays that hit it. {'rays_faces_intersection': {'intersection':[], 'ray':[]}}
        
    miss_faces : array of faces
        faces that are not hit by any rays.
    """
    #extract the origin and dir of the rays 
    rays_xyz = np.array([[r.origin,r.dirx] for r in ray_list])
    #need to first triangulate the faces
    tris_xyz = np.array([])
    ntris = 0
    tidx_faces = []
    for f in face_list:
        verts_indx = modify.triangulate_face(f, indices = True)
        verts = verts_indx[0]
        indx = verts_indx[1]
        c_verts = np.take(verts, indx, axis=0)
        tris_xyz = np.append(tris_xyz, c_verts)
        ntri_aface = len(indx)
        tidx_faces.append(np.arange(ntri_aface) + ntris)
        ntris += ntri_aface
    
    #reshape the tris
    tris_xyz = np.reshape(tris_xyz, (ntris, 3, 3))
    res_pts, res_mag, res_ray_indx, res_tri_indx = rays_xyz_tris_intersect(rays_xyz, tris_xyz)
    if len(res_pts) != 0:
        #now i need to sort it into faces
        hit_faces = []
        miss_faces = []
        for cnt, tidx_face in enumerate(tidx_faces):
            #if the triidx_face are in the hit tri indx, this face is hit by a ray
            hit_true = np.in1d(res_tri_indx, tidx_face)
            hit_indx = np.where(hit_true)[0]
            nhits = len(hit_indx)
            if nhits == 0:
                #this face is not hit by any rays
                miss_faces.append(face_list[cnt])
            elif nhits == 1:
                #this face is hit by one of the ray
                #find the ray and put it in its attributes
                face_ray = ray_list[res_ray_indx[hit_indx[0]]]
                intersect = res_pts[hit_indx[0]]
                hit_face = face_list[cnt]
                hf_attribs = hit_face.attributes
                if 'rays_faces_intersection' in hf_attribs:
                    ht_rf_att = hf_attribs['rays_faces_intersection']
                    ht_rf_att['intersection'].append(intersect)
                    ht_rf_att['ray'].append(face_ray)
                else:
                    attribs = {'rays_faces_intersection':
                                   { 'intersection': [intersect],
                                    'ray': [face_ray]
                                    }
                               }
                    hit_face.update_attributes(attribs)
                hit_faces.append(hit_face)
                
                #check if the attribs alr exist in the rays
                rattribs = face_ray.attributes
                if 'rays_faces_intersection' in rattribs:
                    rays_rf_att = rattribs['rays_faces_intersection']
                    rays_rf_att['intersection'].append(intersect)
                    rays_rf_att['hit_face'].append(hit_face)
                else:
                    rattribs = {'rays_faces_intersection':
                                    {'intersection': [intersect],
                                     'hit_face': [hit_face]}
                                }
                    face_ray.update_attributes(rattribs)
                # print('1_hit')
            elif nhits > 1:
                #first take a look at the ray index
                #find the rays that are hitting this face
                pts_face = np.take(res_pts, hit_indx, axis=0)
                mags_face = np.take(res_mag, hit_indx, axis=0)
                raysIds_face = np.take(res_ray_indx, hit_indx, axis=0)
                #check if they are instances where a single ray is hitting the same surfaces >1
                idDups = utility.id_dup_indices_1dlist(raysIds_face)
                all_indx = np.arange(len(raysIds_face))
                if len(idDups) > 0:
                    # a single ray hits a face more than once
                    # the ray is hitting the face at its triangulated seams    
                    idDups_flat = list(chain(*idDups))
                    nonDup_idxs = utility.find_xs_not_in_ys(all_indx, idDups_flat)
                    # double check if hitting at seams is true
                    for idDup in idDups:
                        mag = mags_face[idDup]
                        id_magDup = utility.id_dup_indices_1dlist(mag)
                        if len(id_magDup) == 1:    
                            if len(id_magDup[0]) == len(mag):
                                #they are hitting the same spot
                                nonDup_idxs = np.append(nonDup_idxs, idDup[0])
                            else:
                                print('SOMETHING IS WRONG WITH THIS SURFACE')
                        else:
                            print('SOMETHING IS WRONG WITH THIS SURFACE')
                            
                else:
                    nonDup_idxs = all_indx
                
                nonDup_idxs.sort()
                raysIds_face = np.take(raysIds_face, nonDup_idxs, axis=0)
                face_rays = np.take(ray_list, raysIds_face, axis=0)
                face_rays = face_rays.tolist()
                intersects = np.take(pts_face, nonDup_idxs, axis=0)
                intersects = intersects.tolist()
                hit_face = face_list[cnt]
                hf_attribs = hit_face.attributes
                
                #need to check is it hit by different rays
                if 'rays_faces_intersection' in hf_attribs:
                    ht_rf_att = hf_attribs['rays_faces_intersection']
                    ht_rf_att['intersection'].extend(intersects)
                    ht_rf_att['ray'].extend(face_rays)
                else:
                    attribs = {'rays_faces_intersection':
                                   { 'intersection': intersects,
                                    'ray': face_rays
                                    }
                               }
                    hit_face.update_attributes(attribs)
                    
                hit_faces.append(hit_face)
                
                #check if the attribs alr exist in the rays
                for fcnt,face_ray in enumerate(face_rays):
                    rattribs = face_ray.attributes
                    if 'rays_faces_intersection' in rattribs:
                        rays_rf_att = rattribs['rays_faces_intersection']
                        rays_rf_att['intersection'].append(intersects[fcnt])
                        rays_rf_att['hit_face'].append(hit_face)
                    else:
                        rattribs = {'rays_faces_intersection':
                                        {'intersection': [intersects[fcnt]],
                                         'hit_face': [hit_face]}
                                    }
                        face_ray.update_attributes(rattribs)
                # print('multiple hits')
        hit_rays = []
        miss_rays = []
        uq_hit_idx = np.unique(res_ray_indx)
        hit_rays = np.take(ray_list, uq_hit_idx , axis=0)
        rindx = np.arange(len(ray_list))
        miss_true = np.in1d(rindx, uq_hit_idx)
        miss_true = np.logical_not(miss_true)
        miss_idx = np.where(miss_true)[0]
        miss_rays = np.take(ray_list, miss_idx, axis=0)
        return hit_rays, miss_rays, hit_faces, miss_faces
    else:
        return [], ray_list, [], face_list

def rays_bboxes_intersect(ray_list, bbox_list):
    """
    This function intersect multiple rays with multiple bboxes
 
    Parameters
    ----------
    ray_list : list of rays
        array of ray objects
        
    bbox_list : a list of bbox object
        A list of bbox
        
    Returns
    -------
    hit_rays : array of rays
        rays that hit faces with new attributes documenting the faces and intersections. {'rays_bboxes_intersection': {'intersection':[], 'hit_bbox':[]}}
        
    miss_rays : array of rays
        rays that did not hit .
        
    hit_bboxes : array of dictionary
        bboxes that are hit with new attributes documenting the intersection pt and the rays that hit it. {'rays_bboxes_intersection': {'intersection':[], 'ray':[]}}
        
    miss_faces : array of bbox
        bboxes that are not hit by any rays.
    """
    #extract the origin and dir of the rays
    rays_xyz = np.array([[r.origin,r.dirx] for r in ray_list])
    res_pts, res_mag, res_ray_indx, res_bbox_indx = rays_xyz_bboxes_intersect(rays_xyz, bbox_list)
    # print(res_pts, res_mag, res_ray_indx, res_bbox_indx)
    if len(res_pts) != 0:
        #now i need to sort it into bboxes
        hit_bboxes = []
        miss_bboxes = []
        for cnt,bbox in enumerate(bbox_list):
            cnt_ls = np.array(cnt)
            hit_true = np.in1d(res_bbox_indx, cnt_ls)
            hit_indx = np.where(hit_true)[0]
            nhits = len(hit_indx)
            if nhits == 0:
                #this bbox is not hit by any rays
                miss_bboxes.append(bbox)
            if nhits == 1:
                # print('1_hit')
                #this face is hit by one of the ray
                #find the ray and put it in its attributes
                bbox_ray = ray_list[res_ray_indx[hit_indx[0]]]
                intersect = res_pts[hit_indx[0]]
                hb_attribs = bbox.attributes
                if 'rays_bboxes_intersection' in hb_attribs:
                    ht_rf_att = hb_attribs['rays_bboxes_intersection']
                    ht_rf_att['intersection'].append(intersect)
                    ht_rf_att['ray'].append(bbox_ray)
                else:
                    attribs = {'rays_bboxes_intersection':
                                    { 'intersection': [intersect],
                                    'ray': [bbox_ray]
                                    }
                                }
                    bbox.update_attributes(attribs)
                hit_bboxes.append(bbox)
    
                #check if the attribs alr exist in the rays
                rattribs = bbox_ray.attributes
                if 'rays_bboxes_intersection' in rattribs:
                    rays_rf_att = rattribs['rays_bboxes_intersection']
                    rays_rf_att['intersection'].append(intersect)
                    rays_rf_att['hit_bbox'].append(bbox)
                else:
                    rattribs = {'rays_bboxes_intersection':
                                    {'intersection': [intersect],
                                      'hit_bbox': [bbox]}
                                }
                    bbox_ray.update_attributes(rattribs)
            elif nhits > 1:
                # print('multiple hits')
                #first take a look at the ray index
                #find the rays that are hitting this bbox
                pts_bbox = np.take(res_pts, hit_indx, axis=0)
                raysIds_bbox = np.take(res_ray_indx, hit_indx, axis=0)
                #check if they are instances where a single ray is hitting the same surfaces >1
                idDups = utility.id_dup_indices_1dlist(raysIds_bbox)
                all_indx = np.arange(len(raysIds_bbox))
                if len(idDups) > 0:
                    print('SOMETHING IS WRONG A RAY SHOULD NOT HIT THE BBOX TWICE')
                else:
                    nonDup_idxs = all_indx
                
                nonDup_idxs.sort()
                raysIds_bbox = np.take(raysIds_bbox, nonDup_idxs, axis=0)
                bbox_rays = np.take(ray_list, raysIds_bbox, axis=0)
                bbox_rays = bbox_rays.tolist()
                intersects = np.take(pts_bbox, nonDup_idxs, axis=0)
                intersects = intersects.tolist()
                hb_attribs = bbox.attributes
                
                #need to check is it hit by different rays before
                if 'rays_bboxes_intersection' in hb_attribs:
                    ht_rf_att = hb_attribs['rays_bboxes_intersection']
                    ht_rf_att['intersection'].extend(intersects)
                    ht_rf_att['ray'].extend(bbox_rays)
                else:
                    attribs = {'rays_bboxes_intersection':
                                    { 'intersection': intersects,
                                    'ray': bbox_rays
                                    }
                                }
                    bbox.update_attributes(attribs)
                    
                hit_bboxes.append(bbox)
                
                #check if the attribs alr exist in the rays
                for fcnt,bbox_ray in enumerate(bbox_rays):
                    rattribs = bbox_ray.attributes
                    if 'rays_bboxes_intersection' in rattribs:
                        rays_rf_att = rattribs['rays_bboxes_intersection']
                        rays_rf_att['intersection'].append(intersects[fcnt])
                        rays_rf_att['hit_bbox'].append(bbox)
                    else:
                        rattribs = {'rays_bboxes_intersection':
                                        {'intersection': [intersects[fcnt]],
                                          'hit_bbox': [bbox]}
                                    }
                        bbox_ray.update_attributes(rattribs)
                
        hit_rays = []
        miss_rays = []
        uq_hit_idx = np.unique(res_ray_indx)
        hit_rays = np.take(ray_list, uq_hit_idx , axis=0)
        rindx = np.arange(len(ray_list))
        miss_true = np.in1d(rindx, uq_hit_idx)
        miss_true = np.logical_not(miss_true)
        miss_idx = np.where(miss_true)[0]
        miss_rays = np.take(ray_list, miss_idx, axis=0)
        return hit_rays, miss_rays, hit_bboxes, miss_bboxes
    
    else:
        return [], ray_list, [], bbox_list

def face_area(face):
    """
    Calculates the area of the face
 
    Parameters
    ----------    
    face : the face object
        the face object.
    
    Returns
    -------
    area : float
        the area of the face.
    """
    def unit_normal(a, b, c):
        x = np.linalg.det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
        y = np.linalg.det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
        z = np.linalg.det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
        magnitude = (x**2 + y**2 + z**2)**.5
        return (x/magnitude, y/magnitude, z/magnitude)
    
    if face.surface_type == geom.SrfType.POLYGON:
        verts = get.vertices_frm_face(face)
        nverts = len(verts)
        if nverts < 3: #not a plane no area
            return 0
        total = [0, 0, 0]
        for i in range(nverts):
            vi1 = verts[i]
            vi1 = vi1.point.xyz
            vi2 = verts[(i+1) % nverts]
            vi2 = vi2.point.xyz
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
            
        result = np.dot(total, unit_normal(verts[0].point.xyz, verts[1].point.xyz, verts[2].point.xyz))
        return abs(result/2)
    
    elif face.surface_type == geom.SrfType.BSPLINE:
        print('Area of face with Bspline surface not yet implemented, convert bspline to polygon surface')

def linexyzs_from_t(ts, linexyzs):
    """
    get a pointxyz on a line with the parameter t. Based on this post https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    Parameters
    ----------
    ts : array of float
        the t parameter, 0-1.
    
    linexyzs : ndarray
        array of lines [line1, line2, ...]. Each line is define as [[x1,y1,z1],[x2,y2,z2]].
        
    Returns
    -------
    xyzs : ndarray
        array of the xyz points on the line with parameter t.
    """
    if type(ts) != np.ndarray:
        ts = np.array(ts)
        
    if type(linexyzs) != np.ndarray:
        linexyzs = np.array(linexyzs)
    
    shape = np.shape(linexyzs)
    xyzs= np.reshape(linexyzs, (shape[0]*shape[1], shape[2]))
    xyzsT = xyzs.T
    xs1 = xyzsT[0][0::2]
    xs2 = xyzsT[0][1::2]
    
    ys1 = xyzsT[1][0::2]
    ys2 = xyzsT[1][1::2]
    
    zs1 = xyzsT[2][0::2]
    zs2 = xyzsT[2][1::2]
    
    xs_t = xs1 + ((xs2-xs1) * ts)
    ys_t = ys1 + ((ys2-ys1) * ts)
    zs_t = zs1 + ((zs2-zs1) * ts)
    
    xyzsT2 = np.array([xs_t, ys_t, zs_t])
    xyzs_t = xyzsT2.T
    
    return xyzs_t
    
def dist_pointxyzs2linexyzs(pointxyzs, linexyzs, int_pts = False):
    """
    Find the distance between the points and the lines. If negative is to the xxx if positive is to the yyy. Based on this post https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    Parameters
    ----------
    pointxyzs : ndarray
        array of points [point1, point2, ...]. Each point is define as [x,y,z].
    
    linexyzs : ndarray
        array of edges [edge1, edge2, ...]. Each edge is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
    
    int_pts: bool
        if true will return the closest point on the line to the point. default == False
        
    Returns
    -------
    distances : ndarray
        array of all the distances.
    
    closest_pts : ndarray, optional
        the closest points on the line to the point. [point1, point2, pointx]. Each point is [x,y,z].
    """
    if type(pointxyzs) != np.ndarray:
        pointxyzs = np.array(pointxyzs)
        
    if type(linexyzs) != np.ndarray:
        linexyzs = np.array(linexyzs)
    
    shape = np.shape(linexyzs)
    xyzs = np.reshape(linexyzs, (shape[0]*shape[1], shape[2]))
    xyzs1 = xyzs[0::2]
    xyzs2 = xyzs[1::2]
    
    a = xyzs1 - pointxyzs
    b = xyzs2 - xyzs1
    c = dot_product(a, b)
    
    dist = abs(dist_btw_xyzs(xyzs2, xyzs1))
    distsq = np.square(dist)
    ts = c/distsq
    ts = ts*-1
    ts = np.where(ts < 0, 0, ts)
    ts = np.where(ts > 1, 1, ts)
    int_xyzs = linexyzs_from_t(ts, linexyzs)
    dists = dist_btw_xyzs(pointxyzs, int_xyzs)
    if int_pts == False:
        return dists
    else:
        return dists, int_xyzs

def _extract_xyzs_from_lineedge(edge_list):
    """
    extract line xyzs from edges and throw an error if the curve is not a line. A line is a curve defined by only two points
    
    Parameters
    ----------
    edge_list : ndarray
        array of edges [edge1, edge2, ...].
    
    Returns
    -------
    xyzs : ndarray
        array of all the points.
    """
    linexyzs = []
    for e in edge_list:
        vs = get.vertices_frm_edge(e)
        if len(vs) == 2:
            xyzs = [v.point.xyz for v in vs]
            linexyzs.append(xyzs)
        else:
            raise ValueError('The curve in the edge needs to be a line defined by 2 vertices')
            
    return linexyzs
    
def dist_vertex2line_edge(vertex_list, edge_list, int_pts = False):
    """
    Find the distance between the vertices to the edges. The edges cannot have curve that has more than 2 points. Only work with lines.
    
    Parameters
    ----------
    vertex_list : list of vertex
        list of vertex.
    
    edge_list : list of edges
        list of edges. The edges can only have line as geometry. That is polygon curve that is define by only two vertices.
    
    int_pts: bool
        if true will return the closest point on the line to the point. default == False
        
    Returns
    -------
    distances : ndarray
        array of all the distances.
    
    closest_verts : list of verts, optional
        the closest vertices on the line to the point.
    """
    pointxyzs = [v.point.xyz for v in vertex_list]
    
    linexyzs = _extract_xyzs_from_lineedge(edge_list)
    #TODO implement for more complex polygon curve can be achieve by breaking each curve into lines
    
    if int_pts == False:
        dists = dist_pointxyzs2linexyzs(pointxyzs, linexyzs, int_pts = False)
        return dists
    else:
        dists, int_pts = dist_pointxyzs2linexyzs(pointxyzs, linexyzs, int_pts = True)
        vlist = create.vertex_list(int_pts)
        return dists, vlist
    
def linexyzs_intersect(linexyzs1, linexyzs2):
    """
    Find the intersections between the lines.
    
    Parameters
    ----------
    linexyzs1 : ndarray
        array of lines [line1, line2, ...]. Each line is define as [[x1,y1,z1],[x2,y2,z2]].
    
    linexyzs2 : ndarray
        array of lines [line1, line2, ...]. Each line is define as [[x1,y1,z1],[x2,y2,z2]].
        
    Returns
    -------
    intersections : ndarray
        array of all the intersection points.
    """
    if type(linexyzs1) != np.ndarray:
        linexyzs1 = np.array(linexyzs1)
    
    if type(linexyzs2) != np.ndarray:
        linexyzs2 = np.array(linexyzs2)
    
    shape1 = np.shape(linexyzs1)
    shape2 = np.shape(linexyzs2)
    
    if shape1 != shape2:
        raise ValueError("Both linexyzs have to be in the same shape")
    
    xyzs1 = np.reshape(linexyzs1, (shape1[0]*shape1[1], shape1[2]))
    xyzs1_1 = xyzs1[0::2]
    xyzs1_2 = xyzs1[1::2]
    
    xyzs2 = np.reshape(linexyzs2, (shape2[0]*shape2[1], shape2[2]))
    xyzs2_1 = xyzs2[0::2]
    xyzs2_2 = xyzs2[1::2]
    
    vector_ab = xyzs2_1 - xyzs1_1
    linexyzs1_dir = xyzs1_2  - xyzs1_1 
    linexyzs2_dir = xyzs2_2  - xyzs2_1 
    vector_perpendicular = cross_product(linexyzs1_dir, linexyzs2_dir)
    
    num = cross_product(vector_ab, linexyzs2_dir)
    num = dot_product(num, vector_perpendicular)
    naxis = len(np.shape(vector_perpendicular))
    denom = np.linalg.norm(vector_perpendicular, ord = 2, axis = naxis-1, keepdims=False)
    denom = denom**2
    cond1 = np.logical_not(num==0)
    num = np.where(cond1, num, np.nan)
    denom = np.where(cond1, denom, np.nan)
    ts = np.where(cond1, num/denom, np.nan)
    shape3 = np.shape(linexyzs1_dir)
    ts = np.reshape(ts, (shape3[0],1))
    cond2 = np.logical_and(ts >= 0, ts <= 1)
    vector_a_scaled = linexyzs1_dir * ts
    int_pts = xyzs1_1 + vector_a_scaled
    int_pts = np.where(cond2, int_pts, np.nan)
    coplanar_xyzs = np.append(linexyzs1, linexyzs2, axis=1)
    is_planar = is_coplanar_xyzs(coplanar_xyzs)
    is_planar = np.reshape(is_planar, (shape3[0],1))
    int_pts = np.where(is_planar, int_pts, np.nan)
    
    dist = dist_pointxyzs2linexyzs(int_pts, linexyzs2, int_pts = False)
    is_on_line2 = np.logical_not(dist!=0)
    is_on_line2 = np.reshape(is_on_line2, (shape3[0],1))
    int_pts = np.where(is_on_line2, int_pts, np.nan)

    return int_pts

def lineedge_intersect(edge_list1, edge_list2):
    """
    Find the intersections between the edge_list1 and edge_list2. The edges need to only have simple lines as curve geometry
    
    Parameters
    ----------
    edge_list1 : ndarray
        array of edges [edge1, edge2, ...].
    
    edge_list2 : ndarray
        array of edges [edge1, edge2, ...].
        
    Returns
    -------
    intersections : ndarray
        array of all the intersection points.
    """
    linexyzs1 = _extract_xyzs_from_lineedge(edge_list1)
    linexyzs2 = _extract_xyzs_from_lineedge(edge_list2)
    #TODO implement for more complex polygon curve can be achieve by breaking each curve into lines
    int_pts = linexyzs_intersect(linexyzs1, linexyzs2)
    vlist = create.vertex_list(int_pts)
    return vlist
    
def polyxyzs_intersections(clipping_polyxyzs, subject_polyxyzs, ref_vecs):
    """
    Find the intersections between the clipping polys and the subject polys. The clipping polygons must be convex. Both the polygons cannot have holes.
    
    Parameters
    ----------
    clipping_polys : ndarray
        array of polygons [clipping_poly1, clipping_poly2, ...]. Each clipping_poly is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
    
    subject_polys : ndarray
        array of polygons [subject_poly1, subject_poly2, ...]. Each subject_poly is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
    
    ref_vecs : ndarray
        The reference vector must be perpendicular to the list of points. Array of vectors. A vec is a tuple that documents the xyz direction of a vector e.g. (x,y,z)
        
    Returns
    -------
    intersections : ndarray
        array of all the intersection points.
    """
    shape1 = np.shape(clipping_polyxyzs)
    shape2 = np.shape(subject_polyxyzs)
    
    nshape1 = len(shape1)
    nshape2 = len(shape2)
    
    if nshape1 == 1 or nshape2 == 1:
        raise ValueError("this is a 1d array. This sometimes happens if arrays have arrays of different length, this is not allowed")
    
    if nshape1 != nshape2:
        raise ValueError("Both polyxyzs need to have the same dimension")
    
    if type(clipping_polyxyzs) != np.ndarray:
        clipping_polyxyzs = np.array(clipping_polyxyzs)
        
    if type(subject_polyxyzs) != np.ndarray:
        subject_polyxyzs = np.array(subject_polyxyzs)
    
    #create edge for the polygons
    clipping_polyxyzs = np.repeat(clipping_polyxyzs, 2, axis=1)
    #roll it so the points are arrange in edges
    clipping_polyxyzs = np.roll(clipping_polyxyzs, -1, axis=1)
    #reshape the arrays into edges
    clipping_polyxyzs = np.reshape(clipping_polyxyzs, (shape1[0],shape1[1],2,shape1[2]))
    #for the edges
    clipping_polyxyzs_edges = np.repeat(clipping_polyxyzs, shape2[1], axis=1)
    clip_edge_shape1 = np.shape(clipping_polyxyzs_edges)
    clipping_polyxyzs_edges = np.reshape(clipping_polyxyzs_edges, (clip_edge_shape1[0]*clip_edge_shape1[1], clip_edge_shape1[2], clip_edge_shape1[3]))
    #for testing ccw
    #repeat the edges according to the number of edges the subject polygons have. We are setting up for a double for loop
    clipping_polyxyzs = np.repeat(clipping_polyxyzs, shape2[1]*2, axis=1)
    shape1_2 = np.shape(clipping_polyxyzs)
    #repeat the edges according to the number of edges the clipping polygons have. We are setting up for a double for loop
    subject_polyxyzs = np.repeat(subject_polyxyzs, 2, axis=1)
    subject_polyxyzs = np.roll(subject_polyxyzs, -1, axis=1)
    subject_polyxyzs = np.reshape(subject_polyxyzs, (shape2[0],shape2[1],2,shape2[2]))
    subject_polyxyzs = np.repeat(subject_polyxyzs, shape1[1], axis=0)
    #flatten the array to the level of the edges
    subject_edge_shape1 = np.shape(subject_polyxyzs)
    subject_polyxyzs_edges = np.reshape(subject_polyxyzs, (subject_edge_shape1[0]*subject_edge_shape1[1], subject_edge_shape1[2], subject_edge_shape1[3]))
    #for testing ccw
    subject_polyxyzs = np.reshape(subject_polyxyzs, (shape1_2[0], shape1_2[1], 1, shape1_2[3]))
    
    for_ccw = np.append(clipping_polyxyzs, subject_polyxyzs, axis=2)
    shape1_3 = np.shape(for_ccw)
    for_ccw = np.reshape(for_ccw, (shape1_3[0]*shape1_3[1], shape1_3[2], shape1_3[3]))
    ref_vecs = np.repeat(ref_vecs, shape1_3[1], axis=0)
    is_ccw = is_anticlockwise(for_ccw, ref_vecs)
    # reshape the result to reflect both the 1st point and 2nd point of the subject polygon
    is_ccw = np.reshape(is_ccw, (subject_edge_shape1[0]*subject_edge_shape1[1],2))
    is_ccwT = is_ccw.T
    # as long as either points from the subject polygon is ccw to the edge, we will perform an intersection to check
    cond1 = np.logical_or(is_ccwT[0], is_ccwT[1]) 
    cond1 = np.reshape(cond1, (len(cond1),1)) 
    int_xyzs = np.where(cond1, linexyzs_intersect(clipping_polyxyzs_edges, subject_polyxyzs_edges), np.nan)
    # if both subject polygon points is ccw, keep only the second point
    cond2 = np.logical_and(is_ccwT[0], is_ccwT[1]) 
    cond2 = np.reshape(cond2, (len(cond1),1))
    print(is_ccw)
    print(np.reshape(cond2, (int(len(cond1)/4), 4, 1)))
    
    if nshape1 == 2:  # only 1 set of polygons each
        pass
    
    elif nshape1 == 3:
        pass

def polygons_intersections(clipping_polys, subject_polys):
    """
    Find the intersections between the clipping polys and the subject polys
    
    Parameters
    ----------
    clipping_polys : ndarray
        array of polygons [clipping_poly1, clipping_poly2, ...]. Each clipping_poly is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
    
    subject_polys : ndarray
        array of polygons [subject_poly1, subject_poly2, ...]. Each subject_poly is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
        
    Returns
    -------
    intersections : ndarray
        array of all the intersection points.
    """
    pass

def grp_faces_on_nrml(face_list: list[topobj.Face], return_idx: bool = False, decimals: int = None) -> tuple[list[list[topobj.Face]], list[topobj.Face]]:
    """
    Group the faces based on the normal of the faces
    
    Parameters
    ----------
    face_list: list[topobj.Face]
        group these faces
    
    return_idx: bool, optional
        only return the indices of the grouped faces. Default to False

    decimals: int, optional
        the number of decimals to round the normals of the faces. Default to 6 

    Returns
    -------
    res_face : tuple[list[list[topobj.Face]], list[topobj.Face]]
        list[Shape[Any,Any]] of grouped faces. list[Shape[Any]] of faces that do not belong to any group.
        
    """
    # get the normals of each tri face
    nrml_ls = np.array([get.face_normal(f) for f in face_list])
    if decimals != None:
        nrml_ls = np.round(nrml_ls, decimals=decimals)
    uniq_nrml = np.unique(nrml_ls, axis=0, return_inverse = True)
    idx = utility.separate_dup_non_dup(uniq_nrml[1])
    non_dup_idx = idx[0]
    dup_idx = idx[1]

    if return_idx == False:
        indv_faces = []
        g_faces = []
        if len(non_dup_idx) !=0:
            indv_faces = np.take(face_list,non_dup_idx)
        if len(dup_idx) != 0:
            g_faces = [np.take(face_list, idx, axis=0) for idx in dup_idx]
            
        return g_faces, indv_faces
    else:
        return dup_idx, non_dup_idx

def find_faces_outline(face_list: list[topobj.Face]) -> tuple[list[topobj.Edge], list[topobj.Edge]]:
    """
    Find non-duplicated edges from a list of faces. Can be used to find the outline of a triangulated surface.
    
    Parameters
    ----------
    face_list: list[topobj.Face]
        find non-duplicated edges of these faces.
    
    Returns
    -------
    non_dup_edges : list[topobj.Edge]
        list of non duplicated edges
    
    dup_edges : list[topobj.Edge]
        list[Shape(Any, Any)] of duplicated edges
    """
    edge_ls = [get.edges_frm_face(f) for f in face_list]
    edge_ls = list(chain(*edge_ls))
    edge_ls = modify.edges2lineedges(edge_ls)
    non_dup_edges, dup_edges = find_non_dup_lineedges(edge_ls)
    return non_dup_edges, dup_edges

def find_non_dup_lineedges(edge_list: list[topobj.Edge]) -> list[topobj.Edge]:
    """
    Find edges that are not duplicated.
    
    Parameters
    ----------
    edge_list: list[topobj.Edge]
        find non duplicated edges from these edges.
    
    Returns
    -------
    non_dup_edges : list[topobj.Edge]
        list of non duplicated edges
    
    dup_edges : list[topobj.Edge]
        list[Shape(Any, Any)] of duplicated edges
    """
    edge_vert_list = [get.vertices_frm_edge(e) for e in edge_list]
    edge_xyz_ls = []
    for edge_verts in edge_vert_list:
        edge_xyzs = [v.point.xyz for v in edge_verts]
        edge_xyz_ls.append(edge_xyzs)

    edge_xyz_ls = np.array(edge_xyz_ls)
    non_dup_idx, dup_idx = find_non_dup_lineedges_xyz(edge_xyz_ls)
    non_dup_edges = np.take(edge_list, non_dup_idx).tolist()
    dup_edges = np.take(edge_list, dup_idx).tolist()
    return non_dup_edges, dup_edges

def find_non_dup_lineedges_xyz(edge_xyz_list: np.ndarray, index: bool = True) -> np.ndarray:
    """
    Find edges that are not duplicated.
    
    Parameters
    ----------
    edge_xyz_list: np.ndarray
        np.ndarray(Shape[Any, 2, 3]) find non duplicated edges from these edges.
    
    index: bool, optional
        Default = True, if True return the indices of the edges that are non-duplicated. False returns the xyzs. 
    
    Returns
    -------
    non_dup_edge_xyzs : np.ndarray
        np.ndarray(Shape[Any, 2]) of non duplicated edges
    
    indices : np.ndarray
        ndarray of the indices [[non_dup_idx], [[dup_idx1], [dup_idx2] ... [dup_idxn]]].
    """
    if type(edge_xyz_list) != np.ndarray:
        np.array(edge_xyz_list)

    arr_shape = np.shape(edge_xyz_list)
    if arr_shape[1] != 2:
        raise ValueError('This is not a line edge !! The edge has more than two vertices')
    else:
        xyz_list = np.reshape(edge_xyz_list, (arr_shape[0]*arr_shape[1], 3)) # flatten the list
        uniq_xyzs = np.unique(xyz_list, axis=0, return_inverse = True) # index the vertices
        edge_idxs = np.reshape(uniq_xyzs[1], (arr_shape[0], arr_shape[1])) # reshape it back to the edge shape
        edge_idxs = np.sort(edge_idxs) # sort the edge indices 
        uniq_edge_idxs = np.unique(edge_idxs, axis=0, return_inverse=True) # compare the edges and identify duplicates
        non_dup_idx, dup_idx = utility.separate_dup_non_dup(uniq_edge_idxs[1]) # separate the dup from non dups
        if index == True:
            return non_dup_idx, dup_idx
        else:
            non_dup = np.take(edge_xyz_list, non_dup_idx)
            return non_dup

def a_connected_path_from_edges(edge_list: list[topobj.Edge], indx: bool = False) -> dict:
    """
    loop through the edges and find a path if the edges are connected. If there are multiple connected paths or branches in the edge list, branches are ignored. Only a single path is returned.
 
    Parameters
    ----------
    edge_list : list[topobj.Edge]
        the list of edge object to check if they are connected.
    
    indx : bool, optional
        return the index instead of the edge object.

    Returns
    -------
    result : dict
        A dictionary {'connected': list[topobj.Edge], 'loose': list[topobj.Edge], 'is_path_closed': bool, 'branch': list[list[int]]}. If indx==True, return the indices instead.
        'is_path_closed': indicate whether if the connected path is close or open. True is closed, False is opened.
        'branches': a list of list of edge index indicating at which edge a branch occur.
    """
    nedges = len(edge_list)
    all_indxs = range(nedges)
    connected_indx_all = []
    branches = []
    prev_cnt_ls = []
    cnt = 0
    while nedges > 0:
        if cnt not in prev_cnt_ls: # if 
            edge_list_dup = edge_list[:]
            del edge_list_dup[cnt]
            res_dict = find_edges_connected2this_edge(edge_list[cnt], edge_list_dup, mode='end_start', indx=True)
            connected_indxs = res_dict['connected']
            n_connections = len(connected_indxs)
            if n_connections != 0:
                if n_connections > 1:
                    branch_indxs = connected_indxs[1:]
                    branch_indxs_calibrated = np.where(branch_indxs >= cnt, branch_indxs+1, branch_indxs)
                    branch = np.insert(branch_indxs_calibrated, 0, cnt).tolist()
                    branches.append(branch)

                if cnt not in connected_indx_all:
                    connected_indx_all.append(cnt)
                first_connection = connected_indxs[0]
                if first_connection >= cnt:
                    first_connection += 1
                connected_indx_all.append(first_connection)
                prev_cnt_ls.append(cnt)
                cnt = first_connection
            else:
                prev_cnt_ls.append(cnt)
                not_in_cnt = utility.find_xs_not_in_ys(all_indxs, prev_cnt_ls).tolist()
                cnt = not_in_cnt[0]

        nedges = nedges-1
    
    is_path_closed = False
    if connected_indx_all[0] == connected_indx_all[-1]:
        is_path_closed = True
        connected_indx_all = connected_indx_all[0:-1]

    loose_indxs = utility.find_xs_not_in_ys(all_indxs, connected_indx_all).tolist()

    if indx == True:
        res = {'connected': connected_indx_all, 'loose': loose_indxs, 'is_path_closed': is_path_closed, 'branches': branches}
    else:
        connected_edges = np.take(edge_list, connected_indx_all).tolist()
        loose_edges = np.take(edge_list, loose_indxs).tolist()
        res = {'connected': connected_edges, 'loose': loose_edges, 'is_path_closed': is_path_closed, 'branches': branches}
    return res

def find_edges_connected2this_edge(this_edge: topobj.Edge, edge_list: list[topobj.Edge], mode: str = 'end_start', indx: bool = False) -> dict:
    """
    find the edges in the edge_list that is connected to this_edge. The algorithm use the end point of this_edge and check it with the start point of the edge_list, directionality of the edges matter.
 
    Parameters
    ----------
    this_edge: topobj.Edge
        the edge object.

    edge_list : list[topobj.Edge]
        the list of edge object to check if they are connected to this_edge.

    mode : str, optional
        the mode to determine the way edges are connected. Default to end_start
        end_start: the end point of this_edge to the start point of the edge_list
        start_end: the start point of this_edge to the end point of the edge_list
        end_end: the end point of this_edge to the end point of the edge_list
        start_start: the start point of this_edge to the start point of the edge_list 

    indx : bool, optional
        return the index instead of the actual edge object.
 
    Returns
    -------
    connected_edges : dict
        A dictionary {'connected': list[topobj.Edge], 'loose': list[topobj.Edge]}. If indx==True, return the indices instead.
    """
    if mode == 'end_start':
        this_xyz = this_edge.end_vertex.point.xyz
        xyz_ls = np.array([e.start_vertex.point.xyz for e in edge_list])
    elif mode == 'start_end':
        this_xyz = this_edge.start_vertex.point.xyz
        xyz_ls = np.array([e.end_vertex.point.xyz for e in edge_list])
    elif mode == 'end_end':
        this_xyz = this_edge.end_vertex.point.xyz
        xyz_ls = np.array([e.end_vertex_vertex.point.xyz for e in edge_list])
    elif mode == 'start_start':
        this_xyz = this_edge.start_vertex.point.xyz
        xyz_ls = np.array([e.start_vertex.point.xyz for e in edge_list])
    
    all_indxs = range(len(xyz_ls))    
    connected_indxs = find_xyzs_identical2this_xyz(this_xyz, xyz_ls)
    loose_indxs = utility.find_xs_not_in_ys(all_indxs, connected_indxs)
    
    if indx == True:
        result = {'connected': connected_indxs, 'loose': loose_indxs}
    else:
        connected_edges = np.take(edge_list, connected_indxs).tolist()
        loose_edges = np.take(edge_list, loose_indxs).tolist()
        result = {'connected': connected_edges, 'loose': loose_edges}
    return result

def find_xyzs_identical2this_xyz(this_xyz: np.ndarray, xyz_list: np.ndarray) -> np.ndarray:
    """
    Compare this_xyz to the xyz_list and find all the xyz that is the same as this_xyz. Returns the index of the identical xyz in the xyz_list.
 
    Parameters
    ----------
    this_xyz : np.ndarray
        a point [x,y,z].
    
    this_xyz : np.ndarray
        a np.ndarray of shape [Nxyz, 3].
 
    Returns
    -------
    index_list : np.ndarray
        List of index of the identical xyzs in the xyz_list
    """
    if type(xyz_list) != np.ndarray:
        xyz_list = np.array(xyz_list)

    xyzT = xyz_list.T
    xlist = xyzT[0]
    ylist = xyzT[1]
    zlist = xyzT[2]

    xvalid = np.where(xlist == this_xyz[0], True, False)
    yvalid = np.where(ylist == this_xyz[1], True, False)
    zvalid = np.where(zlist == this_xyz[2], True, False)

    cond1 = np.logical_and(xvalid, yvalid)
    cond2 = np.logical_and(cond1, zvalid)
    index_list = np.where(cond2)[0]
    return index_list