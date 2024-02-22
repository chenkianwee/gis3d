import time
import multiprocessing as mp

import laspy
import shapely

import numpy as np
import geopandas as gpd

# import geomie3d
from .geom import geomie3d

def estimate_tree_height(tree_path: str, pt_layer_ls: list[str], classification_val: int, percentile: int, z_threshold: float, feedback = None) -> np.ndarray:
    """
    compute the tree height of the layer
    
    Parameters
    ----------
    tree_path : str
        path of the tree gpkg file.
    
    pt_layer_ls : list[str]
        list of path of all the point cloud las files.

    classification_val : int
        the classification of the points to use for the computation.
    
    percentile : int
        the percentile to consider the height of the tree.
    
    percentile : int
        the percentile of the height of the points to consider as the height 
    
    feedback : QGIS feedback object, optional
        default=None. The interface to communicate to the qgis console window

    Returns
    -------
    processed_res : np.ndarray
        np.ndarray(shape(2, number of trees)). fid of feature of tree_path and height. Height = 0 means the height is lower than z_threshold, there are no trees
    """
    t1 = time.perf_counter()
    # ----------------------------------------------------------------------------------------------------
    # region: read all the polygons from the tree_path layer
    # ----------------------------------------------------------------------------------------------------
    gpd.options.io_engine = "pyogrio"
    msg = 'Reading tree bbox layer ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
        feedback.setProgress(1)
    else:
        print(msg)

    tree_df = gpd.read_file(tree_path, fid_as_index=True)
    tree_crs = tree_df.crs
    tree_epsg = tree_crs.to_epsg()
    are_polygons = tree_df.geom_type.apply(check_geom_type).to_numpy()
    if False in are_polygons: 
        feedback_msg = 'There are non polygon in Tree Grid layer'
        if feedback != None:
            feedback.pushInfo(feedback_msg)
        else:
            print(feedback_msg)

    tree_fid_ls = tree_df.index
    tree_poly = tree_df['geometry'].exterior
    tree_bbox_arrs = tree_poly.apply(extract_bbox)
    tree_bboxes = [geomie3d.create.bbox_frm_arr(tree_bbox_arr) for tree_bbox_arr in tree_bbox_arrs]
    glob_bbox = geomie3d.calculate.bbox_frm_bboxes(tree_bboxes)
    t2 = time.perf_counter()
    # endregion: ead all the polygons from the tree_path layer
    # ----------------------------------------------------------------------------------------------------
    # region: read all the las files and find all the las files that intersects with the tree_bboxes
    # ----------------------------------------------------------------------------------------------------
    msg = 'Reading all the las file layers ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
        feedback.setProgress(2)
    else:
        print(msg)

    # find all the las files that intersects with the main tree layer
    las_bboxes = []
    for pt_layer in pt_layer_ls:
        with laspy.open(pt_layer) as fh:
            header = fh.header
            crs = header.parse_crs()
            epsg = crs.to_epsg()
            if epsg == tree_epsg:
                las_mins = header.mins
                las_mxs = header.maxs
                las_bbox_arr = np.array([las_mins, las_mxs]).flatten()
                las_bbox = geomie3d.create.bbox_frm_arr(las_bbox_arr)
                las_bboxes.append(las_bbox)
            else:
                error_msg = pt_layer + ' is not the same CRS as the tree layer, please convert the layers to the appropriate CRS'
                if feedback != None:
                    feedback.pushInfo(error_msg)
                else:
                    print(error_msg)

    tree_glob_bboxes = np.array([glob_bbox])
    tree_glob_bboxes = np.repeat(tree_glob_bboxes, len(las_bboxes), axis=0)
    las_bboxes = np.array(las_bboxes)
    are_bboxes_related = geomie3d.calculate.are_bboxes1_related2_bboxes2(tree_glob_bboxes, las_bboxes, zdim = False)
    las_indx = np.where(are_bboxes_related)[0]
    t3 = time.perf_counter()
    # endregion: read all the las files and find all the las files that intersects with the tree_bboxes
    # ----------------------------------------------------------------------------------------------------
    # region: read all the las files that intersects with the tree_bboxes, get all the points create bbox and divide them up for quick processing.
    # ----------------------------------------------------------------------------------------------------
    msg = 'Finding the boundary of the chosen las files ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
        feedback.setProgress(3)
    else:
        print(msg)

    las_chosen_ls = []
    las_pts_all = []
    for las_id in las_indx:
        las_chosen = laspy.read(pt_layer_ls[las_id])
        classify = np.unique(las_chosen.classification)
        if classification_val in classify:
            las_pts = las_chosen.xyz[las_chosen.classification == classification_val]
        else:
            las_pts = las_chosen.xyz
            error_msg = 'classification does not exist in las file, will just use all the points in the las file'
            if feedback != None:
                feedback.pushInfo(error_msg)
            else:
                print(error_msg)
        
        las_pts_all.append(las_pts)
        las_chosen_ls.append(las_chosen)

    las_pts_all = np.vstack(las_pts_all)
    las_pts_bbox = geomie3d.calculate.bbox_frm_xyzs(las_pts_all)
    t4 = time.perf_counter()
    # endregion: read all the las files that intersects with the tree_bboxes, get all the points create bbox and divide them up for quick processing.
    # ----------------------------------------------------------------------------------------------------
    # region: get all the points and tree_bboxes in the div_bboxes
    # ----------------------------------------------------------------------------------------------------
    # Init multiprocessing.Pool()
    # cpu_cnt = mp.cpu_count()
    # pool = mp.Pool(int(cpu_cnt/3))
    # split the points into parallel lists
    msg = 'Dividing the data into grids ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
        feedback.setProgress(4)
    else:
        print(msg)
    xrange = las_pts_bbox.bbox_arr[3] - las_pts_bbox.bbox_arr[0]
    yrange = las_pts_bbox.bbox_arr[4] - las_pts_bbox.bbox_arr[1]
    div_bboxes_x = int(xrange/100)
    div_bboxes_y = int(yrange/100)
    div_bboxes = geomie3d.create.grid3d_from_bbox(las_pts_bbox, div_bboxes_x, div_bboxes_y, 0)
    tree_midpts = geomie3d.calculate.bboxes_centre(tree_bboxes)
    msg = 'Dividing the tree bboxes data into grids ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
    else:
        print(msg)
    # 3000 points per calculation
    ntree_split = int(len(tree_midpts)/3000)
    tree_results = match_pts2bboxes(tree_midpts, div_bboxes, ntree_split)
    # 100k per calculation
    msg = 'Dividing the las point cloud data into grids ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
    else:
        print(msg)
    nlas_split = int(len(las_pts_all)/100000)
    las_results = match_pts2bboxes(las_pts_all, div_bboxes, nlas_split)
    
    div_bbox_id = range(len(div_bboxes))
    tree_id_in_bbox_id = sep_into_bbx_ls(tree_results, div_bbox_id)
    las_pt_id_in_bbox_id = sep_into_bbx_ls(las_results, div_bbox_id)
    # ----------------------------------------------------------------------------------------------------
    # region:for viz purpose
    # ----------------------------------------------------------------------------------------------------
    # colour_ls = ['red', 'green', 'blue']
    # colour_ls2 = ['blue', 'red', 'green']
    # viz_ls = []
    # for cnt, dbb_id in enumerate(div_bbox_id):
    #     las_pts_in_bbox = np.take(las_pts_all, las_pt_id_in_bbox_id[dbb_id], axis=0)
    #     tree_bboxes_in_bbox = np.take(tree_bboxes, tree_id_in_bbox_id[dbb_id])
    #     npts_in_bbox = len(las_pts_in_bbox)
    #     ntrees_in_bbox = len(tree_bboxes_in_bbox)
    #     if npts_in_bbox != 0 and ntrees_in_bbox != 0:
    #         vs = geomie3d.create.vertex_list(las_pts_in_bbox)
    #         boxes = geomie3d.create.boxes_frm_bboxes(tree_bboxes_in_bbox)
    #         cmp = geomie3d.create.composite(boxes)
    #         edges = geomie3d.get.edges_frm_composite(cmp)
            
    #         viz_ls.append({'topo_list': vs, 'colour': colour_ls[cnt%3]})
    #         viz_ls.append({'topo_list': edges, 'colour': colour_ls2[cnt%3]})

    #     print('Computing grid', cnt+1, len(div_bbox_id))
    
    # div_boxes = geomie3d.create.boxes_frm_bboxes(div_bboxes)
    # div_cmp = geomie3d.create.composite(div_boxes)
    # div_edges = geomie3d.get.edges_frm_composite(div_cmp)
    # viz_ls.append({'topo_list': div_edges, 'colour': 'white'})
    # geomie3d.viz.viz(viz_ls)
    # ----------------------------------------------------------------------------------------------------
    # endregion:for viz purpose
    # ----------------------------------------------------------------------------------------------------
    # endregion: get all the points and tree_bboxes in the div_bboxes
    # ----------------------------------------------------------------------------------------------------
    # region: compute tree heights
    # ----------------------------------------------------------------------------------------------------
    msg = 'Matching las to tree bbox ... ...'
    if feedback != None:
        feedback.pushInfo(msg)
        feedback.setProgress(5)
    else:
        print(msg)

    res_fids = []
    res_heights = []

    total = 55.0 / len(div_bbox_id) if len(div_bbox_id) else 0

    for cnt, dbb_id in enumerate(div_bbox_id):
        if feedback != None:
            if feedback.isCanceled():
                break
            # Update the progress bar
            feedback.setProgress(int(cnt * total)+5)
        else:
            print('Computing grid', cnt+1, len(div_bbox_id))

        las_pts_in_bbox = np.take(las_pts_all, las_pt_id_in_bbox_id[dbb_id], axis=0)
        tree_global_ids = tree_id_in_bbox_id[dbb_id]
        tree_bboxes_in_bbox = np.take(tree_bboxes, tree_global_ids)

        npts_in_bbox = len(las_pts_in_bbox)
        ntrees_in_bbox = len(tree_bboxes_in_bbox)
        if npts_in_bbox != 0 and ntrees_in_bbox != 0:
            match_ids = geomie3d.calculate.match_xyzs_2_bboxes(las_pts_in_bbox, tree_bboxes_in_bbox, zdim = False)
            ref_tree_ids = range(len(tree_bboxes_in_bbox))
            bboxid_heights = compute_tree_height(match_ids, ref_tree_ids, las_pts_in_bbox, tree_global_ids, percentile, z_threshold = z_threshold)
            bbox_fids = np.take(tree_fid_ls, bboxid_heights[0], axis=0).to_numpy()
            res_fids.append(bbox_fids)
            res_heights.append(bboxid_heights[1])
            # print(bbox_fids)
            # print(bboxid_heights[1])
            # print(len(bbox_fids))
            # print(len(bboxid_heights[1]))

    # ----------------------------------------------------------------------------------------------------
    # endregion: compute tree heights
    # ----------------------------------------------------------------------------------------------------
    res_fids = np.concatenate(res_fids)
    res_heights = np.concatenate(res_heights)
    res_fids_heights = np.array([res_fids, res_heights])
    t5 = time.perf_counter()
    et5 = t5-t1
    et5 = int(et5/60)
    ttk_msg = 'Time taken (mins): ' + str(et5)
    if feedback != None:
        feedback.pushInfo(ttk_msg)
    else:
        print(ttk_msg)

    return res_fids_heights

def compute_tree_height(pt_bbox_indx: np.ndarray, ref_bbox_ids: np.ndarray, xyzs: np.ndarray, bbox_global_ids: np.ndarray, percentile: int, z_threshold: float = 3.0) -> np.ndarray:
    """
    compute the tree height within the bbox
    
    Parameters
    ----------
    pt_bbox_indx : np.ndarray
        np.ndarray(shape(2, number of points)).
    
    ref_bbox_ids : np.ndarray
        The ref id list of the box.

    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)). The points to process.
    
    bbox_global_ids : np.ndarray
        np.ndarray(shape(number of bboxes)). The global id of the bboxes.
    
    percentile : int
        the percentile of the height of the points to consider as the height 
    
    z_threshold : float, optional
        the minimum height threshold to consider the bbox to have tree. Default = 3m

    Returns
    -------
    processed_res : np.ndarray
        np.ndarray(shape(2, number of boxes)). bbox indices and height. Height = 0 means the height is lower than z_threshold, there are no trees
    """
    bbox_dict = sep_into_bbx_ls(pt_bbox_indx, ref_bbox_ids)
    bbox_ids = []
    heights = []
    for key in bbox_dict.keys():
        bbox_global_id = bbox_global_ids[key]
        pt_ids = bbox_dict[key]
        if len(pt_ids) != 0:
            bbox_ids.append(bbox_global_id)
            xyzs_chosen = np.take(xyzs, pt_ids, axis=0)
            xyzs_chosenT = xyzs_chosen.T
            z = xyzs_chosenT[2]
            height = np.percentile(z, percentile)
            if height < z_threshold:
                heights.append(0)
            else:
                heights.append(height)

    return np.array([bbox_ids, heights])

def match_pts2bboxes(xyzs: np.ndarray, bboxes: list[geomie3d.utility.Bbox], nsplit: int) -> np.ndarray:
    """
    split the points into smaller chunks based on nsplit, then process them chunk by chunk 
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)). The points to process.
    
    bboxes : list[geomie3d.utility.Bbox]
        the bboxes used to process the points.
    
    nsplit : int
        the number of chunk to split.

    Returns
    -------
    processed_res : np.ndarray
        np.ndarray(shape(2, number of points in the bboxes)). Point and bbox indices.
    """
    xyzs_split = split_pts(xyzs, nsplit)
    pt_id_ls = []
    bbox_id_ls = []
    prev_cnt = 0
    for cnt, a_xyzs in enumerate(xyzs_split):
        match_res = geomie3d.calculate.match_xyzs_2_bboxes(a_xyzs, bboxes, zdim = False)
        pt_ids = match_res[0] + (prev_cnt*cnt)
        pt_id_ls.append(pt_ids)
        bbox_id_ls.append(match_res[1])
        prev_cnt = len(a_xyzs)
    
    pt_id_ls = np.concatenate(pt_id_ls)
    bbox_id_ls = np.concatenate(bbox_id_ls)
    return np.array([pt_id_ls, bbox_id_ls])

def sep_into_bbx_ls(pt_bbox_indx: np.ndarray, ref_bbox_ids: np.ndarray) -> dict:
    """
    separate the point indices according to the bbox they belong to.
    
    Parameters
    ----------
    pt_bbox_indx : np.ndarray
        np.ndarray(shape(2, number of points)).
    
    ref_bbox_ids : np.ndarray
        The ref id list of the box.
    
    Returns
    -------
    separated_pt_ids : dict
        dict with bbox_id as keys and the point_ids as value.
    """
    pt_indx = pt_bbox_indx[0]
    bbox_indx = pt_bbox_indx[1]
    separated_pt_ids = {}
    for ref_bbox_id in ref_bbox_ids:
        unq_indx = np.where(bbox_indx == ref_bbox_id)[0]
        chosen_pt_ids = np.take(pt_indx, unq_indx)
        separated_pt_ids[ref_bbox_id] = chosen_pt_ids
    return separated_pt_ids

def split_pts(xyzs: np.ndarray, nsplit: int) -> list:
    """
    Splits the xyzs based on the nsplit parameter.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)), where each point is [x,y,z].
    
    nsplit : int
        the number of splits.
            
    Returns
    -------
    splitted : list
        list(shape(nsplit, number of points/nsplit, 3)).
        If number of points are not divisible by nsplit. It will return with the extra remainders at the last array.
    """
    npts = len(xyzs)
    div = npts/nsplit
    n_per_grp = int(div)
    remainder = npts%nsplit
    if remainder != 0:
        nelements = nsplit*n_per_grp
        xyzs1 = xyzs[:nelements]
        xyzs1r = np.reshape(xyzs1, (nsplit, n_per_grp, 3))
        xyzs2 = xyzs[nelements:].tolist()
        xyzs_split = xyzs1r.tolist()
        xyzs_split[-1].extend(xyzs2)
    else:
        xyzs_split = np.reshape(xyzs, (nsplit, n_per_grp, 3)).tolist()
    return xyzs_split

def extract_bbox(linear_ring: shapely.LinearRing):
    coordsT = np.array(linear_ring.coords).T
    x = coordsT[0]
    y = coordsT[1]

    mnx = np.min(x)
    mny = np.min(y)
    mxx = np.amax(x)
    mxy = np.amax(y)
    return np.array([mnx, mny, 0, mxx, mxy, 0])

def check_geom_type(geom_type: str):
     if geom_type == 'Polygon':
          return True
     else:
          return False

if __name__ == '__main__':
    tree_path = '/home/chenkianwee/kianwee_work/get/projects/r2/models/gpkg/grid5m_porous_test.gpkg'
    # tree_path = '/home/chenkianwee/kianwee_work/get/projects/r2/models/gpkg/grid5m_porous.gpkg'
    pt_layer_ls = ['/home/chenkianwee/kianwee_work/get/projects/r2/models/las/01_18TWK528467.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/02_18TWK529467.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/03_18TWK531467.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/04_18TWK528465.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/05_18TWK529465.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/06_18TWK531465.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/07_18TWK528464.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/08_18TWK529464.laz',
                   '/home/chenkianwee/kianwee_work/get/projects/r2/models/las/09_18TWK531464.laz']
    
    # pt_layer_ls = ['/home/chenkianwee/kianwee_work/get/projects/r2/models/las/04_18TWK528465.laz']
    
    estimate_tree_height(tree_path, pt_layer_ls, 1, 90, 3.0)