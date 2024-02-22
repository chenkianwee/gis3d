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
from itertools import chain

import numpy as np

from . import modify
from . import topobj
from . import geom

def topo_atts(topo: topobj.Topology):
    """
    Return attributes from the topology.
 
    Parameters
    ----------
    topo : topo object
        get the attributes from this topology.
        
    Returns
    -------
    attributes : dictionary
        attributes of the topology.
    """
    att = topo.attributes
    return att
    
def point_frm_vertex(vertex):
    """
    Return point from vertex.
 
    Parameters
    ----------
    vertex : topo object
        the vertex object to explore.
        
    Returns
    -------
    pt : point
        point object.
    """
    
    return vertex.point

def points_frm_wire(wire):
    """
    Return points from wire.
 
    Parameters
    ----------
    wire : topo object
        the wire object to explore.
        
    Returns
    -------
    pt_list : list of points
        A list of points.
    """
    edge_list = wire.edge_list
    #TODO currently only account for polyline edges
    #when getting vertices from non polyline curve, will have to do an approximation to
    #get the vertex list
    
    #TODO need to inherit the attributes too
    vertex_2dlist = np.array([edge.vertex_list for edge in edge_list 
                              if edge.curve_type == geom.CurveType.POLYLINE], dtype=object)
    
    vertex_list = list(chain(*vertex_2dlist))
    point_list = np.array([v.point for v in vertex_list])
    #fused the points
    fused_pts = modify.fuse_points(point_list)
    return fused_pts

def vertices_frm_edge(edge):
    """
    Return vertices from edge.
 
    Parameters
    ----------
    edge : topo object
        the edge object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    return edge.vertex_list

def vertices_frm_wire(wire):
    """
    Return vertices from wire.
 
    Parameters
    ----------
    wire : topo object
        the wire object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    edge_list = wire.edge_list
    vertex_2dlist = np.array([edge.vertex_list for edge in edge_list], dtype=object)
    
    vertex_list = np.array(list(chain(*vertex_2dlist)))
    #fused the points
    fused_vertices = modify.fuse_vertices(vertex_list)
    return fused_vertices

def face_normal(face):
    """
    Return normal from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    normal : tuple
        Tuple specifying normal.
    """
    return face.normal

def vertices_frm_face(face):
    """
    Return vertices from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    bdry_wire = face.bdry_wire
    hole_wires = face.hole_wire_list
    
    vertex_list = vertices_frm_wire(bdry_wire)
    
    hole_verts_2dlist = np.array([vertices_frm_wire(hole_wire) 
                                  for hole_wire in hole_wires], dtype=object)
    
    hole_verts = np.array(list(chain(*hole_verts_2dlist)))
    return np.append(vertex_list, hole_verts)

def bdry_vertices_frm_face(face):
    """
    Return boundary vertices from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    bdry_wire = face.bdry_wire
    vertex_list = vertices_frm_wire(bdry_wire)
    return vertex_list

def hole_vertices_frm_face(face):
    """
    Return boundary vertices from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    vertex_2dlist : 2d list of vertices
        A 2d list of vertices.
    """
    hole_wires = face.hole_wire_list
    hole_verts_2dlist = np.array([vertices_frm_wire(hole_wire) 
                                  for hole_wire in hole_wires], dtype=object)
    
    return hole_verts_2dlist

def vertices_frm_shell(shell):
    """
    Return vertices from shell.
 
    Parameters
    ----------
    shell : topo object
        the shell object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    face_list = shell.face_list
    vert2dlist = np.array([vertices_frm_face(face) for face in face_list], dtype=object)
    vertex_list = list(chain(*vert2dlist))
    
    return np.array(vertex_list)

def vertices_frm_solid(solid):
    """
    Return vertices from solid.
 
    Parameters
    ----------
    solid : topo object
        the solid object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    shell = solid.shell
    vertex_list = vertices_frm_shell(shell)
    
    return vertex_list

def vertices_frm_composite(composite):
    """
    Return vertices from composite.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    vertex_list : list of vertices
        A list of vertices.
    """
    sorted_d = unpack_composite(composite)
    vertex_list = np.array([])
    
    vertices = sorted_d['vertex']
    if len(vertices) > 0:
        vertex_list = np.append(vertex_list, vertices)
    
    edges = sorted_d['edge']
    if len(edges) > 0:
        e_verts_2darr = np.array([vertices_frm_edge(e) for e in edges], dtype=object)
        e_verts = list(chain(*e_verts_2darr))
        vertex_list = np.append(vertex_list, e_verts)
    
    wires = sorted_d['wire']
    if len(wires) > 0:
        w_verts_2darr = np.array([vertices_frm_wire(w) for w in wires], dtype=object)
        w_verts = list(chain(*w_verts_2darr))
        vertex_list = np.append(vertex_list, w_verts)
    
    faces = sorted_d['face']
    if len(faces) > 0:
        f_verts_2darr = np.array([vertices_frm_face(f) for f in faces], dtype=object)
        f_verts = list(chain(*f_verts_2darr))
        vertex_list = np.append(vertex_list, f_verts)

    shells = sorted_d['shell']
    if len(shells) > 0:
        sh_verts_2darr = np.array([vertices_frm_shell(sh) for sh in shells], dtype=object)
        sh_verts = list(chain(*sh_verts_2darr))
        vertex_list = np.append(vertex_list, sh_verts)
        
    solids = sorted_d['solid']
    if len(solids) > 0:
        sl_verts_2darr = np.array([vertices_frm_solid(sl) for sl in solids], dtype=object)
        sl_verts = list(chain(*sl_verts_2darr))
        vertex_list = np.append(vertex_list, sl_verts)
        
    return vertex_list

def edges_frm_wire(wire):
    """
    Return edges from wire.
 
    Parameters
    ----------
    wire : topo object
        the wire object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    return wire.edge_list

def edges_frm_face(face):
    """
    Return edges from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    bdry_wire = face.bdry_wire
    hole_wire_list = face.hole_wire_list
    
    bdry_edges = edges_frm_wire(bdry_wire)
    hole_edges_2dlist = np.array([edges_frm_wire(hw) for hw in hole_wire_list], dtype=object) 
    hole_edges = list(chain(*hole_edges_2dlist))
    edge_list = np.append(bdry_edges, hole_edges)
    return edge_list
    
def bdry_edges_frm_face(face):
    """
    Return boundary edges from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    bdry_wire = face.bdry_wire
    
    bdry_edges = edges_frm_wire(bdry_wire)
    return bdry_edges

def hole_edges_frm_face(face):
    """
    Return hole edges from face.
 
    Parameters
    ----------
    face : topo object
        the face object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    hole_wire_list = face.hole_wire_list
    hole_edges_2dlist = np.array([edges_frm_wire(hw) for hw in hole_wire_list], dtype=object) 
    return hole_edges_2dlist

def edges_frm_shell(shell):
    """
    Return edges from shell.
 
    Parameters
    ----------
    shell : topo object
        the shell object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    face_list = shell.face_list
    edge_2dlist = np.array([edges_frm_face(f) for f in face_list], dtype=object)
    edge_list = list(chain(*edge_2dlist))
    return np.array(edge_list)

def edges_frm_solid(solid):
    """
    Return edges from solid.
 
    Parameters
    ----------
    solid : topo object
        the solid object to explore.
        
    Returns
    -------
    edge_list : list of edges
        A list of edges.
    """
    shell = solid.shell
    edge_list = edges_frm_shell(shell)
    return edge_list
    
def edges_frm_composite(composite: topobj.Composite) -> list[topobj.Edge]:
    """
    Return edges from composite.
 
    Parameters
    ----------
    composite : topobj.Composite
        the topo object to explore.
        
    Returns
    -------
    edge_list : list[topobj.Edge]
        A list of edges.
    """
    sorted_d = unpack_composite(composite)
    edge_list = np.array([])
    
    edges = sorted_d['edge']
    if len(edges) > 0:
        edge_list = np.append(edge_list, edges)
    
    wires = sorted_d['wire']
    if len(wires) > 0:
        w_edges_2darr = np.array([edges_frm_wire(w) for w in wires], dtype=object)
        w_edges = list(chain(*w_edges_2darr)) 
        edge_list = np.append(edge_list, w_edges)
    
    faces = sorted_d['face']
    if len(faces) > 0:
        f_edges_2darr = np.array([edges_frm_face(f) for f in faces], dtype=object)
        f_edges = list(chain(*f_edges_2darr))
        edge_list = np.append(edge_list, f_edges)

    shells = sorted_d['shell']
    if len(shells) > 0:
        sh_edges_2darr = np.array([edges_frm_shell(sh) for sh in shells], dtype=object)
        sh_edges = list(chain(*sh_edges_2darr))
        edge_list = np.append(edge_list, sh_edges)
        
    solids = sorted_d['solid']
    if len(solids) > 0:
        sl_edges_2darr = np.array([edges_frm_solid(sl) for sl in solids], dtype=object)
        sl_edges = list(chain(*sl_edges_2darr))
        edge_list = np.append(edge_list, sl_edges)
        
    return edge_list

def wires_frm_face(face):
    """
    Return wires from face.
 
    Parameters
    ----------
    face : topo object
        the topo object to explore.
        
    Returns
    -------
    wire_list : list of wires
        A list of wires.
    """
    bdry_wire = np.array([face.bdry_wire], dtype=object)
    hole_wire_list = face.hole_wire_list
    return np.append(bdry_wire, hole_wire_list)

def bdry_wires_frm_face(face):
    """
    Return boundary wires from face.
 
    Parameters
    ----------
    face : topo object
        the topo object to explore.
        
    Returns
    -------
    wire : wire
        A wire.
    """
    bdry_wire = np.array([face.bdry_wire], dtype=object)

    return bdry_wire

def hole_wires_frm_face(face):
    """
    Return hole wires from face.
 
    Parameters
    ----------
    face : topo object
        the topo object to explore.
        
    Returns
    -------
    wire_list : a list of wires
        A list of hole wires.
    """
    hole_wire_list = face.hole_wire_list
    return hole_wire_list

def wires_frm_shell(shell):
    """
    Return wires from shell.
 
    Parameters
    ----------
    shell : topo object
        the topo object to explore.
        
    Returns
    -------
    wire_list : list of wires
        A list of wires.
    """
    face_list = shell.face_list
    wire_2dlist = np.array([wires_frm_face(f) for f in face_list], dtype=object)
    wire_list = list(chain(*wire_2dlist))
    return np.array(wire_list)
    
def wires_frm_solid(solid):
    """
    Return wires from solid.
 
    Parameters
    ----------
    solid : topo object
        the topo object to explore.
        
    Returns
    -------
    wire_list : list of wires
        A list of wires.
    """
    shell = solid.shell
    wire_list = wires_frm_shell(shell)
    
    return wire_list
    
def wires_frm_composite(composite):
    """
    Return wires from composite.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    wire_list : list of wires
        A list of wires.
    """
    
    sorted_d = unpack_composite(composite)
    wire_list = np.array([])
    
    wires = sorted_d['wire']
    if len(wires) > 0:
        wire_list = np.append(wire_list, wires)
    
    faces = sorted_d['face']
    if len(faces) > 0:
        f_wires_2darr = np.array([wires_frm_face(f) for f in faces], dtype=object)
        f_wires = list(chain(*f_wires_2darr))
        wire_list = np.append(wire_list, f_wires)

    shells = sorted_d['shell']
    if len(shells) > 0:
        sh_wires_2darr = np.array([wires_frm_shell(sh) for sh in shells], dtype=object)
        sh_wires = list(chain(*sh_wires_2darr))
        wire_list = np.append(wire_list, sh_wires)
        
    solids = sorted_d['solid']
    if len(solids) > 0:
        sl_wires_2darr = np.array([wires_frm_solid(sl) for sl in solids], dtype=object)
        sl_wires = list(chain(*sl_wires_2darr))
        wire_list = np.append(wire_list, sl_wires)
        
    return wire_list

def faces_frm_shell(shell):
    """
    Return faces from shell.
 
    Parameters
    ----------
    shell : topo object
        the topo object to explore.
        
    Returns
    -------
    face_list : list of faces
        A list of faces.
    """
    return shell.face_list

def faces_frm_solid(solid):
    """
    Return faces from solid.
 
    Parameters
    ----------
    solid : topo object
        the topo object to explore.
        
    Returns
    -------
    face_list : list of faces
        A list of faces.
    """
    shell = solid.shell
    return shell.face_list

def faces_frm_composite(composite):
    """
    Return faces from composite.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    face_list : list of faces
        A list of faces.
    """
    sorted_d = unpack_composite(composite)
    face_list = np.array([])
    
    faces = sorted_d['face']
    if len(faces) > 0:
        face_list = np.append(face_list, faces)

    shells = sorted_d['shell']
    if len(shells) > 0:
        sh_faces_2darr = np.array([faces_frm_shell(sh) for sh in shells], dtype=object)
        sh_faces = list(chain(*sh_faces_2darr))
        face_list = np.append(face_list, sh_faces)
        
    solids = sorted_d['solid']
    if len(solids) > 0:
        sl_faces_2darr = np.array([faces_frm_solid(sl) for sl in solids], dtype=object)
        sl_faces = list(chain(*sl_faces_2darr))
        face_list = np.append(face_list, sl_faces)
        
    return face_list

def shell_frm_solid(solid):
    """
    Return shell from solid.
 
    Parameters
    ----------
    solid : topo object
        the topo object to explore.
        
    Returns
    -------
    shel; : shell object
        A shell object.
    """
    return solid.shell

def shells_frm_composite(composite):
    """
    Return shells from composite.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    shell_list : list of shells
        A list of shells.
    """
    sorted_d = unpack_composite(composite)
    shell_list = np.array([])
    
    shells = sorted_d['shell']
    if len(shells) > 0:
        shell_list = np.append(shell_list, shells)
        
    solids = sorted_d['solid']
    if len(solids) > 0:
        sl_shells_2darr = np.array([shell_frm_solid(sl) for sl in solids], dtype=object)
        sl_shells = list(chain(*sl_shells_2darr))
        shell_list = np.append(shell_list, sl_shells)
        
    return shell_list

def solids_frm_composite(composite):
    """
    Return shells from composite.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    solid_list : list of solids
        A list of solids.
    """
    sorted_d = unpack_composite(composite)
    solid_list = np.array([])
    
    solids = sorted_d['solid']
    if len(solids) > 0:
        solid_list = np.append(solid_list, solids)
                    
    return solid_list
    
def unpack_composite(composite):
    """
    Return a dictionary sorted into 'vertex', 'edge', 'face', 'shell', 'solid' keywords.
 
    Parameters
    ----------
    composite : topo object
        the topo object to explore.
        
    Returns
    -------
    sorted_dictionary : dictionary
        A dictionary of sorted topology.
    """
    
    def unpack_composites(composites, sorted_d):
        composites2 = []
        for cmp in composites:
            sorted_d['vertex'] = np.append(sorted_d['vertex'], cmp['vertex'])
            sorted_d['edge'] = np.append(sorted_d['edge'], cmp['edge'])
            sorted_d['wire'] = np.append(sorted_d['wire'], cmp['wire'])
            sorted_d['face'] = np.append(sorted_d['face'], cmp['face'])
            sorted_d['shell'] = np.append(sorted_d['shell'], cmp['shell'])
            sorted_d['solid'] = np.append(sorted_d['solid'], cmp['solid'])
            if len(cmp['composite']) > 0:
                composites2.extend(cmp['composite'])
        return composites2
            
    sorted_d = composite.sorted2dict()
    composites = sorted_d['composite']
    sorted_d.pop('composite')
    ncmp = len(composites)
    while ncmp > 0:
        composites2 = unpack_composites(composites, sorted_d)
        ncmp = len(composites2)
        composites = composites2
    
    return sorted_d

def topo_explorer(topo2explore, topo2find):
    """
    Explore a topology and return the topology of interest if exist.
 
    Parameters
    ----------
    topo2explore : topo object
        the topo object to explore.
        
    topo2find : TopoType Object
        The topology object to find in the topo2xplore.
 
    Returns
    -------
    topo_list : list of topo
        A list of topo found, return empty list if nothing is found.
    """
    found_topos = np.array([])
    if topo2find == topobj.TopoType.VERTEX:
        if topo2explore.topo_type == topobj.TopoType.VERTEX:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.EDGE:
            found_topos = vertices_frm_edge(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.WIRE:
            found_topos = vertices_frm_wire(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.FACE:
            found_topos = vertices_frm_face(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SHELL:
            found_topos = vertices_frm_shell(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = vertices_frm_solid(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = vertices_frm_composite(topo2explore)
            
    elif topo2find == topobj.TopoType.EDGE:
        if topo2explore.topo_type == topobj.TopoType.EDGE:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.WIRE:
            found_topos = edges_frm_wire(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.FACE:
            found_topos = edges_frm_face(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SHELL:
            found_topos = edges_frm_shell(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = edges_frm_solid(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = edges_frm_composite(topo2explore)
            
    elif topo2find == topobj.TopoType.WIRE:
        if topo2explore.topo_type == topobj.TopoType.WIRE:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.FACE:
            found_topos = wires_frm_face(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SHELL:
            found_topos = wires_frm_shell(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = wires_frm_solid(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = wires_frm_composite(topo2explore)
    
    elif topo2find == topobj.TopoType.FACE:
        if topo2explore.topo_type == topobj.TopoType.FACE:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SHELL:
            found_topos = faces_frm_shell(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = faces_frm_solid(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = faces_frm_composite(topo2explore)
    
    elif topo2find == topobj.TopoType.SHELL:
        if topo2explore.topo_type == topobj.TopoType.SHELL:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = shell_frm_solid(topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = shells_frm_composite(topo2explore)
            
    elif topo2find == topobj.TopoType.SOLID:
        if topo2explore.topo_type == topobj.TopoType.SOLID:
            found_topos = np.append(found_topos, topo2explore)
        elif topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = solids_frm_composite(topo2explore)
    
    elif topo2find == topobj.TopoType.COMPOSITE:
        if topo2explore.topo_type == topobj.TopoType.COMPOSITE:
            found_topos = np.append(found_topos, topo2explore)
    
    return found_topos