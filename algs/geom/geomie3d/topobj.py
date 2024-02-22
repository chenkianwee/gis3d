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
from enum import Enum
from . import geom
from . import get
from . import create
from .geomdl import construct, operations
import numpy as np

class TopoType(Enum):
    
    VERTEX = 0
    EDGE = 1
    WIRE = 2
    FACE = 3
    SHELL = 4
    SOLID = 5
    COMPOSITE = 6

class Topology(object):
    """
    base class of all topology
    
    Parameters
    ----------
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
        
    Attributes
    ----------    
    attributes : dictionary
        The dictionary of attributes appended to the object.
    
    is_topo : bool
        an attribute to identify the object as a topology object.
        
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        self.attributes = attributes
        self.is_topo = True
    
    def overwrite_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att

class Vertex(Topology):
    """
    A vertex object
    
    Parameters
    ----------
    point : Point Object
        The Point Geometry
        
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------    
    point : Point Object
        The Point Object.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, point, attributes = {}):
        """Initialises the class"""
        super(Vertex, self).__init__(attributes = attributes)
        self.topo_type = TopoType.VERTEX
        self.point = point
    
    def to_dict(self):
        """
        This function creates a vertex dictionary.
        """
        topod = {'topo_type': 0, 'attributes': self.attributes, 'point': self.point.xyz.tolist()}
        return topod
        
class Edge(Topology):
    """
    An edge object
    
    Parameters
    ----------
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------
    curve_type : interger
        0 = polyline
        
    curve : Curve Object
        The curve Geometry
        
    vertex_list : list of vertex Object
        The vertices that defines the curve in the edge.
    
    start_vertex : vertex object
        The starting vertex.
    
    end_vertex : vertex object
        The end vertex.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        super(Edge, self).__init__(attributes = attributes)
        self.topo_type = TopoType.EDGE
        self.curve_type = None
        self.curve = None
        self.vertex_list = None
        self.start_vertex = None
        self.end_vertex = None
        
    def add_polyline_curve(self, vertex_list):
        """
        This function creates a polyline edge topology.
     
        Parameters
        ----------
        vertex_list : list of vertex Object
            The vertices that defines the polyline in the edge.
        """        
        self.curve_type = geom.CurveType.POLYLINE
        self.vertex_list = vertex_list
        self.start_vertex = vertex_list[0]
        self.end_vertex = vertex_list[-1]
        point_list = [v.point for v in vertex_list]
        curve = geom.PolylineCurve(point_list)
        self.curve = curve
    
    def add_bspline_curve(self, bspline_crv):
        """
        This function creates a bspline edge topology.
     
        Parameters
        ----------
        bspline_crv : geomdl.BSPLINE crv
            The bspline curve from NURBS-python.
        """
        self.curve_type = geom.CurveType.BSPLINE
        self.curve = bspline_crv
        pts = bspline_crv.evalpts
        vertex_list = create.vertex_list(pts)
        self.vertex_list = vertex_list
        self.start_vertex = vertex_list[0]
        self.end_vertex = vertex_list[-1]
    
    def to_dict(self):
        """
        This function creates a edge dictionary.
        """
        if self.curve_type == geom.CurveType.POLYLINE:
            topod = {'topo_type': 1, 'attributes': self.attributes, 'curve_type': 0, 
                     'vertex_list': [v.point.xyz.tolist() for v in self.vertex_list]}
        elif self.curve_type == geom.CurveType.BSPLINE:
            bcrv = self.curve
            ctrlpts = bcrv.ctrlpts
            degree = bcrv.degree
            resolution = bcrv.delta
            topod = {'topo_type': 1, 'attributes': self.attributes, 'curve_type': 1, 'ctrlpts': ctrlpts, 'degree': degree, 
                     'resolution' : resolution}
                
        return topod
        
class Wire(Topology):
    """
    A wire object
    
    Parameters
    ----------
    edge_list : List of Edge Objects
        List of Edge Topology
    
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------    
    edge_list : List of Edge Objects
        List of Edge Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, edge_list, attributes = {}):
        """Initialises the class"""
        super(Wire, self).__init__(attributes = attributes)
        if type(edge_list) != np.ndarray:
            edge_list = np.array(edge_list)
        self.topo_type = TopoType.WIRE
        self.edge_list = edge_list
    
    def to_dict(self):
        """
        This function creates a wire dictionary.
        """
        topod = {'topo_type': 2, 'attributes': self.attributes, 'edge_list': [e.to_dict() for e in self.edge_list]}
        return topod
        
class Face(Topology):
    """
    A face object
    
    Parameters
    ----------
    surface : Surface Object
        The Surface Geometry
        
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------
    surface_type : interger
        0 = polygon
        
    surface : Surface Object
        The Surface Geometry
        
    bdry_wire : Wire Object
        The wire object that defines the outer boundary of the face.
    
    hole_wire_list : list of wire Object
        The list of wire objects that define the holes in the face.
    
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        super(Face, self).__init__(attributes = attributes)
        self.topo_type = TopoType.FACE
        self.surface_type = None
        self.surface = None
        self.bdry_wire = None
        self.hole_wire_list = None
        self.normal = None
        
    def add_polygon_surface(self, bdry_wire, hole_wire_list = []):
        """
        This function adds a polygon to the face object.
     
        Parameters
        ----------
        bdry_wire : Wire Object
        The wire object that defines the outer boundary of the face.
    
        hole_wire_list : list of wire Object, optional
            The list of wire objects that define the holes in the face.
     
        """
        if type(hole_wire_list) != np.ndarray:
            hole_wire_list = np.array(hole_wire_list)
            
        self.surface_type = geom.SrfType.POLYGON
        self.bdry_wire = bdry_wire
        self.hole_wire_list = hole_wire_list
        #get the point list from the bdry wire
        pt_list = get.points_frm_wire(bdry_wire)
        
        hole_point_2dlist = [get.points_frm_wire(hole_wire) 
                             for hole_wire in hole_wire_list]
       
        surface = geom.PolygonSurface(pt_list, 
                                      hole_point_2dlist = hole_point_2dlist)
        self.surface = surface
        self.normal = surface.normal
    
    def update_polygon_surface(self):
        srf = self.surface
        bdry_wire = self.bdry_wire
        hole_wire_list = self.hole_wire_list 
        
        #get the point list from the bdry wire
        pt_list = get.points_frm_wire(bdry_wire)
        
        hole_point_2dlist = [get.points_frm_wire(hole_wire) 
                             for hole_wire in hole_wire_list]
        
        srf.update(pt_list,  hole_point_2dlist = hole_point_2dlist)
        self.normal = srf.normal
        
    def add_bspline_surface(self, bspline_srf):
        """
        This function adds a polygon to the face object.
     
        Parameters
        ----------
        bspline_srf : geomdl.BSPLINE srf
            The wire object that defines the outer boundary of the face.
    
        """
        self.surface_type = geom.SrfType.BSPLINE
        self.surface = bspline_srf
        crvs = construct.extract_curves(bspline_srf)
        ucrvs = crvs['u']
        vcrvs = crvs['v']
        crv1 = ucrvs[0]
        crv2 = vcrvs[-1]
        crv3 = ucrvs[-1]
        crv4 = vcrvs[0]
        
        crv3.reverse()
        crv4.reverse()
        
        crv1.delta = 0.1
        crv2.delta = 0.1
        crv3.delta = 0.1
        crv4.delta = 0.1
        
        edge1 = Edge()
        edge1.add_bspline_curve(crv1)
        edge2 = Edge()
        edge2.add_bspline_curve(crv2)
        edge3 = Edge()
        edge3.add_bspline_curve(crv3)        
        edge4 = Edge()
        edge4.add_bspline_curve(crv4)
        bdry_wire = create.wire_frm_edges([edge1,edge2,edge3,edge4])
        normal = list(operations.normal(bspline_srf, (0.5,0.5))[1])
        self.bdry_wire = bdry_wire
        self.hole_wire_list = []
        self.normal = normal
    
    def to_dict(self):
        """
        This function creates a face dictionary.
        """
        if self.surface_type == geom.SrfType.BSPLINE:
            print('bspline surface is not supported, face converted to polygon and write it to dictionary')
            
        verts = get.vertices_frm_wire(self.bdry_wire)
        hole_vertex_list = []
        for hw in self.hole_wire_list:
            hvs = get.vertices_frm_wire(hw)
            hole_vertex_list.append([hv.point.xyz.tolist() for hv in hvs])

            
        topod = {'topo_type': 3, 'attributes': self.attributes, 'surface_type': 0, 
                 'vertex_list': [v.point.xyz.tolist() for v in verts], 'hole_vertex_list': hole_vertex_list}
        
        return topod
    
class Shell(Topology):
    """
    A shell object
    
    Parameters
    ----------
    face_list : List of Face Objects
        List of Face Topology
    
    Attributes
    ----------    
    face_list : List of Face Objects
        List of Face Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, face_list, attributes = {}):
        """Initialises the class"""
        super(Shell, self).__init__(attributes = attributes)
        if type(face_list) != np.ndarray:
            face_list = np.array(face_list)
            
        self.topo_type = TopoType.SHELL
        self.face_list = face_list
        self.attributes = attributes
    
    def to_dict(self):
        """
        This function creates a shell dictionary.
        """
        topod = {'topo_type': 4, 'attributes': self.attributes, 'face_list': [f.to_dict() for f in self.face_list]}
        return topod

class Solid(Topology):
    """
    A solid object
    
    Parameters
    ----------
    shell : Shell Object
        Shell Topology
    
    Attributes
    ----------    
    shell : Shell Object
        Shell Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, shell, attributes = {}):
        """Initialises the class"""
        super(Solid, self).__init__(attributes = attributes)
        self.topo_type = TopoType.SOLID
        self.shell = shell
        self.attributes = attributes
    
    def to_dict(self):
        """
        This function creates a solid dictionary.
        """
        topod = {'topo_type': 5, 'attributes': self.attributes, 'shell': self.shell.to_dict()}
        return topod

class Composite(Topology):
    """
    A composite object
    
    Parameters
    ----------
    topology_list : List of Topology
        List of Topology
    
    Attributes
    ----------    
    topology_list : List of Topology
        List of Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, topology_list, attributes = {}):
        """Initialises the class"""
        super(Composite, self).__init__(attributes = attributes)
        if type(topology_list) != np.ndarray:
            topology_list = np.array(topology_list, dtype=object)
        self.topo_type = TopoType.COMPOSITE
        self.topology_list = topology_list
        self.vertex_list = None
        self.edge_list = None
        self.wire_list = None
        self.face_list = None
        self.shell_list = None
        self.solid_list = None
        self.composite_list = None
        self.composite_list2 = None
        
    def sorted2dict(self):
        self.vertex_list = []
        self.edge_list = []
        self.wire_list = []
        self.face_list = []
        self.shell_list = []
        self.solid_list = []
        self.composite_list = []
        self.composite_list2 = []
        
        topo_list = self.topology_list
        for topo in topo_list:
            if topo.topo_type == TopoType.VERTEX:
                self.vertex_list.append(topo)
            elif topo.topo_type == TopoType.EDGE:
                self.edge_list.append(topo)
            elif topo.topo_type == TopoType.WIRE:
                self.wire_list.append(topo)
            elif topo.topo_type == TopoType.FACE:
                self.face_list.append(topo)
            elif topo.topo_type == TopoType.SHELL:
                self.shell_list.append(topo)
            elif topo.topo_type == TopoType.SOLID:
                self.solid_list.append(topo)
            elif topo.topo_type == TopoType.COMPOSITE:
                self.composite_list.append(topo.sorted2dict())
                self.composite_list2.append(topo)
        
        return {'vertex': np.array(self.vertex_list), 
                'edge': np.array(self.edge_list),
                'wire': np.array(self.wire_list),
                'face': np.array(self.face_list),
                'shell': np.array(self.shell_list),
                'solid': np.array(self.solid_list),
                'composite': np.array(self.composite_list)}
    
    def to_dict(self):
        """
        This function creates a composite dictionary.
        """
        self.sorted2dict
        topod = {'topo_type': 6, 'attributes': self.attributes,
                 'vertex_list':[v.to_dict() for v in self.vertex_list],
                 'edge_list': [e.to_dict() for e in self.edge_list],
                 'wire_list': [w.to_dict() for w in self.wire_list],
                 'face_list': [f.to_dict() for f in self.face_list],
                 'shell_list': [sh.to_dict() for sh in self.shell_list],
                 'solid_list': [so.to_dict() for so in self.solid_list],
                 'composite_list': [c.to_dict() for c in self.composite_list2]}
        
        return topod