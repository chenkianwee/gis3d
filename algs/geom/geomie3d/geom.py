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
from . import calculate
import numpy as np 

class Point(object):
    """
    A point geometry
    
    Parameters
    ----------
    xyz : tuple
        The xyz coordinates
    
    Attributes
    ----------    
    xyz : 1D ndarry
        1D array specifying the xyz coordinates
        
    """
    def __init__(self, xyz):
        """Initialises the class"""
        if type(xyz) != np.ndarray:
            xyz = np.array(xyz)
        self.xyz = xyz
        
class CurveType(Enum):
    
    POLYLINE = 0
    BSPLINE = 1
    
class PolylineCurve(object):
    """
    A curve geometry
    
    Parameters
    ----------
    point_list : List of Point Geometry
        List of Point Geometry
        
    Attributes
    ----------    
    point_list : List of Point Geometry
        List of Point Geometry
    
    """
    def __init__(self, point_list):
        """Initialises the class"""
        if type(point_list) != np.ndarray:
            point_list = np.array(point_list)
        self.curve_type = CurveType.POLYLINE
        self.point_list = point_list
        
class Surface(object):
    """
    base class of all surface geometry
    
    Parameters
    ----------
    point_list : List of Point Geometry
        List of Point Geometry
    
    Attributes
    ----------    
    point_list : List of Point Geometry
        List of Point Geometry
        
    """
    def __init__(self):
        """Initialises the class"""
        self.normal = None


class SrfType(Enum):
    
    POLYGON = 0
    BSPLINE = 1


class PolygonSurface(Surface):
    """
    A polygon surface geometry
    
    Parameters
    ----------
    point_list : List of Point Geometry
        List of Point Geometry defining the boundary of the polygon
        
    hole_point_2dlist : 2D list of Point Geometry, optional
        2d list of Point Geometry defining the holes of the polygon
    
    Attributes
    ----------    
    point_list : List of Point Geometry
        List of Point Geometry defining the boundary of the polygon
        
    hole_point_2dlist : 2D list of Point Geometry
        2d list of Point Geometry defining the holes of the polygon
        
    """
    def __init__(self, point_list, hole_point_2dlist = []):
        """Initialises the class"""
        super(PolygonSurface, self).__init__()
        if type(point_list) != np.ndarray:
            point_list = np.array(point_list)
            
        self.surface_type = SrfType.POLYGON
        self.hole_point_2dlist = None
        if len(hole_point_2dlist) !=0:
            if type(hole_point_2dlist) != np.ndarray:
                hole_point_2dlist = np.array(hole_point_2dlist)
            self.hole_point_2dlist = hole_point_2dlist
        
        self.point_list = point_list
        self.calc_normal()
    
    def calc_normal(self):
        point_list = self.point_list
        #calculate the normal of the surface
        xyz_list = [point.xyz for point in point_list]
        vector1 = xyz_list[1] - xyz_list[0]
        vector2 = xyz_list[-1] - xyz_list[0]
        normal = calculate.cross_product(vector1, vector2)
        normal = calculate.normalise_vectors(normal)
        self.normal = normal
    
    def update(self, point_list,  hole_point_2dlist = []):
        self.point_list = point_list
        self.hole_point_2dlist = None
        if len(hole_point_2dlist) !=0:
            if type(hole_point_2dlist) != np.ndarray:
                hole_point_2dlist = np.array(hole_point_2dlist)
            self.hole_point_2dlist = hole_point_2dlist
            
        self.calc_normal()
        