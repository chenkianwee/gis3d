# -*- coding: utf-8 -*-
"""
/***************************************************************************
 gis3d
                                 A QGIS plugin
 This plugins generates a 3D model based on the inputs
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-11-30
        copyright            : (C) 2023 by chenkianwee
        email                : chenkianwee@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'chenkianwee'
__date__ = '2023-11-30'
__copyright__ = '(C) 2023 by chenkianwee'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load gis3d class from file gis3d.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .gis3d import gis3dPlugin
    return gis3dPlugin(iface)
