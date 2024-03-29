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
"""

__author__ = 'chenkianwee'
__date__ = '2023-11-30'
__copyright__ = '(C) 2023 by chenkianwee'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

import os
import pathlib
import inspect
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterString,
                       QgsProcessingParameterEnum,
                       QgsField)

from .algs import estimate_tree_height
import numpy as np

class treeHeightGrids(QgsProcessingAlgorithm):
    """
    This algorithms takes in polygon grids that represents the boundaries of tree locations. All layers must be in projected coordinate system.
    Check the point cloud layers and get the heights of all the points within each polygon boundary. Either use the maximum or percentile to decide the height of the 'tree'.
    """
    TREE_HEIGHT = 'TREE_HEIGHT'
    HEIGHT_ATT = 'HEIGHT ATTRIBUTE'
    POINT_CLOUDS = 'POINT_CLOUDS'
    PERCENTILE = 'PERCENTILE'
    RES_ATT = 'RES_ATT'
    CLASSIFICATION = 'CLASSIFICATION'
    Z_THRESHOLD = 'Z_THRESHOLD'

    def initAlgorithm(self, config):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.TREE_HEIGHT,
                self.tr('Tree Grid Layer'),
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.POINT_CLOUDS,
                self.tr('Point Clouds'),
                QgsProcessing.TypePointCloud
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERCENTILE,
                self.tr('Percentile of Point Height to Assume as Tree Height'),
                QgsProcessingParameterNumber.Integer,
                90, False, 0, 100
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.Z_THRESHOLD,
                self.tr('Minimum Height to be Considered a Tree (meter)'),
                QgsProcessingParameterNumber.Double,
                3.0, False, 0, 100000
            )
        )

        self.addParameter(QgsProcessingParameterString(
                self.RES_ATT,
                self.tr('Name of Attribute to Store the Estimated Tree Heights'),
                self.tr('gis3d_tree_height'),
                False, False
            )
        )

        self.CLASSIFICATIONS_SCHEME = [self.tr('Created, Never Classified'),
                                       self.tr('Unclassified'),
                                       self.tr('Ground'),
                                       self.tr('Low Vegetation'),
                                       self.tr('Medium Vegetation'),
                                       self.tr('High Vegetation')]
        
        self.addParameter(QgsProcessingParameterEnum(
                self.CLASSIFICATION,
                self.tr('Use Points in this Classification for Tree Heights Analysis'),
                self.CLASSIFICATIONS_SCHEME,
                defaultValue=1
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # get the file path of the tree layer
        tree_info = self.parameterDefinition(self.TREE_HEIGHT)
        tree_info = tree_info.valueAsJsonObject(parameters[self.TREE_HEIGHT], context)
        tree_info_split = tree_info.split('|')
        tree_path = tree_info_split[0]
        
        # get the file paths of the point cloud layers
        pt_layers = self.parameterDefinition(self.POINT_CLOUDS)
        pt_layers_ls = pt_layers.valueAsJsonObject(parameters[self.POINT_CLOUDS], context)

        percentile = self.parameterAsInt(parameters, self.PERCENTILE, context)
        z_threshold = self.parameterAsDouble(parameters, self.Z_THRESHOLD, context)
        class_val = self.parameterAsEnum(parameters, self.CLASSIFICATION, context) #int
        tree_height_name = self.parameterAsString(parameters, self.RES_ATT, context)
        #-----------------------------------------------------------------------------------------------------------------
        # estimate tree height algorithm
        fids_heights = estimate_tree_height(tree_path, pt_layers_ls, class_val, percentile, z_threshold, feedback=feedback)
        #-----------------------------------------------------------------------------------------------------------------
        tree = self.parameterAsVectorLayer(parameters, self.TREE_HEIGHT, context)
        tree_data = tree.dataProvider()
        tree_fields = tree.fields().names()
        if tree_height_name not in tree_fields:
            tree_data.addAttributes([QgsField(tree_height_name, QVariant.Double)])
            tree.updateFields()

        tree_fields = tree.fields().names()
        tree_height_field_id = tree_fields.index(tree_height_name)

        fids = fids_heights[0].astype(int)
        tree_heights = fids_heights[1]
        tree_heights = np.round(tree_heights, decimals=2)
        tree_heights = tree_heights.tolist()

        total = 30.0 / len(fids) if len(fids) else 0

        feedback.pushInfo('Changing attributes of Tree layer')
        for cnt, fid in enumerate(fids):
            # Stop the algorithm if cancel button has been clicked
            if feedback.isCanceled():
                break
            
            # Update the progress bar
            feedback.setProgress(int(cnt * total) + 60)

            attrs = {tree_height_field_id: tree_heights[cnt]}
            tree_data.changeAttributeValues({fid:attrs})

        results = {}
        results[self.TREE_HEIGHT] = tree_path
        # Update the progress bar
        feedback.setProgress(100)
        return results
    
    def name(self):
        return 'tree_height_grids'

    def groupId(self):
        return ''

    def shortHelpString(self):
        return  "<b>General:</b><br>"\
                "This algorithm estimates the tree height using a polygon grid layer (represents the boundary of the trees) with a list of point cloud layers.<br>"\
                "The algorithm checks the point cloud layers and get the heights of all the points within each polygon boundary. Use the user specified percentile value to decide the height of the 'tree'.<br>"\
                "<b>Parameters:</b><br>"\
                "Following Parameters must be set to run the algorithm:"\
                "<ul><li>Tree Grid Layer</li>"\
                "<li>Point Clouds</li></ul>"\
                "<b>Output:</b><br>"\
                "The output of the algorithm is adding a tree_height attribute as specified by the user to the Tree Grid Layer"
    
    def displayName(self):
        return self.tr(self.name())

    def group(self):
        return self.tr(self.groupId())

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)
    
    def icon(self):
        cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
        cmd_folder = str(pathlib.Path(cmd_folder).parent.absolute())
        icon = QIcon(os.path.join(cmd_folder, 'logo', 'tree_logo.png'))
        return icon

    def createInstance(self):
        return treeHeightGrids()