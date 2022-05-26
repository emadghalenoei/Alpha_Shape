# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:09:05 2021

@author: emadg
"""
import numpy as np
from matplotlib import path

def Alpha_shape_mask(grid_model,hull,XnZn):
    
    coord = hull.__geo_interface__
    flags_tot = np.zeros((np.size(grid_model)), dtype=bool)
    
    if hull.is_empty == False:
        Npoly = len(coord['coordinates'])
        
        for ipoly in np.arange(Npoly):
            if Npoly == 1:
                vert = coord['coordinates'][ipoly]
            else:
                vert = coord['coordinates'][ipoly][0]
                
            p = path.Path(vert)
            flags = p.contains_points(XnZn)
            flags_tot = flags_tot + flags
          
    Density_Model = flags_tot * grid_model                 
    return Density_Model