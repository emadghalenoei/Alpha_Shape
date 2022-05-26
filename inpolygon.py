# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:30:50 2020

@author: emadg
"""

import numpy as np
from matplotlib import path

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    
    return p.contains_points(q).reshape(shape)
