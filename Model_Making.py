# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:56:51 2020

@author: emadg
"""

import numpy as np
from inpolygon import inpolygon

def Model_Making(X,Z):
    A = [0.1, 1]
    B = [0.1, 0.9]
    C = [0.9, 0.9]
    D = [0.9, 1]
    xv = np.array([A[0], B[0], C[0], D[0]])
    yv = np.array([A[1], B[1], C[1], D[1]])
    inp = inpolygon(X,Z, xv, yv)
    base1 = inp * 0.35

    A = [0.3, 0.9]
    B = [0.3, 0.8]
    C = [0.8, 0.8]
    D = [0.8, 0.9]
    xv = np.array([A[0], B[0], C[0], D[0]])
    yv = np.array([A[1], B[1], C[1], D[1]])
    inp = inpolygon(X,Z, xv, yv)
    base2 = inp * 0.25

    A = [0.3, 0.8]
    B = [0.45, 0.8]
    C = [np.mean([A[0],B[0]]), 0.4]
    xv = np.array([A[0], B[0], C[0]])
    yv = np.array([A[1], B[1], C[1]])
    inp = inpolygon(X,Z, xv, yv)
    salt1 = inp * -0.35

    A = [0.45, 0.8]
    B = [0.8, 0.8]
    C = [0.8, 0.55]
    D = [0.5, 0.55]
    xv = np.array([A[0], B[0], C[0], D[0]])
    yv = np.array([A[1], B[1], C[1], D[1]])
    inp = inpolygon(X,Z, xv, yv)
    salt2 = inp * -0.25

    TrueDensityModel = base1 + base2 + salt1 + salt2
    TrueSUSModel = TrueDensityModel / 50
    TrueSUSModel[TrueDensityModel<0.2]=0
    # TF = TrueDensityModel>=0.2
    # TrueSUSModel = np.multiply(TrueSUSModel,TF)
    return [TrueDensityModel, TrueSUSModel]