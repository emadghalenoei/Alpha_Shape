# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:25:37 2020

@author: emadg
"""

def Chain2xz(Chain):
    Nnode = int(Chain[1])
    NARg = int(Chain[2])
    NART = int(Chain[3])
    ZL = Chain[4].copy()
    alpha = Chain[5].copy()
    ARg = Chain[6:6+NARg].copy()
    ART = Chain[6+NARg:6+NARg+NART].copy()
    x = Chain[6+NARg+NART:6+NARg+NART+Nnode].copy()
    z = Chain[6+NARg+NART+Nnode:6+NARg+NART+2*Nnode].copy()
    rho = Chain[6+NARg+NART+2*Nnode:6+NARg+NART+3*Nnode].copy()
    return [x, z, rho, alpha, ZL, ARg, ART]
