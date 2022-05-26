# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:22:13 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from Chain2xz import Chain2xz

def Initializing(Chain,XnZn,globals_par,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,Chain_MaxL,loaddesk):
    
    Kmin = int(globals_par[0,0])
    Kmax = int(globals_par[0,1])
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    KminAR = int(globals_par[3,0])
    KmaxAR = int(globals_par[3,1])
    zn_min = globals_par[4,0]
#     AR_min = globals_par[4,0]
#     AR_max = globals_par[4,1]

    if loaddesk == 0:

        Nnode = Kmin
        xc = np.random.rand(Nnode)
        zc = zn_min+np.random.rand(Nnode)*(1-zn_min)
        xc[0] = 0.
        xc[1] = 0.5
        xc[2] = 1.
        zc[0:3] = 1; ## x of first three point are variable but their z are fixed (stuck at the bottom)
        ZLc = 0.7
        alpha_c = 4.
        r = np.random.rand(Nnode)
        logic_salt  = np.logical_and.reduce((zc<=ZLc,zc<=ZLc))
        logic_base  = np.logical_and.reduce((zc>ZLc,zc>ZLc))
        rhoc = logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min)) 
        ARgc = np.array([0.])
        ARTc = np.array([0.])
    else:
        [xc, zc, rhoc, alpha_c, ZLc, ARgc, ARTc]= Chain2xz(Chain_MaxL)
        ARgc = np.array([0.])
        ARTc = np.array([0.])
    
    LogLc = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgc,ARTc,XnZn)[0]
    
    Chain[0] = LogLc.copy()
    Chain[1] = np.size(xc)
    Chain[2] = np.size(ARgc)
    Chain[3] = np.size(ARTc)
    Chain[4] = ZLc
    Chain[5] = alpha_c
    Chain[6:6+np.size(ARgc)] = ARgc.copy()
    Chain[6+np.size(ARgc):6+np.size(ARgc)+np.size(ARTc)] = ARTc.copy()
    Chain[6+np.size(ARgc)+np.size(ARTc):6+np.size(ARgc)+np.size(ARTc)+np.size(xc)*3] = np.concatenate((xc,zc,rhoc)).copy()
    
    return Chain

    
    