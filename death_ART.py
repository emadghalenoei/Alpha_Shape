# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:18:39 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood

def death_ART(XnZn,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):

    #i = np.random.randint(0, np.size(ARc))
    ARTp = ARTc.copy()
    if np.size(ARTc) == 1:
        ARTp[0] = 0
        bk_AR = 1/bk_AR
    else:    
        ARTp = np.delete(ARTp, -1).copy() # because birth adds new element at the end of the arrays, so death deletes the last element
        bk_AR = 1. 
    
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgc,ARTp,XnZn)[0]

    MHP = bk_AR * np.exp((LogLp - LogLc)/T)        
    if np.random.rand()<=MHP:
        LogLc = LogLp
        ARTc = ARTp.copy() 
       
    return [LogLc,ARTc]