# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:18:39 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def death_ARg(XnZn,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):

    #i = np.random.randint(0, np.size(ARc))
    ARgp = ARgc.copy()
    if np.size(ARgc) == 1:
        ARgp[0] = 0
        bk_AR = 1/bk_AR
    else:    
        ARgp = np.delete(ARgp, -1).copy() # because birth adds new element at the end of the arrays, so death deletes the last element
        bk_AR = 1.
    
    # Check if AR model is stationary
#     coeff = np.flipud(-ARgp)
#     coeff = np.append(coeff,1)
#     zroots=np.roots(coeff)
#     TF = all(abs(zroots)>1) # True means it is stationary
    
#     if TF == True:
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgp,ARTc,XnZn)[0]

    MHP = bk_AR * np.exp((LogLp - LogLc)/T)        
    if np.random.rand()<=MHP:
        LogLc = LogLp
        ARgc = ARgp.copy() 
       
    return [LogLc,ARgc]