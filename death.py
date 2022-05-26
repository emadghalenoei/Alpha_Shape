# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:18:39 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def death(XnZn,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    i = np.random.randint(3, np.size(xc))
    xp = xc.copy()
    zp = zc.copy()
    rhop = rhoc.copy()
    xp = np.delete(xp, i)
    zp = np.delete(zp, i)
    rhop = np.delete(rhop, i)
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,zp,rhop,alpha_c,ARgc,ARTc,XnZn)[0]
            
    MHP = np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
        LogLc = LogLp
        xc = xp.copy()
        zc = zp.copy()
        rhoc = rhop.copy()
        
    return [LogLc,xc,zc,rhoc]
     
     
