# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:51:33 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
def birth_ART(XnZn,AR_bounds,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR):
    
#     AR_min = globals_par[4,0]
#     AR_max = globals_par[4,1]
    
#     arp = AR_min + np.random.rand() * (AR_max-AR_min)

    if ARTc[0] == 0:
        AR_min = AR_bounds[1, 0]
        AR_max = AR_bounds[1, 1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
        
        ARTp = ARTc.copy()
        ARTp[0] = arp
        
    else:
        AR_min = AR_bounds[len(ARTc)+1, 0]
        AR_max = AR_bounds[len(ARTc)+1, 1]
        arp = AR_min + np.random.rand() * (AR_max-AR_min)
        ARTp = np.append(ARTc,arp).copy() # new ar coeff will be added at the end of the array 
        bk_AR = 1.

    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgc,ARTp,XnZn)[0]

    MHP = bk_AR * np.exp((LogLp - LogLc)/T)    
    if np.random.rand()<=MHP:
        LogLc = LogLp
        ARTc = ARTp.copy() 

    return [LogLc,ARTc]


