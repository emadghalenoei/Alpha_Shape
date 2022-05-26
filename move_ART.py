# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:52:59 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from cauchy_dist import cauchy_dist

def move_ART(XnZn,AR_bounds,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    NAR = int(np.size(ARTc))

    for iar in np.arange(NAR):
        
        AR_min = AR_bounds[iar+1, 0]
        AR_max = AR_bounds[iar+1, 1]
        std_cauchy = abs(AR_max-AR_min)/40
        
        ARTp = ARTc.copy()
        ARTp[iar] = cauchy_dist(ARTc[iar],std_cauchy,AR_min,AR_max,ARTc[iar])
        if np.isclose(ARTc[iar] , ARTp[iar])==1: continue

        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgc,ARTp,XnZn)[0]

        MHP = np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARTc = ARTp.copy()            
    
    return [LogLc,ARTc]