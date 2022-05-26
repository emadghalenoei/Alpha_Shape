# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:52:59 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from cauchy_dist import cauchy_dist

def move_ARg(XnZn,AR_bounds,LogLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    NAR = int(np.size(ARgc))
    
#     AR_min = globals_par[4,0]
#     AR_max = globals_par[4,1]

    
    for iar in np.arange(NAR):
        
        AR_min = AR_bounds[iar+1, 0]
        AR_max = AR_bounds[iar+1, 1]
        std_cauchy = abs(AR_max-AR_min)/40
        
        ARgp = ARgc.copy()
        ARgp[iar] = cauchy_dist(ARgc[iar],std_cauchy,AR_min,AR_max,ARgc[iar])
        if np.isclose(ARgc[iar] , ARgp[iar])==1: continue
            
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhoc,alpha_c,ARgp,ARTc,XnZn)[0]

        MHP = np.exp((LogLp - LogLc)/T)
        if np.random.rand()<=MHP:
            LogLc = LogLp
            ARgc = ARgp.copy()            
    
    return [LogLc,ARgc]