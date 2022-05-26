# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:51:33 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood

def birth(XnZn,globals_par,LogLc,ZLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    zn_min = globals_par[4,0] 
    
    xp = np.random.rand()
    zp = zn_min+np.random.rand()*(1-zn_min)
    r = np.random.rand()
    
    logic_salt = (zp<=ZLc)
    logic_salt = logic_salt.astype(float)
    logic_base = (zp>ZLc)
    logic_base = logic_base.astype(float)
    rhop = logic_salt*(rho_salt_min+r*(rho_salt_max-rho_salt_min))+(logic_base)*(rho_base_min+r*(rho_base_max-rho_base_min))
    xp = np.append(xc,xp).copy()
    zp = np.append(zc,zp).copy()
    rhop = np.append(rhoc,rhop).copy()
    
    LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,zp,rhop,alpha_c,ARgc,ARTc,XnZn)[0]
    MHP = np.exp((LogLp - LogLc)/T)
    if np.random.rand()<=MHP:
        LogLc = LogLp
        xc = xp.copy() 
        zc = zp.copy() 
        rhoc = rhop.copy() 
        
    return [LogLc,xc,zc,rhoc]
    
    
#     k1 = np.size(ARc)
#     k2 = k1 + 1
#     Priork1 = (k1 - 20)**2 if k1>20 else 1
#     Priork2 = (k2 - 20)**2 if k2>20 else 1
    
    


