# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:52:59 2020

@author: emadg
"""
import numpy as np
from Log_Likelihood import Log_Likelihood
from cauchy_dist import cauchy_dist

def move(XnZn,globals_par,LogLc,ZLc,xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs):
    
    Nnode=int(np.size(xc))
    rho_salt_min = globals_par[1,0]
    rho_salt_max = globals_par[1,1]
    rho_base_min = globals_par[2,0]
    rho_base_max = globals_par[2,1]
    zn_min = globals_par[4,0]  
    alpha_min = globals_par[5,0] 
    alpha_max = globals_par[5,1]

    dsalt = rho_salt_max-rho_salt_min
    dbase = rho_base_max-rho_base_min
    
    for inode in np.arange(Nnode):
        for ipar in np.arange(1,4): # 1 or 2 or 3

            xp = xc.copy()
            zp = zc.copy()
            rhop = rhoc.copy()
            
            if ipar == 1 and inode>=3:
                xp[inode] = cauchy_dist(xc[inode],0.1,0,1,xc[inode])
                if np.isclose(xc[inode] , xp[inode])==1: continue
                
            elif ipar == 2 and inode>=3:
                zp[inode] = cauchy_dist(zc[inode],0.1,zn_min,1,zc[inode])
                if np.isclose(zc[inode] , zp[inode])==1: continue
        
            else:
                if rhoc[inode]<0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.02,rho_salt_min,rho_salt_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
                elif rhoc[inode]>0:
                    rhop[inode] = cauchy_dist(rhoc[inode],0.02,rho_base_min,rho_base_max,rhoc[inode])
                    if np.isclose(rhoc[inode] , rhop[inode])==1: continue
        
            
         
        
            if ipar<=2:
 
                logic_salt = (zp[inode]<=ZLc) and (rhoc[inode]>0)
                logic_base = (zp[inode]> ZLc) and (rhoc[inode]<0)
            

                if logic_salt==1 or logic_base==1:
                    r = np.random.rand()
                    rhop[inode] = logic_salt*(rho_salt_min+r*dsalt)+logic_base*(rho_base_min+r*dbase).copy()
            
            LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xp,zp,rhop,alpha_c,ARgc,ARTc,XnZn)[0]
            
            MHP = np.exp((LogLp - LogLc)/T)
            if np.random.rand()<=MHP:
                LogLc = LogLp
                xc = xp.copy()
                zc = zp.copy()
                rhoc = rhop.copy()
            
    ### Hyper Parameters
    for ipar in np.arange(2):
        
        rhop = rhoc.copy()
        ZLp = ZLc.copy()
        alpha_p = alpha_c.copy()
        
        if ipar == 0:
            
            ZLp = cauchy_dist(ZLc,0.1,zn_min,1,ZLc)
            logic_salt = np.logical_and(zc<=ZLp,rhoc>=0)
            logic_base = np.logical_and(zc>ZLp,rhoc<0)

            r = np.random.rand()
            rhop[logic_salt] = rho_salt_min+r*dsalt
            rhop[logic_base] = rho_base_min+r*dbase
            
        else:
            alpha_p = cauchy_dist(alpha_c,0.2,alpha_min,alpha_max,alpha_c)
            
    
        LogLp = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,xc,zc,rhop,alpha_p,ARgc,ARTc,XnZn)[0]

        MHP = np.exp((LogLp - LogLc)/T)
        
        if np.random.rand()<=MHP:
            LogLc = LogLp
            rhoc = rhop.copy()
            ZLc = ZLp
            alpha_c = alpha_p
       
    return [LogLc,ZLc,xc,zc,rhoc,alpha_c]