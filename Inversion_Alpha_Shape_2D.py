# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:14:42 2020

@author: emadghalenoei
"""
from mpi4py import MPI
import os
import sys
import shutil
import numpy as np
import math
import time
from scipy.interpolate import CubicSpline
from Model_Making import Model_Making
from Gravity_Kernel_Expanded import Gravity_Kernel_Expanded
from Mag_Kernel_Expanded import Mag_Kernel_Expanded
import matplotlib.pyplot as plt
from Initializing import Initializing
from Chain2xz import Chain2xz
from select_step import select_step
from birth import birth
from death import death
from move import move
from birth_ARg import birth_ARg
from death_ARg import death_ARg
from move_ARg import move_ARg
from birth_ART import birth_ART
from death_ART import death_ART
from move_ART import move_ART
from Imshow_BestChain import Imshow_BestChain
from datetime import datetime
from scipy.signal import lfilter


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nchain = comm.Get_size()-1 # No. MCMC chains or MPI threads


plt.close('all')

Ndatapoints = 30         # Number of total data in 1D array shape
CX = 100                 # must be dividable by downsample rate of x 
CZ = 100                 # must be dividable by downsample rate of z

XnZn = np.zeros(CX*CZ)
dg_obs = np.zeros(Ndatapoints,dtype=float)
dT_obs = np.zeros(Ndatapoints,dtype=float)
Kernel_Grv = np.zeros((Ndatapoints,CX*CZ))
Kernel_Mag = np.zeros((Ndatapoints,CX*CZ))
globals_par = np.zeros((6,2))
globals_xyz = np.zeros((2,2))
AR_bounds = np.zeros((4,2))

Kmin = 6
Kmax = 50
KminAR = 0
KmaxAR = 3
Chain = np.zeros(6+2*KmaxAR+Kmax*3)

NT1 = int(np.floor((Nchain+1)/2))           # number of chains with T=1
dt = 1.2                                # ratio between temperature levels
TempLevels=np.arange(Nchain-NT1,0,-1)   # define Temp Levels
Temp = np.hstack((pow(dt,TempLevels),np.ones(NT1)))

NKEEP = 1000          # dump a binary file to desk every NKEEP records
NMCMC = 100000*NKEEP   #number of random walks



if rank == 0:
    
    loaddesk = 1

    fpath_loaddesk = os.getcwd()+'/loaddesk'
    
    daytime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    fpath = os.getcwd()+'//'+daytime
    if os.path.exists(fpath) and os.path.isdir(fpath):
        shutil.rmtree(fpath)
    os.mkdir(fpath)
    
    fpath_bnfiles = fpath+'/BinFormat'
    os.mkdir(fpath_bnfiles)
    fpath_PDF = fpath+'/PDF'
    os.mkdir(fpath_PDF)
    fpath_Restart = fpath+'/Restart'
    os.mkdir(fpath_Restart)
    
    ikeep = 0 #counter when writing to output files
    ChainKeep = np.zeros((NKEEP,Chain.size))
    
    ChainAll = np.zeros((Nchain,Chain.size)) # ChainAll keeps the latest Chain of each source, for restart program...
    ChainHistory = np.empty((0,Chain.size))
        
    if loaddesk == 1:
        ChainAll = np.load(fpath_loaddesk+'//'+'ChainAll.npy') # use this, if the latest result exists
#         ChainAll[:10] = ChainAll[-10:]
        
    
    Gravity_Data = np.loadtxt('GRV_Profile.txt')
    Magnetic_Data = np.loadtxt('RTP_Profile.txt')

    DIS_GRV = np.sqrt((Gravity_Data[:,0]-Gravity_Data[0,0])**2 + (Gravity_Data[:,1]-Gravity_Data[0,1])**2)
    DIS_MAG = np.sqrt((Magnetic_Data[:,0]-Magnetic_Data[0,0])**2 + (Magnetic_Data[:,1]-Magnetic_Data[0,1])**2)

    xs = np.linspace(Gravity_Data[0,0],Gravity_Data[-1,0],Ndatapoints)
    ys = np.linspace(Gravity_Data[0,1],Gravity_Data[-1,1],Ndatapoints)
    dis_s = np.sqrt((xs-xs[0])**2 + (ys-ys[0])**2)
    
    GRV_SPLINE = CubicSpline(DIS_GRV, Gravity_Data[:,2])
    dg_obs = GRV_SPLINE(dis_s)
    
    MAG_SPLINE = CubicSpline(DIS_MAG, Magnetic_Data[:,2])
    dT_obs = MAG_SPLINE(dis_s)
    
    # model space
    Z0 = 0                
    ZEND = 10000         
    Pad_Length = 5000
    
    Azimuth = math.atan2(xs[-1]-xs[0],ys[-1]-ys[0])
    xmodel = np.linspace(xs[0]-Pad_Length*math.sin(Azimuth),xs[-1]+Pad_Length*math.sin(Azimuth),CX)
    ymodel = np.linspace(ys[0]-Pad_Length*math.cos(Azimuth) ,ys[-1]+Pad_Length*math.cos(Azimuth) ,CX)
    dismodel = np.linspace(dis_s[0]-Pad_Length,dis_s[-1]+Pad_Length,CX)
    zmodel = np.linspace(Z0,ZEND,CZ)
    
    X, Z = np.meshgrid(xmodel, zmodel)
    Y, Z = np.meshgrid(ymodel, zmodel)
    DISMODEL, Z = np.meshgrid(dismodel, zmodel)
    
    dx=abs(X[0,1]-X[0,0])
    dy=abs(Y[0,1]-Y[0,0])
    dz = abs(Z[1,0]-Z[0,0])
    dDis = abs(DISMODEL[0,1]-DISMODEL[0,0])
    
    x_min=np.min(X)-dx/2
    x_max=np.max(X)+dx/2
    y_min=np.min(Y)-dy/2
    y_max=np.max(Y)+dy/2
    z_min=np.min(Z)-dz/2
    z_max=np.max(Z)+dz/2
    dis_min = np.min(DISMODEL)-dDis/2
    dis_max = np.max(DISMODEL)+dDis/2
    
    Xn_Grid = np.divide(DISMODEL-dis_min,dis_max-dis_min)
    Zn_Grid = np.divide(Z-z_min,z_max-z_min)
    
    Xn = Xn_Grid.flatten('F')
    Zn = Zn_Grid.flatten('F')
    XnZn = np.column_stack((Xn,Zn))
    

    #[TrueDensityModel, TrueSUSModel] = Model_Making(Xn_Grid,Zn_Grid)
    
    TrueDensityModel = np.load(fpath_loaddesk+'//'+'TrueDensityModel.npy')
    TrueSUSModel = np.load(fpath_loaddesk+'//'+'TrueSUSModel.npy')
    
    
    Kernel_Grv = Gravity_Kernel_Expanded(DISMODEL,Z,dis_s)
    Kernel_Grv = Kernel_Grv*1e8
    
    I = 90 # inclination
    Fe = 43314 #(nT)
    Azimuth = math.atan2(xs[-1]-xs[0],ys[-1]-ys[0])
    Azimuth = Azimuth *180/math.pi
    Kernel_Mag = Mag_Kernel_Expanded(DISMODEL,Z,dis_s,I,Azimuth)
    Kernel_Mag = 2*Fe* Kernel_Mag
    
    dg_true = Kernel_Grv @ TrueDensityModel.flatten('F') # Unit(mGal)     
    dT_true = Kernel_Mag @ TrueSUSModel.flatten('F')      # Unit(nT) 
    
   # Adding noise

    AR_parameters_original_g = np.array([0.6,-0.5])
    noise_g_level = 0.08
    sigma_g_original=noise_g_level*max(abs(dg_true))
    uncorr_noise_g_original = sigma_g_original*np.random.randn(Ndatapoints)
    corr_noise_g_original = lfilter(np.atleast_1d(1),np.insert(-AR_parameters_original_g, 0, 1), uncorr_noise_g_original)   
    dg_obs = dg_true + corr_noise_g_original
    
    
    if loaddesk == 1:
        dg_obs = np.load(fpath_loaddesk+'//'+'dg_obs_08.npy') # use this, if the latest result exists 
        
    
    AR_parameters_original_T = np.array([0.])
    noise_T_level = 0.04
    sigma_T_original=noise_T_level*max(abs(dT_true))
    uncorr_noise_T_original = sigma_T_original*np.random.randn(Ndatapoints)
    corr_noise_T_original = lfilter(np.atleast_1d(1),np.insert(-AR_parameters_original_T, 0, 1), uncorr_noise_T_original)    
    dT_obs = dT_true + corr_noise_T_original 
    
    
    if loaddesk == 1:
        dT_obs = np.load(fpath_loaddesk+'//'+'dT_obs_04.npy') # use this, if the latest result exists 
        
    
    
    rho_salt_min = -0.4
    rho_salt_max = -0.2
    rho_base_min = 0.2
    rho_base_max = 0.4
    sus_base_max = rho_base_max/50
    zn_min = 0/z_max
    alpha_min = 1.
    alpha_max = 6.
    
    globals_par = np.matrix([[Kmin, Kmax], [rho_salt_min, rho_salt_max], [rho_base_min, rho_base_max], [KminAR, KmaxAR], [zn_min, sus_base_max], [alpha_min, alpha_max]])
    globals_xyz = np.matrix([[dis_min, dis_max], [z_min, z_max]])
    
    AR_bounds =  np.matrix([[0., 0.], [-0.85, 0.9], [-0.85, 0.1], [-0.25, 0.25]]) # AR0, AR1, AR2, AR3
    
    for ichain in np.arange(1,Nchain+1): #sources (ranks) 1,2,...,Nchain
        
        Chain[:] = 0.
        
#         ind = np.argsort(ChainAll[:,0])[::-1]
        #Chain_MaxL = ChainAll[ind[0]].copy()
        Chain_MaxL = ChainAll[ichain-1,:]
        
        if ichain <=9:
            loaddesk = 1
        else:
            loaddesk = 1
            
        Chain = Initializing(Chain,XnZn,globals_par,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,Chain_MaxL,loaddesk).copy()

            
        comm.Send(Chain, dest=ichain, tag=ichain)
    # investigate seed lator --> generate int number == workers
    
    # Save important arrays for posterior process
    ChainAll = np.zeros((Nchain,Chain.size))
    ChainAll_str = fpath_Restart+'//'+'ChainAll.npy'
    np.save(ChainAll_str, ChainAll)
    
    dg_obs_str = fpath_Restart+'//'+'dg_obs.npy'
    np.save(dg_obs_str, dg_obs)
    
    dT_obs_str = fpath_Restart+'//'+'dT_obs.npy'
    np.save(dT_obs_str, dT_obs)

    XnZn_str = fpath_Restart+'//'+'XnZn.npy'
    np.save(XnZn_str, XnZn)
    
    Kernel_Grv_str = fpath_Restart+'//'+'Kernel_Grv.npy'
    np.save(Kernel_Grv_str, Kernel_Grv)
    
    Kernel_Mag_str = fpath_Restart+'//'+'Kernel_Mag.npy'
    np.save(Kernel_Mag_str, Kernel_Mag)
    
    globals_par_str = fpath_Restart+'//'+'globals_par.npy'
    np.save(globals_par_str, globals_par)
    
    globals_xyz_str = fpath_Restart+'//'+'globals_xyz.npy'
    np.save(globals_xyz_str, globals_xyz)
    
    AR_bounds_str = fpath_Restart+'//'+'AR_bounds.npy'
    np.save(AR_bounds_str, AR_bounds)
    
    TrueDensityModel_str = fpath_Restart+'//'+'TrueDensityModel.npy'
    np.save(TrueDensityModel_str, TrueDensityModel)
    
    TrueSUSModel_str = fpath_Restart+'//'+'TrueSUSModel.npy'
    np.save(TrueSUSModel_str, TrueSUSModel)
    
    ChainHistory_str = fpath_Restart+'//'+'ChainHistory.npy'
    #np.save(ChainHistory_str, ChainHistory)
    
    AR_parameters_original_g_str = fpath_Restart+'//'+'AR_parameters_original_g.npy'
    np.save(AR_parameters_original_g_str, AR_parameters_original_g)
    
    AR_parameters_original_T_str = fpath_Restart+'//'+'AR_parameters_original_T.npy'
    np.save(AR_parameters_original_T_str, AR_parameters_original_T)
    
    Chain_raw_str = fpath_Restart+'//'+'Chain_raw.npy' 
    

if rank>0:
    comm.Recv(Chain, source=0, tag=rank)

XnZn = comm.bcast(XnZn, root=0) 
dg_obs = comm.bcast(dg_obs, root=0)
dT_obs = comm.bcast(dT_obs, root=0)
Kernel_Grv = comm.bcast(Kernel_Grv, root=0)
Kernel_Mag = comm.bcast(Kernel_Mag, root=0)
globals_par = comm.bcast(globals_par, root=0)
globals_xyz = comm.bcast(globals_xyz, root=0)
AR_bounds = comm.bcast(AR_bounds, root=0)
# #Chain = comm.bcast(Chain, root=0)

comm.Barrier()

if rank > 0: 
    T = Temp[rank-1]
else:
    T = 1

#### Inversion

########################################################################
## workers

if rank > 0:
    
    bk_Nodes = 0.3 # probability from M(k) to M(k+1)
    bk_AR = 0.3   # probability from M(k) to M(k+1)

    for imcmc in np.arange(1,NMCMC+1):  # 1 to NMCMC
        LogLc = Chain[0].copy()
        [xc, zc, rhoc, alpha_c, ZLc, ARgc, ARTc]= Chain2xz(Chain)
        
        if imcmc % 4 != 0:
        
            for istep in np.arange(1,np.random.randint(1,3)):
          
                step = select_step(globals_par[0,0],globals_par[0,1],np.size(xc),bk_Nodes)
        
                if step==91:
                    [LogLc,xc,zc,rhoc] = birth(XnZn,globals_par,LogLc,ZLc,
                                               xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs)
                elif step==92:
                    [LogLc,xc,zc,rhoc] = death(XnZn,LogLc,
                                               xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs)
                else:
                    [LogLc,ZLc,xc,zc,rhoc,alpha_c] = move(XnZn,globals_par,LogLc,ZLc,
                                                  xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs)
        
        elif imcmc>5000 and imcmc % 4 == 0:
            
            if ARgc[0] == 0:
                step = 91
            else:
                step = select_step(globals_par[3,0],globals_par[3,1],np.size(ARgc),bk_AR)

            if step==91:
                [LogLc,ARgc] = birth_ARg(XnZn,AR_bounds,LogLc,
                                         xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            elif step==92:
                [LogLc,ARgc] = death_ARg(XnZn,LogLc,
                                         xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            else:
                [LogLc,ARgc] = move_ARg(XnZn,AR_bounds,LogLc,
                                        xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs)
                
                
            if ARTc[0] == 0:
                step = 91
            else:
                step = select_step(globals_par[3,0],globals_par[3,1],np.size(ARTc),bk_AR)

            if step==91:
                [LogLc,ARTc] = birth_ART(XnZn,AR_bounds,LogLc,
                                         xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            elif step==92:
                [LogLc,ARTc] = death_ART(XnZn,LogLc,
                                         xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,bk_AR)
            else:
                [LogLc,ARTc] = move_ART(XnZn,AR_bounds,LogLc,
                                        xc,zc,rhoc,alpha_c,ARgc,ARTc,T,Kernel_Grv,Kernel_Mag,dg_obs,dT_obs)
                
        
        Chain[:] = 0.
        Chain[0] = LogLc.copy()
        Chain[1] = np.size(xc)
        Chain[2] = np.size(ARgc)
        Chain[3] = np.size(ARTc)
        Chain[4] = ZLc.copy()
        Chain[5] = alpha_c.copy()
        Chain[6:6+np.size(ARgc)+np.size(ARTc)+np.size(xc)*3] = np.concatenate((ARgc,ARTc,xc,zc,rhoc)).copy()
                
#         if imcmc % 2000 == 0:
#             print("rank: ",rank,", T: ","%.2f" %T, ", Iteration: ",imcmc, ", LogL: ","%.2f" %Chain[0], ", k: ",Chain[1], ", kg: ",Chain[2], ", kT: ",Chain[3])
#             sys.stdout.flush()
        
        ## Sending model to Master 
        comm.Send(Chain, dest=0, tag=rank)
        
        ## Receiving back from Master
        Chain[:] = 0.
        comm.Recv(Chain, source=0, tag=MPI.ANY_TAG)       
        
        
## MASTER rank == 0
else:
    #t_start = time.time()
    c = 0
    with open(ChainHistory_str,"wb") as f:
        
        for imcmc in np.arange(1,NMCMC+1):  # 1 to NMCMC
            raw = 0
            Chain[:] = 0.
            comm.Recv(Chain, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
            Chain_p = Chain.copy()
            source_p = status.source
            Tp = Temp[source_p-1].copy()
            LogLp = Chain_p[0].copy()  

            Chain[:] = 0.
            comm.Recv(Chain, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,status=status)
            Chain_q = Chain.copy()
            source_q = status.source
            Tq = Temp[source_q-1].copy()
            LogLq = Chain_q[0].copy()     

            if Tp != Tq:
                Prob=np.exp(((1/Tp)-(1/Tq))*(LogLq-LogLp))

                if np.random.rand()<=Prob:
                    Chain_0 = Chain_p.copy()
                    Chain_p = Chain_q.copy()
                    Chain_q = Chain_0.copy()

            Chain = Chain_p.copy()
            comm.Send(Chain, dest=source_p, tag=rank)
            Chain = Chain_q.copy()
            comm.Send(Chain, dest=source_q, tag=rank)

            ChainAll[source_p-1,:] = Chain_p.copy()
            ChainAll[source_q-1,:] = Chain_q.copy()

            if np.isin(ChainAll[:,2:4], 1).all() and np.isin(ChainAll[:,6:8], 0.).all():
                Chain_raw = ChainAll.copy()
                raw = 1

            ## save to binary format

            if Tp == 1:  
                ChainKeep[ikeep,:] = Chain_p.copy()
                ikeep += 1
            if (Tq == 1) and (ikeep<NKEEP):
                ChainKeep[ikeep,:] = Chain_q.copy()
                ikeep += 1    

            if ikeep == NKEEP:

                #if imcmc > 100000 and raw ==0: # after Burn-in
                if raw == 0: # after Burn-in
                    ChainKeep.tofile(f)
                    #np.savetxt(f,ChainKeep)
                    
                    #PLOT_LogL(ChainHistory[:,0],fpath_PDF,'LogL_c')
                else:
                    np.save(Chain_raw_str, Chain_raw)

                np.save(ChainAll_str, ChainAll)
                
                c += 1
                if c % 100 == 0:
                    
                    Imshow_BestChain(dis_min/1000,dis_max/1000,z_min/1000,z_max/1000,XnZn,
                                    CX,CZ,globals_par,ChainAll,fpath_PDF,'MaxLG_',c)
                
               # Imshow_BestChain(0,1,0,1,Xn_Grid,Zn_Grid,pnt,CX,CZ,globals_par,ChainKeep,fpath_PDF,'MaxLG_',imcmc)

    #             Imshow_Data(dis_s,dg_obs,dT_obs,XnZn,Kernel_Grv,Kernel_Mag,ChainKeep,fpath_PDF,'Data_',imcmc)

                #imcmc_str = fpath_bnfiles+'//'+str(imcmc)+'.npy'
                #np.save(imcmc_str, ChainKeep)
                ikeep = 0
                ChainKeep[:,:] = 0.
               


    #         if imcmc % 2000 == 0:
    #             t_stop = time.time()
    #             print("rank: ",rank,', Time for imcmc: ',imcmc," is: ",(t_stop-t_start)/3600,' hours')
    #             sys.stdout.flush()
    #             #t_start = time.time()


            ##ChainLoad = np.load(imcmcstr)
