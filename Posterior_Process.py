from mpi4py import MPI
import os
import sys
import shutil
import numpy as np
from Chain2xz import Chain2xz
import matplotlib.pyplot as plt
import time
from mean_online import mean_online
from std_online import std_online
from Log_Likelihood import Log_Likelihood
from scipy.linalg import toeplitz
from shapely.geometry import Point


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nworkers = comm.Get_size()-1 # No. MCMC chains or MPI threads


file_name = '2022_01_28-05_37_44_PM'
    
fpath_Restart = os.getcwd()+'//'+file_name+'//'+'Restart'

fpath_output = os.getcwd()+'//'+file_name+'//'+'Output'
if rank == 0:
    if os.path.exists(fpath_output) and os.path.isdir(fpath_output):
        shutil.rmtree(fpath_output)
    os.mkdir(fpath_output)

dg_obs = np.load(fpath_Restart+'//'+'dg_obs.npy')
dT_obs = np.load(fpath_Restart+'//'+'dT_obs.npy')
XnZn = np.load(fpath_Restart+'//'+'XnZn.npy')
Kernel_Grv = np.load(fpath_Restart+'//'+'Kernel_Grv.npy')
Kernel_Mag = np.load(fpath_Restart+'//'+'Kernel_Mag.npy')
globals_par = np.load(fpath_Restart+'//'+'globals_par.npy')
globals_xyz = np.load(fpath_Restart+'//'+'globals_xyz.npy')
# TrueDensityModel = np.load(fpath_Restart+'//'+'TrueDensityModel.npy')
# TrueSUSModel = np.load(fpath_Restart+'//'+'TrueSUSModel.npy')
# AR_parameters_original_g = np.load(fpath_Restart+'//'+'AR_parameters_original_g.npy')
# AR_parameters_original_T = np.load(fpath_Restart+'//'+'AR_parameters_original_T.npy')
AR_bounds = np.load(fpath_Restart+'//'+'AR_bounds.npy')
ChainHistory_str = fpath_Restart+'//'+'ChainHistory.npy'


Kmin = int(globals_par[0,0])
Kmax = int(globals_par[0,1])
rho_salt_min = globals_par[1,0]
rho_salt_max = globals_par[1,1]
rho_base_min = globals_par[2,0]
rho_base_max = globals_par[2,1]
KminAR = int(globals_par[3,0])
KmaxAR = int(globals_par[3,1])
alpha_min = globals_par[5,0]
alpha_max = globals_par[5,1]

AR0_min = AR_bounds[0,0]
AR0_max = AR_bounds[0,1]
AR1_min = AR_bounds[1,0]
AR1_max = AR_bounds[1,1]
AR2_min = AR_bounds[2,0]
AR2_max = AR_bounds[2,1]
AR3_min = AR_bounds[3,0]
AR3_max = AR_bounds[3,1]

dis_min = globals_xyz[0,0]
dis_max = globals_xyz[0,1]
z_min = globals_xyz[1,0]
z_max = globals_xyz[1,1]

CX = 100
CZ = 100
Ndatapoints = np.size(dg_obs)

Xn_Grid = np.reshape(XnZn[:,0],(CX,CZ),'F').copy()
Zn_Grid = np.reshape(XnZn[:,1],(CX,CZ),'F').copy()

Ncol = 6+2*KmaxAR+Kmax*3
ndelete = 400000 # must be dividable by Nworkers (e.x. 10, 100, ...)
Nchain = 30000   # numder of chains(rows) per each worker

with open(ChainHistory_str, 'rb') as f:
    
    Chainsplit = np.fromfile(f,count = ndelete*Ncol, dtype='float64')
    Chainsplit = np.fromfile(f,count = ndelete*Ncol, dtype='float64')
    Chainsplit = np.fromfile(f,count = ndelete*Ncol, dtype='float64')
    Chainsplit = np.fromfile(f,count = ndelete*Ncol, dtype='float64')

    for irank in np.arange(0,rank+1):    
        #f.seek(0,1)
        Chainsplit = np.fromfile(f,count = Nchain*Ncol, dtype='float64').reshape(Nchain,Ncol)

#Chainsplit = np.loadtxt(fpath_Restart+'//'+'ChainHistory.npy', skiprows= ndelete+rank*Nchain, max_rows=Nchain, dtype=np.float32)

print('rank: ', rank, ', number of chains: ', Chainsplit.shape[0])
sys.stdout.flush()


LogLkeep = np.zeros(Nchain).copy()
Nnode = np.zeros(Nchain).astype(int)
NARg = np.zeros(Nchain).astype(int)
NART = np.zeros(Nchain).astype(int)
ARgkeep = np.zeros((Nchain,KmaxAR)).copy()
ARTkeep = np.zeros((Nchain,KmaxAR)).copy()
sigmakeep_g = np.zeros(Nchain).copy()
sigmakeep_T = np.zeros(Nchain).copy()
rho_keep = np.zeros(1).copy()
alphakeep = np.zeros(Nchain).copy()

grid_g_mean = np.zeros(Xn_Grid.shape).copy()
grid_g_std  = np.zeros(Xn_Grid.shape).copy()

grid_T_mean = np.zeros(Xn_Grid.shape).copy()
grid_T_std  = np.zeros(Xn_Grid.shape).copy()

data_g_mean = np.zeros(dg_obs.size)
data_g_std  = np.zeros(dg_obs.size)

data_T_mean = np.zeros(dg_obs.size)
data_T_std  = np.zeros(dg_obs.size)

Cov_g_mean = np.zeros((Ndatapoints,Ndatapoints))
Cov_T_mean = np.zeros((Ndatapoints,Ndatapoints))

Cov_g_mean_0 = np.zeros((Ndatapoints,Ndatapoints))
Cov_T_mean_0 = np.zeros((Ndatapoints,Ndatapoints))

Cov_g_mean_1 = np.zeros((Ndatapoints,Ndatapoints))
Cov_T_mean_1 = np.zeros((Ndatapoints,Ndatapoints))

Cov_g_mean_2 = np.zeros((Ndatapoints,Ndatapoints))
Cov_T_mean_2 = np.zeros((Ndatapoints,Ndatapoints))

Cov_g_mean_3 = np.zeros((Ndatapoints,Ndatapoints))
Cov_T_mean_3 = np.zeros((Ndatapoints,Ndatapoints))

Cov_g_keep_0 = np.zeros((100,Ndatapoints*Ndatapoints))
Cov_T_keep_0 = np.zeros((100,Ndatapoints*Ndatapoints))

Cov_g_keep_1 = np.zeros((100,Ndatapoints*Ndatapoints))
Cov_T_keep_1 = np.zeros((100,Ndatapoints*Ndatapoints))

Cov_g_keep_2 = np.zeros((100,Ndatapoints*Ndatapoints))
Cov_T_keep_2 = np.zeros((100,Ndatapoints*Ndatapoints))

Cov_g_keep_3 = np.zeros((100,Ndatapoints*Ndatapoints))
Cov_T_keep_3 = np.zeros((100,Ndatapoints*Ndatapoints))

autocorr_rg_mean = np.zeros(dg_obs.size)
autocorr_rT_mean = np.zeros(dg_obs.size)
autocorr_stand_rg_mean = np.zeros(dg_obs.size)
autocorr_stand_rT_mean = np.zeros(dg_obs.size)

standardized_rg_mean = np.zeros(dg_obs.size)
standardized_rT_mean = np.zeros(dg_obs.size)

kg0 = 0
kg1 = 0
kg2 = 0
kg3 = 0
kT0 = 0
kT1 = 0
kT2 = 0
kT3 = 0
icovg0 = 0
icovg1 = 0
icovg2 = 0
icovg3 = 0
icovT0 = 0
icovT1 = 0
icovT2 = 0
icovT3 = 0


for ichain in np.arange(Nchain):
    
        Chain = Chainsplit[ichain,:].copy()
        [x, z, rho, alpha, ZL, ARg, ART]= Chain2xz(Chain).copy()
        
        [LogL, model_vec_g, model_vec_T, rg, rT, sigma_g, sigma_T, uncor_g, uncor_T] = Log_Likelihood(Kernel_Grv,Kernel_Mag,dg_obs,dT_obs,x,z,rho,alpha,ARg,ART,XnZn)
       
        
        LogLkeep[ichain] = Chain[0].copy()
        Nnode[ichain] = Chain[1].copy()
        NARg[ichain] = Chain[2].copy()
        NART[ichain] = Chain[3].copy()
        ARgkeep[ichain,0:NARg[ichain]] = ARg.copy()
        ARTkeep[ichain,0:NART[ichain]] = ART.copy()
        sigmakeep_g[ichain] = sigma_g
        sigmakeep_T[ichain] = sigma_T
        alphakeep[ichain] = alpha.copy()
        
        rho_keep = np.append(rho_keep,rho[rho != 0])
        
        gridi_g = model_vec_g.reshape((CX,CZ),order="F").copy()
        gridi_T = model_vec_T.reshape((CX,CZ),order="F").copy() 
                
        grid_g_mean = mean_online(ichain+1,gridi_g,grid_g_mean)
        grid_g_std = std_online(ichain+1,grid_g_std,gridi_g,grid_g_mean)
        
        grid_T_mean = mean_online(ichain+1,gridi_T,grid_T_mean)
        grid_T_std = std_online(ichain+1,grid_T_std,gridi_T,grid_T_mean)
        
        dg_pre = dg_obs - rg
        dT_pre = dT_obs - rT
        
        data_g_mean = mean_online(ichain+1,dg_pre,data_g_mean)
        data_g_std = std_online(ichain+1,data_g_std,dg_pre,data_g_mean)
        
        data_T_mean = mean_online(ichain+1,dT_pre,data_T_mean)
        data_T_std = std_online(ichain+1,data_T_std,dT_pre,data_T_mean)
        
        TT = np.zeros(Ndatapoints).copy()
        TT[0] = 1.
        TT[1:len(ARg)+1] = -ARg.copy()
        TT_toep = toeplitz(TT).copy()
        TT_tril = np.tril(TT_toep).copy()
        Corr_mat = np.linalg.inv(TT_tril).copy()
        lower_cholesky = Corr_mat * sigma_g # lower cholesky decomposition
        Cov_g = lower_cholesky @ lower_cholesky.T
        Cov_g_mean = mean_online(ichain+1,Cov_g,Cov_g_mean)
        
        if len(ARg) == 1 and ARg[0]==0:
            Cov_g_mean_0 = mean_online(kg0+1,Cov_g,Cov_g_mean_0)
            kg0 += 1
            if ichain>Nchain-200 and icovg0<100:
                Cov_g_keep_0[icovg0,:] = Cov_g.flatten()
                icovg0 += 1
            
            
            
        elif len(ARg) == 1 and ARg[0]!=0:
            Cov_g_mean_1 = mean_online(kg1+1,Cov_g,Cov_g_mean_1)
            kg1 += 1
            if ichain>Nchain-200 and icovg1<100:
                Cov_g_keep_1[icovg1,:] = Cov_g.flatten()
                icovg1 += 1
            
            
        elif len(ARg) == 2:
            Cov_g_mean_2 = mean_online(kg2+1,Cov_g,Cov_g_mean_2)
            kg2 += 1
            if ichain>Nchain-200 and icovg2<100:
                Cov_g_keep_2[icovg2,:] = Cov_g.flatten()
                icovg2 += 1            
            
            
        elif len(ARg) == 3:
            Cov_g_mean_3 = mean_online(kg3+1,Cov_g,Cov_g_mean_3)
            kg3 += 1
            if ichain>Nchain-200 and icovg3<100:
                Cov_g_keep_3[icovg3,:] = Cov_g.flatten()
                icovg3 += 1            
            
        
        TT = np.zeros(Ndatapoints).copy()
        TT[0] = 1.
        TT[1:len(ART)+1] = -ART.copy()
        TT_toep = toeplitz(TT).copy()
        TT_tril = np.tril(TT_toep).copy()
        Corr_mat = np.linalg.inv(TT_tril).copy()
        lower_cholesky = Corr_mat * sigma_T # lower cholesky decomposition
        Cov_T = lower_cholesky @ lower_cholesky.T
        Cov_T_mean = mean_online(ichain+1,Cov_T,Cov_T_mean)
        
        if len(ART) == 1 and ART[0]==0:
            Cov_T_mean_0 = mean_online(kT0+1,Cov_T,Cov_T_mean_0)
            kT0 += 1
            if ichain>Nchain-200 and icovT0<100:
                Cov_T_keep_0[icovT0,:] = Cov_T.flatten()
                icovT0 += 1              
            
            
        elif len(ART) == 1 and ART[0]!=0:
            Cov_T_mean_1 = mean_online(kT1+1,Cov_T,Cov_T_mean_1)
            kT1 += 1
            if ichain>Nchain-200 and icovT1<100:
                Cov_T_keep_1[icovT1,:] = Cov_T.flatten()
                icovT1 += 1              
            
            
        elif len(ART) == 2:
            Cov_T_mean_2 = mean_online(kT2+1,Cov_T,Cov_T_mean_2)
            kT2 += 1
            if ichain>Nchain-200 and icovT2<100:
                Cov_T_keep_2[icovT2,:] = Cov_T.flatten()
                icovT2 += 1             
            
            
        elif len(ART) == 3:
            Cov_T_mean_3 = mean_online(kT3+1,Cov_T,Cov_T_mean_3)
            kT3 += 1
            if ichain>Nchain-200 and icovT3<100:
                Cov_T_keep_3[icovT3,:] = Cov_T.flatten()
                icovT3 += 1            
            
        autocorr_rg = np.correlate(rg, rg, mode='full')
        autocorr_rg = autocorr_rg / autocorr_rg.max()
        autocorr_rg_mean = mean_online(ichain+1, autocorr_rg, autocorr_rg_mean)
        
        standardized_rg = uncor_g/sigma_g
        standardized_rg_mean = mean_online(ichain+1, standardized_rg, standardized_rg_mean)
        
        autocorr_stand_rg = np.correlate(standardized_rg, standardized_rg, mode='full')
        autocorr_stand_rg = autocorr_stand_rg / autocorr_stand_rg.max()
        autocorr_stand_rg_mean = mean_online(ichain+1, autocorr_stand_rg, autocorr_stand_rg_mean)
        
        autocorr_rT = np.correlate(rT, rT, mode='full')
        autocorr_rT = autocorr_rT / autocorr_rT.max()
        autocorr_rT_mean = mean_online(ichain+1, autocorr_rT, autocorr_rT_mean)
        
        standardized_rT = uncor_T/sigma_T
        standardized_rT_mean = mean_online(ichain+1, standardized_rT, standardized_rT_mean)
        
        autocorr_stand_rT = np.correlate(standardized_rT, standardized_rT, mode='full')
        autocorr_stand_rT = autocorr_stand_rT / autocorr_stand_rT.max()
        autocorr_stand_rT_mean = mean_online(ichain+1, autocorr_stand_rT, autocorr_stand_rT_mean)
        
        if ichain % 2000 == 0:
            print('rank: ', rank, ', ichain: ', ichain, ' from ', Nchain)
            sys.stdout.flush()

        
grid_g_mean_ToT = comm.reduce(grid_g_mean, op=MPI.SUM, root=0)
grid_g_std_ToT = comm.reduce(grid_g_std, op=MPI.SUM, root=0)

grid_T_mean_ToT = comm.reduce(grid_T_mean, op=MPI.SUM, root=0)
grid_T_std_ToT = comm.reduce(grid_T_std, op=MPI.SUM, root=0)

data_g_mean_ToT = comm.reduce(data_g_mean, op=MPI.SUM, root=0)
data_g_std_ToT = comm.reduce(data_g_std, op=MPI.SUM, root=0)

data_T_mean_ToT = comm.reduce(data_T_mean, op=MPI.SUM, root=0)
data_T_std_ToT = comm.reduce(data_T_std, op=MPI.SUM, root=0)

Cov_g_mean_ToT = comm.reduce(Cov_g_mean, op=MPI.SUM, root=0)
Cov_T_mean_ToT = comm.reduce(Cov_T_mean, op=MPI.SUM, root=0)

Cov_g_mean_0_ToT = comm.reduce(Cov_g_mean_0, op=MPI.SUM, root=0)
Cov_T_mean_0_ToT = comm.reduce(Cov_T_mean_0, op=MPI.SUM, root=0)

Cov_g_mean_1_ToT = comm.reduce(Cov_g_mean_1, op=MPI.SUM, root=0)
Cov_T_mean_1_ToT = comm.reduce(Cov_T_mean_1, op=MPI.SUM, root=0)

Cov_g_mean_2_ToT = comm.reduce(Cov_g_mean_2, op=MPI.SUM, root=0)
Cov_T_mean_2_ToT = comm.reduce(Cov_T_mean_2, op=MPI.SUM, root=0)

Cov_g_mean_3_ToT = comm.reduce(Cov_g_mean_3, op=MPI.SUM, root=0)
Cov_T_mean_3_ToT = comm.reduce(Cov_T_mean_3, op=MPI.SUM, root=0)



autocorr_rg_mean_ToT = comm.reduce(autocorr_rg_mean, op=MPI.SUM, root=0)
autocorr_rT_mean_ToT = comm.reduce(autocorr_rT_mean, op=MPI.SUM, root=0)

autocorr_stand_rg_mean_ToT = comm.reduce(autocorr_stand_rg_mean, op=MPI.SUM, root=0)
autocorr_stand_rT_mean_ToT = comm.reduce(autocorr_stand_rT_mean, op=MPI.SUM, root=0)

standardized_rg_mean_ToT = comm.reduce(standardized_rg_mean, op=MPI.SUM, root=0)
standardized_rT_mean_ToT = comm.reduce(standardized_rT_mean, op=MPI.SUM, root=0)

# gather all local arrays on process root, will return a list of numpy arrays
LogLkeep_TOT = comm.gather(LogLkeep, root=0)
alphakeep_TOT = comm.gather(alphakeep, root=0)
Nnode_TOT = comm.gather(Nnode, root=0)
NARg_TOT = comm.gather(NARg, root=0)
NART_TOT = comm.gather(NART, root=0)
ARgkeep_TOT = comm.gather(ARgkeep, root=0)
ARTkeep_TOT = comm.gather(ARTkeep, root=0)
sigmakeep_g_TOT = comm.gather(sigmakeep_g, root=0)
sigmakeep_T_TOT = comm.gather(sigmakeep_T, root=0)
rho_keep = np.delete(rho_keep, 0).copy()
rho_keep_TOT = comm.gather(rho_keep, root=0)

Cov_g_keep_0_TOT = comm.gather(Cov_g_keep_0, root=0)
Cov_T_keep_0_TOT = comm.gather(Cov_T_keep_0, root=0)
Cov_g_keep_1_TOT = comm.gather(Cov_g_keep_1, root=0)
Cov_T_keep_1_TOT = comm.gather(Cov_T_keep_1, root=0)
Cov_g_keep_2_TOT = comm.gather(Cov_g_keep_2, root=0)
Cov_T_keep_2_TOT = comm.gather(Cov_T_keep_2, root=0)
Cov_g_keep_3_TOT = comm.gather(Cov_g_keep_3, root=0)
Cov_T_keep_3_TOT = comm.gather(Cov_T_keep_3, root=0)
################################################################################################################
### save to desk

if rank == 0:
    
    # turn the list of arrays into a single array
    LogLkeep_TOT = np.concatenate(LogLkeep_TOT)
    alphakeep_TOT = np.concatenate(alphakeep_TOT)
    Nnode_TOT = np.concatenate(Nnode_TOT)
    NARg_TOT = np.concatenate(NARg_TOT)
    NART_TOT = np.concatenate(NART_TOT)
    ARgkeep_TOT = np.concatenate(ARgkeep_TOT)
    ARTkeep_TOT = np.concatenate(ARTkeep_TOT)
    sigmakeep_g_TOT = np.concatenate(sigmakeep_g_TOT)
    sigmakeep_T_TOT = np.concatenate(sigmakeep_T_TOT)
    rho_keep_TOT = np.concatenate(rho_keep_TOT)
    Cov_g_keep_0_TOT = np.concatenate(Cov_g_keep_0_TOT)
    Cov_T_keep_0_TOT = np.concatenate(Cov_T_keep_0_TOT)
    Cov_g_keep_1_TOT = np.concatenate(Cov_g_keep_1_TOT)
    Cov_T_keep_1_TOT = np.concatenate(Cov_T_keep_1_TOT)
    Cov_g_keep_2_TOT = np.concatenate(Cov_g_keep_2_TOT)
    Cov_T_keep_2_TOT = np.concatenate(Cov_T_keep_2_TOT)
    Cov_g_keep_3_TOT = np.concatenate(Cov_g_keep_3_TOT)
    Cov_T_keep_3_TOT = np.concatenate(Cov_T_keep_3_TOT)
    
    PMD_g = grid_g_mean_ToT/(Nworkers+1)
    PMD_g_str = fpath_output+'//'+'PMD_g.npy'
    np.save(PMD_g_str, PMD_g)
    
    STD_g = grid_g_std_ToT/(Nworkers+1)
    STD_g_str = fpath_output+'//'+'STD_g.npy'
    np.save(STD_g_str, STD_g)
    
    PMD_T = grid_T_mean_ToT/(Nworkers+1)
    PMD_T_str = fpath_output+'//'+'PMD_T.npy'
    np.save(PMD_T_str, PMD_T)
    
    STD_T = grid_T_std_ToT/(Nworkers+1)
    STD_T_str = fpath_output+'//'+'STD_T.npy'
    np.save(STD_T_str, STD_T)
    
    PMD_data_g = data_g_mean_ToT/(Nworkers+1)
    PMD_data_g_str = fpath_output+'//'+'PMD_data_g.npy'
    np.save(PMD_data_g_str, PMD_data_g)
    
    STD_data_g = data_g_std_ToT/(Nworkers+1)
    STD_data_g_str = fpath_output+'//'+'STD_data_g.npy'
    np.save(STD_data_g_str, STD_data_g)
    
    PMD_data_T = data_T_mean_ToT/(Nworkers+1)
    PMD_data_T_str = fpath_output+'//'+'PMD_data_T.npy'
    np.save(PMD_data_T_str, PMD_data_T)
    
    STD_data_T = data_T_std_ToT/(Nworkers+1)
    STD_data_T_str = fpath_output+'//'+'STD_data_T.npy'
    np.save(STD_data_T_str, STD_data_T)
    
    PMD_Cov_g = Cov_g_mean_ToT/(Nworkers+1)
    PMD_Cov_g_str = fpath_output+'//'+'PMD_Cov_g.npy'
    np.save(PMD_Cov_g_str, PMD_Cov_g)
    
    
    PMD_Cov_g_0 = Cov_g_mean_0_ToT/(Nworkers+1)
    PMD_Cov_g_0_str = fpath_output+'//'+'PMD_Cov_g_0.npy'
    np.save(PMD_Cov_g_0_str, PMD_Cov_g_0)
    
    PMD_Cov_g_1 = Cov_g_mean_1_ToT/(Nworkers+1)
    PMD_Cov_g_1_str = fpath_output+'//'+'PMD_Cov_g_1.npy'
    np.save(PMD_Cov_g_1_str, PMD_Cov_g_1)
    
    PMD_Cov_g_2 = Cov_g_mean_2_ToT/(Nworkers+1)
    PMD_Cov_g_2_str = fpath_output+'//'+'PMD_Cov_g_2.npy'
    np.save(PMD_Cov_g_2_str, PMD_Cov_g_2)
    
    PMD_Cov_g_3 = Cov_g_mean_3_ToT/(Nworkers+1)
    PMD_Cov_g_3_str = fpath_output+'//'+'PMD_Cov_g_3.npy'
    np.save(PMD_Cov_g_3_str, PMD_Cov_g_3)
    
    
    PMD_Cov_T = Cov_T_mean_ToT/(Nworkers+1)
    PMD_Cov_T_str = fpath_output+'//'+'PMD_Cov_T.npy'
    np.save(PMD_Cov_T_str, PMD_Cov_T)
    
    PMD_Cov_T_0 = Cov_T_mean_0_ToT/(Nworkers+1)
    PMD_Cov_T_0_str = fpath_output+'//'+'PMD_Cov_T_0.npy'
    np.save(PMD_Cov_T_0_str, PMD_Cov_T_0)
    
    PMD_Cov_T_1 = Cov_T_mean_1_ToT/(Nworkers+1)
    PMD_Cov_T_1_str = fpath_output+'//'+'PMD_Cov_T_1.npy'
    np.save(PMD_Cov_T_1_str, PMD_Cov_T_1)
    
    PMD_Cov_T_2 = Cov_T_mean_2_ToT/(Nworkers+1)
    PMD_Cov_T_2_str = fpath_output+'//'+'PMD_Cov_T_2.npy'
    np.save(PMD_Cov_T_2_str, PMD_Cov_T_2)
    
    PMD_Cov_T_3 = Cov_T_mean_3_ToT/(Nworkers+1)
    PMD_Cov_T_3_str = fpath_output+'//'+'PMD_Cov_T_3.npy'
    np.save(PMD_Cov_T_3_str, PMD_Cov_T_3)
    
    PMD_autocorr_rg = autocorr_rg_mean_ToT/(Nworkers+1)
    PMD_autocorr_rg_str = fpath_output+'//'+'PMD_autocorr_rg.npy'
    np.save(PMD_autocorr_rg_str, PMD_autocorr_rg)
    
    PMD_autocorr_rT = autocorr_rT_mean_ToT/(Nworkers+1)
    PMD_autocorr_rT_str = fpath_output+'//'+'PMD_autocorr_rT.npy'
    np.save(PMD_autocorr_rT_str, PMD_autocorr_rT)
    
    PMD_autocorr_stand_rg = autocorr_stand_rg_mean_ToT/(Nworkers+1)
    PMD_autocorr_stand_rg_str = fpath_output+'//'+'PMD_autocorr_stand_rg.npy'
    np.save(PMD_autocorr_stand_rg_str, PMD_autocorr_stand_rg)
    
    PMD_autocorr_stand_rT = autocorr_stand_rT_mean_ToT/(Nworkers+1)
    PMD_autocorr_stand_rT_str = fpath_output+'//'+'PMD_autocorr_stand_rT.npy'
    np.save(PMD_autocorr_stand_rT_str, PMD_autocorr_stand_rT)
    
    PMD_standardized_rg = standardized_rg_mean_ToT/(Nworkers+1)
    PMD_standardized_rg_str = fpath_output+'//'+'PMD_standardized_rg.npy'
    np.save(PMD_standardized_rg_str, PMD_standardized_rg)
    
    PMD_standardized_rT = standardized_rT_mean_ToT/(Nworkers+1)
    PMD_standardized_rT_str = fpath_output+'//'+'PMD_standardized_rT.npy'
    np.save(PMD_standardized_rT_str, PMD_standardized_rT)
    
    LogLkeep_str = fpath_output+'//'+'LogLkeep.npy'
    np.save(LogLkeep_str, LogLkeep_TOT)
    
    alphakeep_str = fpath_output+'//'+'alphakeep.npy'
    np.save(alphakeep_str, alphakeep_TOT)

    Nnode_str = fpath_output+'//'+'Nnode.npy'
    np.save(Nnode_str, Nnode_TOT)

    NARg_str = fpath_output+'//'+'NARg.npy'
    np.save(NARg_str, NARg_TOT)
    
    NART_str = fpath_output+'//'+'NART.npy'
    np.save(NART_str, NART_TOT)

    ARgkeep_str = fpath_output+'//'+'ARgkeep.npy'
    np.save(ARgkeep_str, ARgkeep_TOT)
    
    ARTkeep_str = fpath_output+'//'+'ARTkeep.npy'
    np.save(ARTkeep_str, ARTkeep_TOT)
    
    sigma_g_str = fpath_output+'//'+'sigma_g.npy'
    np.save(sigma_g_str, sigmakeep_g_TOT)
    
    sigma_T_str = fpath_output+'//'+'sigma_T.npy'
    np.save(sigma_T_str, sigmakeep_T_TOT)
    
    rho_keep_str = fpath_output+'//'+'rho_keep.npy'
    np.save(rho_keep_str, rho_keep_TOT)
    
    Cov_g_keep_TOT_str = fpath_output+'//'+'Cov_g_keep_0.npy'
    np.save(Cov_g_keep_TOT_str, Cov_g_keep_0_TOT)
    
    Cov_g_keep_TOT_str = fpath_output+'//'+'Cov_g_keep_1.npy'
    np.save(Cov_g_keep_TOT_str, Cov_g_keep_1_TOT)
    
    Cov_g_keep_TOT_str = fpath_output+'//'+'Cov_g_keep_2.npy'
    np.save(Cov_g_keep_TOT_str, Cov_g_keep_2_TOT)
    
    Cov_g_keep_TOT_str = fpath_output+'//'+'Cov_g_keep_3.npy'
    np.save(Cov_g_keep_TOT_str, Cov_g_keep_3_TOT)
    
    
    Cov_T_keep_TOT_str = fpath_output+'//'+'Cov_T_keep_0.npy'
    np.save(Cov_T_keep_TOT_str, Cov_T_keep_0_TOT)
    
    Cov_T_keep_TOT_str = fpath_output+'//'+'Cov_T_keep_1.npy'
    np.save(Cov_T_keep_TOT_str, Cov_T_keep_1_TOT)
    
    Cov_T_keep_TOT_str = fpath_output+'//'+'Cov_T_keep_2.npy'
    np.save(Cov_T_keep_TOT_str, Cov_T_keep_2_TOT)
    
    Cov_T_keep_TOT_str = fpath_output+'//'+'Cov_T_keep_3.npy'
    np.save(Cov_T_keep_TOT_str, Cov_T_keep_3_TOT)

    print('The End')
    sys.stdout.flush()
    
MPI.Finalize
 