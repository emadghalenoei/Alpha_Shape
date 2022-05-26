from mpi4py import MPI
import os
import sys
import shutil
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
status = MPI.Status()
Nworkers = comm.Get_size()-1 # No. MCMC chains or MPI threads


file_name = '2022_01_28-05_37_44_PM'

fpath_Restart = os.getcwd()+'//'+file_name+'//'+'Restart'

fpath_output = os.getcwd()+'//'+file_name+'//'+'Output'
# if rank == 0:
#     if os.path.exists(fpath_output) and os.path.isdir(fpath_output):
#         shutil.rmtree(fpath_output)
#     os.mkdir(fpath_output)


ChainHistory_str = fpath_Restart+'//'+'ChainHistory.npy'

globals_par = np.load(fpath_Restart+'//'+'globals_par.npy')
Kmin = int(globals_par[0,0])
Kmax = int(globals_par[0,1])
KminAR = int(globals_par[3,0])
KmaxAR = int(globals_par[3,1])

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


print('rank: ', rank, ', number of chains: ', Chainsplit.shape[0])
sys.stdout.flush()

LogLMax = -9999999

ChainMax = np.zeros(Ncol)


for ichain in np.arange(Nchain):
    
        Chain = Chainsplit[ichain,:].copy()
        LogL = Chain[0].copy()
        Nnode = Chain[1].copy()
        
        if LogL>=LogLMax and Nnode>=20 and Nnode<=21:
            LogLMax = LogL
            ChainMax = Chain.copy()
        
        
       
        
        if ichain % 2000 == 0:
            print('rank: ', rank, ', ichain: ', ichain, ' from ', Nchain)
#             print(Nnode)
            sys.stdout.flush()

#print('rank: ', rank, ', LogLMax: ', LogLMax,', k: ', 10)
#sys.stdout.flush()        

# gather all local arrays on process root, will return a list of numpy arrays
ChainMax_TOT = comm.gather(ChainMax, root=0)

################################################################################################################
### save to desk

if rank == 0:
    
    # turn the list of arrays into a single array
    ChainMax_TOT = np.concatenate(ChainMax_TOT)
    ChainMax_TOT = ChainMax_TOT.reshape((Nworkers+1, Ncol))

    
    ChainMax_str = fpath_output+'//'+'ChainMax.npy'
    np.save(ChainMax_str, ChainMax_TOT)

    print('The End')
    sys.stdout.flush()
    
MPI.Finalize
 