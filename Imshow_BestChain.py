# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:45:09 2020

@author: emadg
"""

import numpy as np
import matplotlib.pyplot as plt
from Chain2xz import Chain2xz
from scipy.interpolate import griddata
import alphashape
from Alpha_shape_mask import Alpha_shape_mask

def Imshow_BestChain(x1,x2,z1,z2,XnZn,CX,CZ,globals_par,Chain,fpath,figname,fignum):
    

    ind = np.argsort(Chain[:,0])[::-1]
    Chain_maxL = Chain[ind[0]].copy()
    
    [x, z, rho, alpha,  ZL, ARg, ART]= Chain2xz(Chain_maxL).copy()
    TrainPoints = np.column_stack((x,z)).copy()
    grid_model = griddata(TrainPoints, rho, (XnZn[:,0], XnZn[:,1]), method='linear', fill_value=0)
    hull = alphashape.alphashape(TrainPoints,alpha)
    DensityModel = Alpha_shape_mask(grid_model,hull,XnZn)
    DensityModel = DensityModel.reshape((CX,CZ),order='F')
    
    fig, ax = plt.subplots(gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 10))
    plt.rc('font', weight='bold')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    
    pos00 = ax.get_position() # get the original position
    pos00.x0 += 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    # pos00.top -= 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.7)
    
    rho_salt_min = globals_par[1,0]
    rho_base_max = globals_par[2,1]
    
    xbins = 5
    zbins = 5
    Xticklabels = np.around(np.linspace(x1,x2,xbins), 2)
    Zticklabels = np.around(np.linspace(z1,z2,zbins), 2)

    im00 = ax.imshow(DensityModel,interpolation='none',
           vmin=rho_salt_min, vmax=rho_base_max, extent=(0,1,1,0), aspect='auto', cmap='seismic')
    
    plt.locator_params(axis='y', nbins=zbins-1)
    plt.locator_params(axis='x', nbins=xbins-1)
 
            
    plt.xlabel("Distance (km)",fontweight="bold", fontsize = 20)
    plt.ylabel("Depth (km)",fontweight="bold", fontsize = 20)

    cbar_pos_density = fig.add_axes([0.1, 0.2, 0.03, 0.4]) 
    cbar_density = plt.colorbar(im00, ax=ax ,shrink=0.3, cax = cbar_pos_density,
                        orientation='vertical', ticklocation = 'left')
    cbar_density.ax.tick_params(labelsize=15)
    cbar_density.set_label(label = 'density contrast ($\mathregular{g/cm^{3}}$)', weight='bold')


    ax.plot(x,z,'ko')
    ax.axhline(y=ZL,color='black', ls='--', lw=2)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax.invert_yaxis()

    # plt.show()
    # plt.pause(0.00001)
    # plt.draw()
    #fig.savefig(fpath+'/'+figname+str(fignum)+'.png')
    fig.savefig(fpath+'/'+figname+str(fignum)+'.pdf')
    plt.close(fig)    # close the figure window