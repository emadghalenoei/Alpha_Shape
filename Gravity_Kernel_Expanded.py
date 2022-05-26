# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:15:52 2020

@author: emadg
"""

import numpy as np
#from numba import jit
#@njit      # or @jit(nopython=True)
#@jit
def Gravity_Kernel_Expanded(X,Z,xs):
    #model space
    dx=abs(X[0,1]-X[0,0]) # Prisms dimension along x
    dz=abs(Z[1,0]-Z[0,0]) # thickness of prisms
    
    cGrav = 6.67408e-11
    
    CX = len(X[0])   # Prisms along x
    CZ = len(Z)      # Prisms along z
    CTOT=CX*CZ       # Total number of prisms

    A = np.ones((len(xs), CTOT))
    
    Ximat = np.matmul(np.diag(xs),A)
    Xjmat = np.matmul(A,np.diag(X.flatten('F')))
    Zjmat = np.matmul(A,np.diag(Z.flatten('F')))
    shiftx = (dx/2)*A
    shiftz = (dz/2)*A
    
    Xplus = Ximat-Xjmat+shiftx
    Xminus=Ximat-Xjmat-shiftx
    Zplus=Zjmat+shiftz
    Zminus=Zjmat-shiftz
    Xplus2= np.power(Xplus,2)
    Xminus2=np.power(Xminus,2)
    Zplus2= np.power(Zplus,2)
    Zminus2=  np.power(Zminus,2)
    
    # Distance and angles
    r1 = np.power(Zminus2+Xplus2,0.5)
    r2 = np.power(Zplus2+Xplus2,0.5)
    r3 = np.power(Zminus2+Xminus2,0.5)
    r4 = np.power(Zplus2+Xminus2,0.5)
    
    teta1 = np.arctan2(Xplus,Zminus)
    teta2 = np.arctan2(Xplus,Zplus)
    teta3 = np.arctan2(Xminus,Zminus)
    teta4 = np.arctan2(Xminus,Zplus)
    
    r2r3 = np.multiply(r2,r3)
    r1r4 = np.multiply(r1,r4)

    at = np.log(np.divide(r2r3,r1r4))
    term1 = np.multiply(Xplus,at)
    term2 = np.multiply(2*shiftx,np.log(np.divide(r4,r3)))
    term3 = np.multiply(Zplus,teta4-teta2)
    term4 = np.multiply(Zminus,teta3-teta1)
   
    A = 2*cGrav*(term1+term2-term3+term4)
    return A

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   