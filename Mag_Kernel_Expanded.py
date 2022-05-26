# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:22:55 2020

@author: emadg
"""
import numpy as np
import math

def Mag_Kernel_Expanded(X,Z,xs,I,Azimuth):
    
    #model space
    dx=abs(X[0,1]-X[0,0]) # Prisms dimension along x
    dz=abs(Z[1,0]-Z[0,0]) # thickness of prisms
    
    I= I * (math.pi/180)    #inclination
    Beta = (90-Azimuth)*math.pi/180   # Angle between East and profile (in telford p90, Beta is angle between x(East) and x'(profile))
    
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
    
    
    teta1=np.arctan2(Zminus,Xplus)
    teta2=np.arctan2(Zplus,Xplus)
    teta3=np.arctan2(Zminus,Xminus)
    teta4=np.arctan2(Zplus,Xminus)
    
    r2r3 = np.multiply(r2,r3)
    r1r4 = np.multiply(r1,r4)

    at = np.log(np.divide(r2r3,r1r4))
    bt=teta1-teta2-teta3+teta4
    ct=(math.cos(I)**2)*(math.sin(Beta)**2)-math.sin(I)**2
    A=math.sin(2*I)*math.sin(Beta)*at+ct*bt
    return A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    