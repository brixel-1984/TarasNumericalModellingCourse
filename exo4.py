# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
import scipy.sparse.linalg
from scipy import linalg as la 
import time

simTimeStart = time.time()

# Grid setup
xsize = 1e5
ysize = 1e5
Nx = 41
Ny = 31
dx = xsize/(Nx-1)
dy = ysize/(Ny-1)
x = np.linspace(0,xsize,Nx)
y = np.linspace(0,ysize,Ny)

rp = 2e4        # Anomaly radius, m
gravY = 9.81    # Gravity (vertical component), m/s^2
eta = 1e21      # Viscosity, Pa*s

# Setup model density (RHO) across the grid
RHO = np.zeros([Ny,Nx])

for j in range(0,Nx):
    for i in range(0,Ny):

        r_current = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)

        if r_current > rp:
            RHO[i,j] = 3300

        else:
            RHO[i,j] = 3200

# Compose global matrix
U = 3
N = Nx*Ny*U
L = sp.csr_matrix((N,N), dtype='float64')
R = np.zeros(N)

for j in range(0,Nx):
    for i in range(0,Ny):

        kvx = ((j)*Ny+i)*U+1
        kvy = kvx+1
        kpm = kvx+2

        if j==1 or j==Nx-1 or i==1 or i==Ny-1 or i==2:
            L[kvx,kvx] = 1
            R[kvx] = 0
        
        else:
            L[kvx,kvx-Ny*U] = eta*(1/dx**2)
            L[kvx,kvx+Ny*U] = eta*(1/dx**2)
            L[kvx,kvx-U]    = eta*(1/dy**2)
            L[kvx,kvx+U]    = eta*(1/dy**2)
            L[kvx,kvx]      = eta*(-2/dx**2-2/dy**2)

            L[kvx,kpm]      = 1/dx
            L[kvx,kpm+Ny*U] =-1/dx

            R[kvx] = 0 

        if j==1 or j==Nx-1 or i==1 or i==Ny:
            L[kvy,kvy] = 1
            R[kvy] = 0
        
        else:
            L[kvy,kvy-Ny*U] = eta*(1/dx**2)
            L[kvy,kvy+Ny*U] = eta*(1/dx**2)
            L[kvy,kvy-U]    = eta*(1/dy**2)
            L[kvy,kvy+U]    = eta*(1/dy**2)
            L[kvy,kvy]      = eta*(-2/dx**2-2/dy**2)

            L[kvy,kpm]      = 1/dy
            L[kvy,kpm+U]    =-1/dy

            R[kvy] = -((RHO[i,j]+RHO[i,j])/2)*gravY
        
        if j==1 or i==1 or (j==2 and i==2) or (j==2 and i==Ny-1) or (j==3 and i==2) or (j==Nx-1 and i==2) or (j==Nx-1 and i== Ny-1):
            L[kpm-1,kpm-1] = 1
            #R[kpm] = 0
        
        #else:
            #L[kpm,kvx-Ny*U] =-1/dx
            #L[kpm,kvx]      = 1/dx
            #L[kpm,kvy-3]    =-1/dy
            #L[kpm,kvy]      = 1/dy

