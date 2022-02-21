# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy import linalg as la 
from scipy.sparse.linalg import spsolve 

import time 

simTimeStart = time.time()

# Spatial discretization
xsize = 1e5
ysize = 1e5
Nx = 41 
Ny = 31
dx = xsize/(Nx-1)
dy = ysize/(Ny-1)
x = np.linspace(0,xsize,Nx)
y = np.linspace(0,ysize,Ny)

rp = 2e4 # Anomaly radius, m 
gravY = 9.81 # Gravity (vertical component), m/s^2
RHO = np.zeros([Ny,Nx]) # Array for model density
ETA = np.zeros([Ny,Nx]) # Array for model viscosity (sigma points)

# Setup model density and viscosity
for j in range(0,Nx):
    for i in range(0,Ny):
        rc = np.sqrt((x[j] - xsize/2)**2 + (y[i] - ysize/2)**2)
        if rc > rp:
            RHO[i,j] = 3300
            ETA[i,j] = 1e21
        else:
            RHO[i,j] = 3200
            ETA[i,j] = 1e19

# Introducing scaled pressure
pscale = 1e21/dx

# Initial model boundary conditions (BC)
BC = 1 # set BC=1 for free slip and BC=1/3 for no slip

# Compose global matrix
U = 3 # Number of unknowns
N = Nx*Ny*U # Global number of unknowns
L = sp.csr_matrix((N,N), dtype='float64') # Matrix of coefficients
R = np.zeros(N) # Vector of unknowns

for j in range(0,Nx):
    for i in range(0,Ny):

        # Define global index in algebraic space for vx, vy and p
        kvx = ((j)*Ny+i)*U
        kvy = kvx+1
        kpm = kvx+2

        # Equation for vx
        if j==0 or j==Nx-1 or i==0 or i==Ny-1 or i==1:
            # Left and Right model boundary
            if j==0 or j==Nx-1 or i==0:
                L[kvx,kvx] = 1
                R[kvx] = 0
            # Top model boundary
            elif i==1:
                L[kvx,kvx] = 1
                L[kvx,kvx+U] = -1
                R[kvx] = 0
            # Bottom model boundary
            elif i==Ny-1:
                L[kvx,kvx] = 1
                L[kvx,kvx-U] = -1
                R[kvx] = 0

        # Internal domain (x-Stokes equation)
        else:
            # Define ETA1 and ETA2
            ETA1 = ETA[i-1,j]
            ETA2 = ETA[i,j]

            # Define ETAPxx_1 and ETAPxx_2 as the harmonic mean of neighbor cells
            ETAPxx_1 = 4/(1/ETA[i,j]+1/ETA[i,j-1]+1/ETA[i-1,j-1]+1/ETA[i-1,j])
            ETAPxx_2 = 4/(1/ETA[i,j]+1/ETA[i-1,j]+1/ETA[i,j+1]+1/ETA[i-1,j+1])

            # Left part (d_sigma'_xx/dx) 
            # vx
            L[kvx,kvx-Ny*U] = ETAPxx_1*(2/dx**2) # vx_1
            L[kvx,kvx-U]    = ETA[i-1,j]*(1/dy**2) # vx_2
            L[kvx,kvx]      = -ETAPxx_2*(2/dx**2) - ETAPxx_1*(2/dx**2) - ETA[i,j]*(1/dy**2) - ETA[i-1,j]*(1/dy**2) # vx3
            L[kvx,kvx+U]    = ETA[i,j]*(1/dy**2) # vx_4
            L[kvx,kvx+Ny*U] = ETAPxx_2*(2/dx**2) # vx_5

            # Left part (d_sigma'_xy/dy)
            # vy
            L[kvx,kvy-U]      = ETA[i-1,j]*(1/(dx*dy)) # vy_1
            L[kvx,kvy]        = ETA[i,j]*(1/(dx*dy)) # vy_2 
            L[kvx,kvy+Ny*U-U] = ETA[i-1,j]*(1/(dx*dy)) # vy_3
            L[kvx,kvy+Ny*U]   = ETA[i,j]*(1/(dx*dy)) # vy_4

            # Left part (dp/dx)
            L[kvx,kpm]      = pscale/dx
            L[kvx,kpm+Ny*U] = -pscale/dx

            # Right part
            R[kvx] = 0

        # Equation for vy
        if j==0 or j==Nx-1 or j==1 or i==0 or i==Ny-1:
            # Top and bottom model boundaries, and fictiious points
            if j==0 or i==0 or i==Ny-1:
                L[kvy,kvy] = 1
                R[kvy] = 0
            # Left vertical model boundary
            elif j==1:
                L[kvy,kvy] = 1
                L[kvy,kvy+Ny*U] = -1
                R[kvy] = 0
            # Right vertical model boundary
            elif j==Nx-1:
                L[kvy,kvy] = 1
                L[kvy,kvy-Ny*U] = -1
                R[kvy] = 0

        # Internal domain (y-Stokes equation)
        else:
            # Define ETAPyy_1 and ETAPyy_2 as the harmonic mean of neighbor cells
            ETAPyy_1 = 4/(1/ETA[i,j]+1/ETA[i,j-1]+1/ETA[i-1,j-1]+1/ETA[i-1,j])
            ETAPyy_2 = 4/(1/ETA[i,j]+1/ETA[i+1,j]+1/ETA[i+1,j-1]+1/ETA[i,j-1])

            # Left part (d_sigma'_yy/dy) 
            # vy
            L[kvy,kvy-Ny*U] = ETA[i,j-1]*(1/dx**2) # vy_1 = ETA_1*1/dx^2
            L[kvy,kvy-U]    = ETAPyy_1*(2/dy**2) # vy_2 = ETAP_yy1*2/dy^2
            L[kvy,kvy]      =-ETAPyy_2*(2/dy**2)-ETAPyy_1*(2/dy**2)-ETA[i,j]*(1/dx**2)-ETA[i,j-1]*(1/dx**2); # vy_3 = -ETAPyy_2*2/dy^2-ETAPyy_1*2/dy^2-ETA_2*1/dx^2-ETA_1*1/dx^2
            L[kvy,kvy+U]    = ETAPyy_2*(2/dy**2) # vy_4 = ETAPyy_2*2/dy^2
            L[kvy,kvy+Ny*U] = ETA[i,j]*(1/dx**2) # vy_5 = ETA_2*1/dx^2

            # Left part (d_sigma'_yx/dy)
            # vx
            L[kvy,kvx-Ny*U]   = ETA[i,j-1]*(1/(dy*dx)) # vx_1 =  ETA_1*1/dy*dx
            L[kvy,kvx-Ny*U+U] =-ETA[i,j-1]*(1/(dy*dx)) # vx_2 = -ETA_1*1/dy*dx
            L[kvy,kvx]        =-ETA[i,j]*(1/(dy*dx)) # vx_3 = -ETA_2*1/dy*dx
            L[kvy,kvx+U]      = ETA[i,j]*(1/(dy*dx)) # vx_4 =  ETA_2*1/dy*dx

            # Left part (dp/dy)
            L[kvy,kpm]=pscale/dy # P1
            L[kvy,kpm+U]=-pscale/dy 

            # Right part
            R[kvy]=-((RHO[i,j]+RHO[i,j-1])/2)*gravY; 

        # Equation for pressure
        if j==0 or i==0 or (j==1 and i==1) or (j==1 and i==Ny-1) or (j==Nx-1 and i==1) or (j==Nx-1 and i==Ny-1) or (i==1 and j==2):
            # Fictious points
            if j==0 or i==0:
                L[kpm,kpm] = 1*pscale
                R[kpm] = 0
            # Local point BC
            elif i==1 and j==2:
                L[kpm,kpm] = 1*pscale # Left part (!=b/c of scaled pressure scheme)
                R[kpm] = 1e9 # Right part 
            # Top left corner
            elif j==1 and i==1:
                L[kpm,kpm] = 1*pscale 
                L[kpm,kpm+Ny*3] = -1*pscale 
                R[kpm] = 0
            # Bottom left corner
            elif j==1 and i==Ny-1:
                L[kpm,kpm] = 1*pscale 
                L[kpm,kpm+Ny*U] = -1*pscale 
                R[kpm] = 0
            # Bottom right corner
            elif j==Nx-1 and i==Ny-1:
                L[kpm,kpm] = 1*pscale 
                L[kpm,kpm-Ny*U] = -1*pscale 
                R[kpm] = 0
            # Top right corner
            elif j==Ny-1 and i==1:
                L[kpm,kpm] = 1*pscale 
                L[kpm,kpm-Ny*U] = -1*pscale 
                R[kpm] = 0

        # Internal points for continuity equation
        else:
            # Left part
            L[kpm,kvx-Ny*U] =-1/dx; # vx_1
            L[kpm,kvx]      = 1/dx; # vx_2
            L[kpm,kvy-3]    =-1/dy; # vy_1
            L[kpm,kvy]      = 1/dy; # vy_2
            
            # Right part
            R[kpm]          = 0;            

# Solve global matrix
S = spsolve(L,R)

simTimeEnd = time.time() - simTimeStart
