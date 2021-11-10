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
W = np.zeros([Ny,Nx], dtype='float64')
PSI = np.zeros([Ny,Nx], dtype='float64')

# Setup model density (RHO) across the grid
rp = 2e4
gravY = 9.81
eta = 1e21

RHO = np.zeros([Ny,Nx])
for j in range(0,Nx):
    for i in range(0,Ny):

        r_current = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)

        if r_current > rp:
            RHO[i,j] = 3300

        else:
            RHO[i,j] = 3200

# Compose global matrix (Vorticity)
N  = Nx*Ny
Lw = sp.csr_matrix((N,N)).toarray()
Rw = np.zeros(N)

for j in range(0,Nx):
    for i in range(0,Ny):

        gk = (j)*Ny+i

        if j==1 or j==Nx-1 or i==1 or i==Ny-1:
            Lw[gk,gk] = 1
            Rw[gk] = 0
        
        else:
            Lw[gk,gk-Ny] = 1/dx**2
            Lw[gk,gk-1]  = 1/dy**2
            Lw[gk,gk]    =-2/dx**2-2/dy**2
            Lw[gk,gk+1]  = 1/dy**2
            Lw[gk,gk+Ny] = 1/dx**2

            Rw[gk] = ((RHO[i,j+1]-RHO[i,j-1])/(2*dx))*(gravY/eta)

# Solve global matrix and reload 
# the algebraic solution to the geometrical array (PHI)
Sw = la.solve(Lw, Rw)

for j in range(0,Nx):
    for i in range(0,Ny):

        gk = (j)*Ny+i
        W[i,j] = Sw[gk]

# Compose global matrix (PSI)
Lpsi = sp.csr_matrix((N,N)).toarray()
Rpsi = np.zeros(N)

for j in range(0,Nx):
    for i in range(0,Ny):

        gk = (j)*Ny+i

        if j==1 or j==Nx-1 or i==1 or i==Ny-1:
            Lpsi[gk,gk] = 1
            Rpsi[gk] = 0
        
        else:
            Lpsi[gk,gk-Ny] = 1/dx**2
            Lpsi[gk,gk-1]  = 1/dy**2
            Lpsi[gk,gk]    =-2/dx**2-2/dy**2
            Lpsi[gk,gk+1]  = 1/dy**2
            Lpsi[gk,gk+Ny] = 1/dx**2

            Rpsi[gk] = Sw[gk]

# Solve global matrix and reload 
# the algebraic solution to the geometrical array (PHI)
Spsi = la.solve(Lpsi, Rpsi)

for j in range(0,Nx):
    for i in range(0,Ny):

        gk = (j)*Ny+i
        PSI[i,j] = Spsi[gk]

simTimeEnd = time.time() - simTimeStart

# Print results
print('')
print('______________')
print('Simulation overview')
print('______________')
print('Grid size: ', str(Nx), 'x', str(Ny))
print('Runtime: ', simTimeEnd, 'sec')
print('')

# Define X and Y component of the velocity vector
vX = np.zeros([Ny,Nx], dtype='float64')
vY = np.zeros([Ny,Nx], dtype='float64')

vX[0,1:Nx-1]  = vX[1,1:Nx-1]    # Boundary
vX[Ny-1,1:Nx-2] = vX[Ny-2,1:Nx-2] # Boundary
vY[1:Ny-2,0]  = vY[1:Ny-2,1]    # Boundary
vY[1:Ny-2,Nx-1] = vY[1:Ny-2,Nx-2] # Boundary

for j in range(0,Nx):
    for i in range(0,Ny):
        if j==1 or j==Nx-1 or i==1 or i==Ny-1:
            vX[i,j] = 0
            vY[i,j] = 0
        
        else:
            vY[i,j] =  (PSI[i,j+1]-PSI[i,j-1])/(2*dx)
            vX[i,j] = -(PSI[i+1,j]-PSI[i-1,j])/(2*dy)

# Plot results
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axes = plt.subplots(2,3,figsize=(9,9),sharex=True, sharey=True)
ax1 = axes[0,0]
ax1.pcolor(x,y,RHO)
ax1.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax1.set_title('colormap of RHO')

ax2 = axes[0,1]
c2 = ax2.pcolor(x,y,W, cmap='RdBu')
ax2.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax2.set_title('colormap of W')
divAx2 = make_axes_locatable(ax2)
cAx2 = divAx2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c2, ax=ax2, cax=cAx2)

ax3 = axes[0,2]
c3 = ax3.pcolor(x,y,PSI,cmap='RdBu')
ax3.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax3.set_title('colormap of PSI')
divAx3 = make_axes_locatable(ax3)
cAx3 = divAx3.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c3, ax=ax3, cax=cAx3)

ax4 = axes[1,0]
c4 = ax4.pcolor(x,y,vX,cmap='RdBu')
ax4.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax4.set_title('colormap of vX')
divAx4 = make_axes_locatable(ax4)
cAx4 = divAx4.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c4, ax=ax4, cax=cAx4)

ax5 = axes[1,1]
c5 = ax5.pcolor(x,y,vY,cmap='RdBu')
ax5.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax5.set_title('colormap of vY')
divAx5 = make_axes_locatable(ax5)
cAx5 = divAx5.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c5, ax=ax5, cax=cAx5)

for ax in axes.flatten():
    ax.set_aspect('equal')