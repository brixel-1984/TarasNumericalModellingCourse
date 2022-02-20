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
eta = 1e6 # Viscosity, Pa*s
RHO = np.zeros([Ny,Nx]) # Array for model density

# Setup model density (RHO) 
for j in range(0,Nx):
    for i in range(0,Ny):
        rc = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)
        if rc > rp:
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

        # Define global index in algebraic space for vx, vy and p
        kvx = ((j)*Ny+i)*U
        kvy = kvx+1
        kpm = kvx+2

        # Equation for vx
        if j==0 or j==Nx-1 or i==0 or i==Ny-1 or i==1:
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

        # Equation for vy
        if j==0 or j==Nx-1 or j==1 or i==0 or i==Ny-1:
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
        
        # Equation for pressure
        if (j==0 or i==0 or (j==1 and i==1) or (j==1 and i==Ny-1) or (j==2 and i==1) or (j==Nx-1 and i==1) or (j==Nx-1 and i==Ny-1)):
            L[kpm,kpm] = 1
            R[kpm] = 0
        
        else:
            L[kpm,kvx-Ny*U] =-1/dx
            L[kpm,kvx]      = 1/dx
            L[kpm,kvy-3]    =-1/dy
            L[kpm,kvy]      = 1/dy

            R[kpm] = 0

# Solve global matrix
#S = la.solve(L, R)
S = spsolve(L,R)

# Reload solution S to geometrical array for omega W()
vX = np.zeros([Ny,Nx]) 
vY = np.zeros([Ny,Nx])
p = np.zeros([Ny,Nx])

for j in range(0,Nx):
    for i in range(0,Ny):

        # Define global index in algebraic space for vx, vy and pm
        kvx = ((j)*Ny+i)*U
        kvy = kvx+1
        kpm = kvx+2

        # Reload solution
        vX[i,j] = S[kvx]
        vY[i,j] = S[kvy]
        p[i,j] = S[kpm]

simTimeEnd = time.time() - simTimeStart

# Print results
print('')
print('______________')
print('Simulation overview')
print('______________')
print('Grid size: ', str(Nx), 'x', str(Ny))
print('Runtime: ', simTimeEnd, 'sec')
print('')

# Plot results
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axes = plt.subplots(2,2,figsize=(9,9),sharex=True, sharey=True)
ax1 = axes[0,0]
ax1.pcolormesh(x,y,RHO,shading='auto')
ax1.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax1.set_title('colormap of RHO')

ax2 = axes[0,1]
c2 = ax2.pcolormesh(x,y,p, cmap='RdBu',shading='auto')
ax2.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax2.set_title('colormap of W')
divAx2 = make_axes_locatable(ax2)
cAx2 = divAx2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c2, ax=ax2, cax=cAx2)

ax4 = axes[1,0]
c4 = ax4.pcolormesh(x,y,vX,cmap='RdBu',shading='auto')
ax4.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax4.set_title('colormap of vX')
divAx4 = make_axes_locatable(ax4)
cAx4 = divAx4.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c4, ax=ax4, cax=cAx4)

ax5 = axes[1,1]
c5 = ax5.pcolormesh(x,y,vY,cmap='RdBu',shading='auto')
ax5.quiver(x[2:Nx:5],y[2:Ny:5],vX[2:Ny:5,2:Nx:5],vY[2:Ny:5,2:Nx:5],color='k')
ax5.set_title('colormap of vY')
divAx5 = make_axes_locatable(ax5)
cAx5 = divAx5.append_axes("right", size="5%", pad=0.05)
fig.colorbar(c5, ax=ax5, cax=cAx5)

for ax in axes.flatten():
    ax.set_aspect('equal')

fig.tight_layout()
plt.show()
