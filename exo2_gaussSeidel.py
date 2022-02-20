# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy import linalg as la

import time

simTimeStart = time.time()

# Spatial discretization
xsize = 2e7
ysize = 2e7
Nx = 101
Ny = 101
dx = xsize/(Nx-1)
dy = ysize/(Ny-1)
x = np.linspace(0,xsize,Nx)
y = np.linspace(0,ysize,Ny)

# Setup model density (RHO) across the grid
rp = 6e6 # planet radius, m
rBC = xsize/2-dx/2 # BC radius, m
RHO = np.zeros([Ny,Nx])

for j in range(0,Nx):
    for i in range(0,Ny):
        r_current = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)
        if r_current > rp:
            RHO[i,j] = 0
        else:
            RHO[i,j] = 5000

# Iterative (Gauss-Seidel) solver
theta = 1.5
iterMax = int(1e4)
tol = 1e-5
N = Nx*Ny
PHI = np.zeros([Ny,Nx], dtype='float64') # Initial cond
R   = np.zeros([Ny,Nx], dtype='float64') # Initial cond
deltaR = np.zeros([Ny,Nx], dtype='float64')
relErrorSim = []

count = 0
for k in range(0,iterMax):
    for j in range(0,Nx):
        for i in range(0,Ny):               
            r_current = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)             
            if r_current > rBC:
                R[i,j] = 0                     
            else:
                R[i,j] = 2/3*4*np.pi*6.672e-11*RHO[i,j]
                deltaR[i,j] = R[i,j]-((PHI[i,j+1]-2*PHI[i,j]+PHI[i,j-1])/dx**2+(PHI[i+1,j]-2*PHI[i,j]+PHI[i-1,j])/dy**2)
                PHI[i,j] = PHI[i,j]+theta*(deltaR[i,j]/(-2/dy**2-2/dx**2))
    count += 1
    relError = np.linalg.norm(deltaR)/np.linalg.norm(R)
    relErrorSim.append(relError)

    if relError<tol:
        break

# Compute vector field
gX = np.zeros((Ny,Nx))
gY = np.zeros((Ny,Nx))

for j in range(0,Nx):
    for i in range(0,Ny):
        r_current = np.sqrt((x[j]-xsize/2)**2+(y[i]-ysize/2)**2)
        if r_current > rBC:
            gX[i,j] = 0
            gY[i,j] = 0
        else:
            gX[i,j] = -(PHI[i,j+1]-PHI[i,j-1])/2/dx
            gY[i,j] = -(PHI[i+1,j]-PHI[i-1,j])/2/dy

simTimeEnd = time.time() - simTimeStart

# Print results
print('')
print('_________________________')
print('Simulation overview')
print('_________________________')
print('Grid size: ', str(Nx), 'x', str(Ny))
print('Runtime: ', round(simTimeEnd,0), 'sec')
print('Number of iteration: ', count)
print('Maximum number of iteration: ', iterMax)
print('Relative error: ', round(relError,10))
print('Solver: Gauss-Seidel')
print('_________________________')
print('')

fig, axes = plt.subplots(2,2,figsize=(9,9))
ax1 = axes[0,0]
ax1.pcolormesh(x,y,RHO)
ax1.quiver(x[0:Ny:5],y[0:Nx:5],gX[0:Ny:5,0:Nx:5],gY[0:Ny:5,0:Nx:5],color='w')
ax1.set_title('colormap of RHO')

ax2 = axes[0,1]
c2  = ax2.pcolormesh(x,y,PHI,cmap='RdBu')
ax2.set_title('colormap of PHI')
fig.colorbar(c2, ax=ax2)

ax3 = axes[1,0]
c3  = ax3.pcolormesh(x,y,gX,cmap='RdBu')
ax3.set_title('colormap of gX')
fig.colorbar(c3, ax=ax3)

ax4 = axes[1,1]
c4  = ax4.pcolormesh(x,y,gY,cmap='RdBu')
ax4.set_title('colormap of gY')
fig.colorbar(c4, ax=ax4)

for ax in axes.flatten():
    ax.set_aspect('equal')

fig.tight_layout()
plt.show()
