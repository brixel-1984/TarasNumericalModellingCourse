# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
import scipy.sparse.linalg
from scipy import linalg as la

# Grid setup
xsize = 1
Nx = 101
dx = xsize/(Nx-1)
x = np.linspace(0,xsize,Nx)
PHI = np.zeros(Nx)

# Compose global matrix
L = sp.csr_matrix((Nx,Nx)).toarray()
R = np.zeros(Nx)

for i in range(0,Nx):
 
    if i==0 or i==Nx-1: 
        L[i,i] = 1
        R[i]  = 0 
 
    else:
        L[i,i-1] = 1/dx**2
        L[i,i]   =-2/dx**2
        L[i,i+1] = 1/dx**2

        R[i] = 2*x[i]**2-(x[i]/2)+np.exp(x[i])

# Solve global matrix and reload 
# the algebraic solution to the geometrical array (PHI)
# S = np.linalg.inv(L).dot(R)   # Compute the inverse of the coefficient matrix and multiply it to to the RHS vector
S = la.solve(L, R)              # LU factorization (best practice)

for i in range(0,Nx):
    PHI[i] = S[i]

# Analytical solution
# (computed on Mathematica)
Nxa = 1001
xa = np.linspace(0,xsize,Nxa)
PHIa = np.zeros(Nx, dtype='float64')
PHIa = xa**4/6-xa**3/12+np.exp(xa)+((+12-12*np.exp(xsize)+xsize**3-2*xsize**4)/12*xsize)*xa-1

# Plot results
fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.scatter(x,PHI,marker='o',color='none',edgecolor='#1f77b4',label='Numerical solution')
ax.plot(xa,PHIa,'-', lw=1.5,color='orange',label='Analytical solution')
ax.set_xlim(0,1)
ax.set_ylim(-0.25,0)
ax.set_xlabel('Model domain, x [m]')
ax.set_ylabel('Potential field')
ax.set_title(r'$\mathrm{\frac{\partial^2\phi}{\partial x^2}=2x^2-\frac{x}{2}+e^x}$',fontsize=16)
ax.legend(loc=1,fancybox=False)