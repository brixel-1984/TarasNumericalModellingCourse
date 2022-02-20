# Taras' Numerical Modelling Course (the Python Way)
Conversion of Taras Matlab models to Python 3

Here you'll find Python 3 versions of some of the models originally developed in Matlab as part of Taras Gerya's introduction to Numerical Modelling.

The aim is to offer an open source alternative to interested students, and compare computation times and code readability between Python and Matlab.

This repository is a work in progress.

## Global indexing in Python
Note that global indexing relations have to adapted because in Python, indexing starts at 0.

So in Matlab, 

<img src="https://latex.codecogs.com/svg.image?gk&space;=&space;(j-1)\times&space;Ny&plus;i" title="gk = (j-1)\times Ny+i" /> 

becomes in Python:

<img src="https://latex.codecogs.com/svg.image?gk&space;=&space;(j)\times&space;Ny&plus;i" title="gk = (j)\times Ny+i" />

And similarly,

<img src="https://latex.codecogs.com/svg.image?gk&space;=&space;((j-1)\times&space;Ny&plus;i-1)\times&space;U&plus;1" title="gk = ((j-1)\times Ny+i-1)\times U+1" />

becomes:

<img src="https://latex.codecogs.com/svg.image?gk&space;=&space;((j)\times&space;Ny&plus;i)\times&space;U" title="gk = ((j)\times Ny+i)\times U" />

## Boundary Conditions in Python
