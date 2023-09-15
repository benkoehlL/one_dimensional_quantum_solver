# One-dimensional quantum solver
A project where a one-dimensional quantum system is
evolved through time by solving the time-dependent
Schrödinger equation for a given discretization
of the potential and the wave function using either
the Visscher [1] or a RK4 integration method.

[1] P.B. Visscher: "A fast explicit algorithm for the time dependent 
    Schrödinger equation", Computers in  Physics 5 (6), 596-598 (1991)

## creating the environment/installing all necessary packages
To be able to use the code, you can create a conda environment
from the .yml file from the project via the terminal command

$ conda env create -f environment.yml

or in a venv with

$ pip install -r requirements.txt

As my code only uses numpy and matplotlib, I do not think 
that much setup time will be necessary.