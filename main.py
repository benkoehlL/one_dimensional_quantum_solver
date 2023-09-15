import os, sys
sys.path.append("./")
from libs.quantum_solver import System

## read data
# potential
f = open(os.getcwd()+"/potential/potential.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n') if len(i.split(',')) == 2]
V = [float(i.split(',')[1]) for i in data.split('\n') if len(i.split(',')) == 2]
potential = {'x': x, 'V': V}

# state
f = open(os.getcwd()+"/states/input_state.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n') if len(i.split(',')) == 2]
Psi = [complex(i.split(',')[1]) for i in data.split('\n') if len(i.split(',')) == 2]

'''
# Test case

x = [float(i.split(',')[0]) for i in data.split('\n')]
Psi = [np.cos(2*np.pi*t/(len(x))) for t in range(len(x))]
'''

state = {'x' : x, 'Psi': Psi}
s = System(state, potential)

final_state_file = str(os.getcwd())+"/states/final_state_Visscher.dat"
s.export_final_state(10, 0.00001, final_state_file)
f = open(final_state_file, "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n') if len(i.split(',')) == 2]
Psi = [complex(i.split(',')[1]) for i in data.split('\n') if len(i.split(',')) == 2]
s.state = {'x' : x, 'Psi': Psi}
s = System(state,potential)
s.plot_x_prob()