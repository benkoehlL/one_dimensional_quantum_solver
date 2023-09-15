import os, sys
import matplotlib.pyplot as plt
sys.path.append("./")
from libs.quantum_solver import System

if __name__ == "__main__":
    ## read data from the supplied files
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
    state = {'x' : x, 'Psi': Psi}

    ## solve time-dependent Schrödinger equation
    # stability condition for Schrödinger wave equations dt < (dx)^2/2 
    h = (x[1]-x[0])**2/2 
    s = System(state, potential)

    for method in ['RK4', 'Visscher']:
        final_state_file = str(os.getcwd())+"/states/final_state_"+str(method)+".dat"
        #s.export_final_state(T=10, h=h/2, file_name=final_state_file, method=method)
        f = open(final_state_file, "r")
        data = f.read()
        x = [float(i.split(',')[0]) for i in data.split('\n') if len(i.split(',')) == 2]
        Psi = [complex(i.split(',')[1]) for i in data.split('\n') if len(i.split(',')) == 2]
        s.state = {'x' : x, 'Psi': Psi}
        
        # plot the final state
        s.plot_x_prob(s.state, label=method)

    # show plot with legend
    plt.legend(loc="upper left")
    plt.show()