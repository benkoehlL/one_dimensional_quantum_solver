import os, sys
import matplotlib.pyplot as plt
sys.path.append("./")
from libs.quantum_solver import System

if __name__ == "__main__":
    ## get paths to the supplied files for potential and initial state
    potential_path = str(os.getcwd())+"/data/potential.dat"
    init_state_path = str(os.getcwd())+"/data/input_state.dat"
    ## solve time-dependent Schrödinger equation
    for method in ['Visscher', 'RK4']:
        s = System(init_state_file_path=init_state_path, 
                   potential_file_path=potential_path)
        # stability condition for Schrödinger wave equations dt < (dx)^2/D 
        h = (s.state['x'][1]-s.state['x'][0])**2
        final_state_file = str(os.getcwd())+"/data/final_state_"+str(method)+".dat"
        s.export_final_state(T=10, h=h/2, file_name=final_state_file, method=method)
        
        # get final state for plotting
        s = System(init_state_file_path=final_state_file, 
                   potential_file_path=potential_path)
        
        # plot the final state
        s.plot_x_prob(s.state, label=method)

    # show plot with legend
    plt.legend(loc="upper left")
    plt.show()