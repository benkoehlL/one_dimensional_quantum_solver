import os
import numpy as np
import matplotlib.pyplot as plt

class system():
    def __init__(self, init_state, potential):
        self.state = init_state
        # check matching x axis

        self.potential = potential
        self.check_x()
        self.get_norm()
        
    def check_x(self):
        ## check whether potential and state x values are the same
        if(len(self.potential['x']) != len(self.state['x'])):
            raise ValueError("Potential and state have different length")
        check = False
        for i, x in enumerate(state['x']):
            if(x not in potential['x']):
                check = True
                print(x)
                break
        if(check):
            raise ValueError("Potential and state have non-matching x range")

    def plot_potential(self):
        plt.plot(self.potential['x'],self.potential['V'])
        plt.ylabel("V(x)")
        plt.xlabel("x")
        plt.show() 

    def plot_x_prob(self):
        plt.plot(self.state['x'],[np.sqrt(i.real**2 + i.imag**2)/self.norm for i in self.state['Psi']])
        plt.ylabel("Prob(x)")
        plt.xlabel("x")

        plt.show() 

    def get_norm(self):
        sum = 0
        for p in self.state['Psi']:
            sum += p*(self.state['x'][1]-self.state['x'][0])
        self.norm = sum.real

## read data
# potential
f = open(os.getcwd()+"/potential/potential.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n')[:-1]]
V = [float(i.split(',')[1]) for i in data.split('\n')[:-1]]
potential = {'x': x, 'V': V}


# state
f = open(os.getcwd()+"/input_state/input_state.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n')[:-1]]
Psi = [complex(i.split(',')[1]) for i in data.split('\n')[:-1]]
state = {'x' : x, 'Psi': Psi}

s = system(state, potential)

