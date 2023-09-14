import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode as c_ode

class System():
    def __init__(self, init_state, potential, boundary = "periodic"):
        self.state = init_state
        self.potential = potential
        if(boundary == "periodic"):
            self.boundary = boundary
        else:
            raise ValueError(f"Option '{boundary}' for boundary condition is not yet implemented")
        # check matching x axis
        self.check_x()
        self.get_norm()
        self.state['Psi'] = [i/self.norm for i in state['Psi']]

    def check_x(self):
        ## check whether potential and state x values are the same
        # user needs to supply a one to one correspondence 
            # between x of potential and state
            # future work: sort potential and state in ascending order of x and
            # do interpolation of potential according to x values of initial state
        if(len(self.potential['x']) != len(self.state['x'])):
            raise ValueError(f"Potential and state have different length ({len(self.potential['x'])} and {len(self.state['x'])})")
        check = False
        for x in state['x']:
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

    def plot_Psi(self, state):
        plt.plot(state['x'],[i.real/self.norm for i in state['Psi']])
        plt.ylabel("Prob(x)")
        plt.xlabel("x")

        plt.show() 

    def plot_x_prob(self, state):
        plt.plot(state['x'],[np.sqrt(i.real**2 + i.imag**2)/self.norm for i in state['Psi']])
        plt.ylabel("Prob(x)")
        plt.xlabel("x")

        plt.show() 
    
    def plot_state_Laplace(self, state):
        L = self.get_Laplace(state)
        plt.plot(L['x'], L['l'].real)
        plt.ylabel("L(x)")
        plt.xlabel("x")

        plt.show() 

    def plot_Hamiltonian(self, state):
        H = self.schroedinger_H(state)
        plt.plot(state['x'], H)
        plt.ylabel("H(x)")
        plt.xlabel("x")

        plt.show() 

    def plot_final_state(self, T, h, method='Visscher'):
        if(method=='Visscher'):
            state = self.evolve_system_Visscher(T, h)
        elif(method=='RK4'):
            state = self.evolve_system_RK4(T, h)
        else:
            raise ValueError(f"Option '{method}' is not yet implemented")
        plt.plot(state['x'],[np.sqrt(i.real**2 + i.imag**2) for i in state['Psi']])
        plt.ylabel("Psi(T)")
        plt.xlabel("x")

        plt.show() 

    def get_norm(self):
        sum = 0
        for p in self.state['Psi']:
            sum += (p.real**2+p.imag**2) *(self.state['x'][1]-self.state['x'][0])
        self.norm = sum.real

    def get_Laplace(self, state):
        laplace = {'x' : state['x'], 'l' : np.zeros(len(x),dtype = complex)}
        laplace['l'][1:-1] = [
            (state['Psi'][i+1]-2*state['Psi'][i]+state['Psi'][i-1])\
                        /(state['x'][i+1]-state['x'][i])**2 \
                            for i in range(1,len(state['x'])-1)
                        ]
        
        if(self.boundary == 'periodic'):
            laplace['l'][0] = (state['Psi'][1]-2.*state['Psi'][0]+state['Psi'][-1])\
                        /(state['x'][1]-state['x'][0])**2
            laplace['l'][-1] = (state['Psi'][0]-2.*state['Psi'][-1]+state['Psi'][-2])\
                        /(state['x'][-1]-state['x'][-2])**2
            
        else:
            raise ValueError(f"Option {self.boundary} for boundary condition is not yet implemented")
        
        return laplace

    def schroedinger_H(self, state):
        L = self.get_Laplace(state)
        #H = [-1j*(self.potential['V'][i]*state['Psi'][i]) for i in range(len(L['x']))]
        H = [-0.5*(L['l'][i]+self.potential['V'][i]*state['Psi'][i]) for i in range(len(L['x']))]
        return(H)

    def evolve_system_RK4(self, T, h):
        state = self.state
        for t in range(int(T/h)):
            print(str(100*t/(int(T/h)))+'%'+'\r',end='')
            state = self.rk4_step(state, h)
        return state

    def evolve_system_Visscher(self,T,h):
        state = self.state
        for t in range(int(T/h)):
            print(str(100*t/(int(T/h)))+'%'+'\r',end='')
            state = self.Visscher_step(state, h)
        return state

    def rk4_step(self, state, h):
        k1 = [-H*1j for H in self.schroedinger_H(state)]
        y1 = {'x': state['x'], 
              'Psi': [y + 0.5*h*k1[i] for i,y in enumerate(state['Psi'])]}
        k2 = [-H*1j for H in self.schroedinger_H(y1)]
        y2 = {'x': state['x'], 
              'Psi': [y + 0.5*h*k2[i] for i,y in enumerate(y1['Psi'])]}
        k3 = [-H*1j for H in self.schroedinger_H(y2)]
        y3 = {'x': state['x'], 
              'Psi': [y + h*k3[i] for i,y in enumerate(y2['Psi'])]}
        k4 = [-H*1j for H in self.schroedinger_H(y3)]
        state = {'x': state['x'], 
                 'Psi': [y + h*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i,y in enumerate(state['Psi'])]}
        return state
        
    def Visscher_step(self, state, h):
        R = {'x' : state['x'], 'Psi' : [i.real for i in state['Psi']]}
        I = {'x' : state['x'], 'Psi' : [i.imag for i in state['Psi']]}
        I['Psi'] = [I['Psi'][i] + 1j*h*s for i,s in enumerate(self.schroedinger_H(R))]
        R['Psi'] = [R['Psi'][i] - 1j*h*s for i,s in enumerate(self.schroedinger_H(I))]
        return ({'x' : state['x'], 'Psi' : [R['Psi'][i] + 1j*I['Psi'][i] for i in range(len(state['Psi']))]})

## read data
# potential
f = open(os.getcwd()+"/potential/potential.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n')]
V = [float(i.split(',')[1]) for i in data.split('\n')]
potential = {'x': x, 'V': V}

# state
f = open(os.getcwd()+"/input_state/input_state.dat", "r")
data = f.read()
x = [float(i.split(',')[0]) for i in data.split('\n')]
Psi = [complex(i.split(',')[1]) for i in data.split('\n')]

'''
# Test case

x = [float(i.split(',')[0]) for i in data.split('\n')]
Psi = [np.cos(2*np.pi*t/(len(x))) for t in range(len(x))]
'''

state = {'x' : x, 'Psi': Psi}
s = System(state, potential)

#s.plot_final_state(.1,0.00001, method='RK4')
s.plot_final_state(1,0.00005, method='Visscher')