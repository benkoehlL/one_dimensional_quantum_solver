import numpy as np
import matplotlib.pyplot as plt


class System():
    """
       This class creates an object consisting of a spatially-discretised 
       initial state and time-independent potential. It allows to propagate
       this system through time by solving the time-dependent Schrödinger 
       equation and to plot and export the resulting final state.
    """
    
    def __init__(self, init_state, potential, boundary = "periodic"):
        """Initialisation of the class

        Args:
            init_state (dict): a dictionary with keys 'x' and 'Psi' for 
                               storing arrays of the wave function Psi's 
                               complex values at different positions x  
            potential (dict): a dictionary with keys 'x' and 'V' for 
                               storing arrays of the potential V's values
                               at different positions x
            boundary (str, optional): the boundary conditions to use for solving
                                      the equation of motion. 
                                      Defaults to "periodic".
                                      options are ["periodic"]

        Raises:
            ValueError: raises value error when giving unknown boundary conditions
        """

        self.state = init_state
        self.potential = potential
        if(boundary == "periodic"):
            self.boundary = boundary
        else:
            raise ValueError(f"Option '{boundary}' for boundary condition is not yet implemented")
        # check matching x axis
        self.check_x()
        # normalise the state
        self.norm = self.get_norm(self.state)
        self.state['Psi'] = [i/self.norm for i in self.state['Psi']]

    def get_Laplace(self, state):
        """
           Returns the Laplacian of a given quantum state

        Args:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x

        Returns:
            dict: a dictionary with an array of values and arguments
                  for the keys 'l' and 'x', respectively 
        """
        laplace = {'x' : state['x'], 'l' : np.zeros(len(state['x']),dtype = complex)}
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
        """
            Returns the Schrödinger Hamiltonian for a given quantum state 
            and the system's potential

        Args:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x

        Returns:
            H (array):  an array for the Hamiltonian corresponding to the system's
                        state
        """
        L = self.get_Laplace(state)
        H = [-0.5*L['l'][i]+self.potential['V'][i]*state['Psi'][i] for i in range(len(L['x']))]
        return H

    def evolve_system_RK4(self, T, h):
        """
            Evolves the system using a fourth order Runge-Kutta integration algorithm
            by time T in steps of h.

        Args:
            T (float): total time the system is evolving
            h (float): stepsize for the propagation 

        Returns:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
        """

        state = self.state
        for t in range(int(T/h)):
            print(" Progress: " + str("{:.2f}".format(100*t/(int(T/h))))+'%'+'\r',end='')
            state = self.rk4_step(state, h)
        return state

    def evolve_system_Visscher(self, T, h):
        """
            Evolves the system using a integration algorithm described in

            P.B. Visscher: "A fast explicit algorithm for the time dependent 
            Schrödinger equation", Computers in  Physics 5 (6), 596-598 (1991)

            by time T in steps of h.

        Args:
            T (float): total time the system is evolving
            h (float): stepsize for the propagation 

        Returns:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
        """

        state = self.state
        for t in range(int(T/h)):
            print(" Progress: " + str("{:.2f}".format(100*t/(int(T/h))))+'%'+'\r',end='')
            state = self.Visscher_step(state, h)
        return state

    def rk4_step(self, state, h):
        """
           A single step of size h using a fourth-order Runge-Kutta method

        Args:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
                            before the time step
            h (float):  size of the integration step

        Returns:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
                            after the time step
        """
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
        """
           A single step of size h using the Visscher method

        Args:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
                            before the time step
            h (float): _description_

        Returns:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
                            after the time step
        """
        R = {'x' : state['x'], 'Psi' : [i.real for i in state['Psi']]}
        I = {'x' : state['x'], 'Psi' : [i.imag for i in state['Psi']]}
        I['Psi'] = [I['Psi'][i] - h*s for i,s in enumerate(self.schroedinger_H(R))]
        R['Psi'] = [R['Psi'][i] + h*s for i,s in enumerate(self.schroedinger_H(I))]
        return ({'x' : state['x'], 'Psi' : [R['Psi'][i] + 1j*I['Psi'][i] for i in range(len(state['Psi']))]})
    
    def get_final_state(self, T, h, method="Visscher"):
        """
            A function to propagate the system by a total time T 
            in steps of h.

        Args:
            T (float): total time the system is evolving
            h (float): stepsize for the propagation 
            method (str, optional): integration method to use.
                                    Defaults to "Visscher".
                                    options are ['Visscher','RK4']

        Raises:
            ValueError: raises an value error when the integration method
                        is not known

        Returns:
            state (dict):   a dictionary with keys 'x' and 'Psi' for 
                            storing arrays of the wave function Psi's 
                            complex values at different positions x
                            after evolving a total time of T
        """
        if(method == 'Visscher'):
            state = self.evolve_system_Visscher(T, h)
        elif(method == 'RK4'):
            state = self.evolve_system_RK4(T, h)
        else:
            raise ValueError(f"Option '{method}' is not yet implemented")
        return state

    def export_final_state(self, T, h, file_name, method="Visscher"):
        """
            Exports a file into a given path with the state after
            evolving for a total time of T with a stepsize of h
            using a specific integration method of the time-dependent
            Schrödinger equation       

        Args:
            T (float): total time the system is evolving
            h (float): stepsize for the propagation 
            file_name (str): path of the file the final state
                             is stored in
            method (str, optional): integration method to use.
                                    Defaults to "Visscher".
                                    options are ['Visscher','RK4']
        """
        state = self.get_final_state(T, h, method)
        # normalise the final state (necessary due to numerical errors)
        #norm = self.get_norm(state)
        #state['Psi'] = [i/norm for i in state['Psi']]

        # open the file in the write mode
        f = open(file_name, 'w')
        for i in range(len(state['x'])):
            f.write(f"{state['x'][i]},{state['Psi'][i]}\n")

    def check_x(self):
        """ 
            check whether potential and state x values are the same
            user needs to supply a one to one correspondence 
            between x of potential and state
            
        Raises:
            ValueError: raises value error when state and potential have unequal length
            ValueError: raises value error when discretisised positions of state and potential
                        do not match
        """
        ## future work: sort potential and state in ascending order of x and
        #  do interpolation of potential according to x values of initial state
        if(len(self.potential['x']) != len(self.state['x'])):
            raise ValueError(f"Potential and state have different length ({len(self.potential['x'])} and {len(self.state['x'])})")
        check = False
        for x in self.state['x']:
            if(x not in self.potential['x']):
                check = True
                print(x)
                break
        if(check):
            raise ValueError("Potential and state have non-matching x range")
        
    def get_norm(self, state):
        """
            Gets the norm of the initial state and saves it in a class variable    
        """
        sum = 0
        for p in state['Psi']:
            sum += np.sqrt((p.real**2+p.imag**2)) *(state['x'][1]-state['x'][0])
        return sum

    def plot_potential(self):
        """
            Plots the (time-independent) potential of the system
        """
        plt.plot(self.potential['x'],self.potential['V'])
        plt.ylabel("V(x)")
        plt.xlabel("x")
        plt.show() 

    def plot_x_prob(self, state, label=None):
        """
           Plots the probability density of the system 
           (to show the plot, you need to use plt.show())
        """
        plt.plot(state['x'],[np.sqrt(i.real**2 + i.imag**2) for i in state['Psi']], 
                 label=label)
        plt.ylabel("Prob(x)")
        plt.xlabel("x")

    def plot_final_state(self, T, h, method='Visscher'):
        """
            Plots the final state after it is evolving
            for a time T in steps of h.

        Args:
            T (float): total time the system is evolving
            h (float): stepsize for the propagation 
            method (str, optional): integration method to be used.
                                    Defaults to 'Visscher'.
                                    options are ['Visscher,"RK4]
        """
        state = self.get_final_state(T,h,method)
        plt.plot(state['x'],[np.sqrt(i.real**2 + i.imag**2) for i in state['Psi']])
        plt.ylabel("Psi(T)")
        plt.xlabel("x")
        plt.show() 