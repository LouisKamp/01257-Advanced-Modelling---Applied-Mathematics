from dataclasses import dataclass
from itertools import product
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.optimize import root

@dataclass
class Magnet():
    position: np.ndarray    # [x,y]-position in modified coordinate system
    moment: np.ndarray      # magnetic moment [x,y]

class Pendulum():
    M_p: float      # Pendulum mass
    m_p: float      # Pendulum magnet moment
    l: float        # Pendulum length
    mu_0: float     # permeability of free space
    g: float        # gravitational acceleration
    alpha: float    # dampening constant

    magnets: list[Magnet]

    def __init__(self, M_p, m_p: float, l: float, mu_0: float, g: float, alpha: float):
        self.M_p = M_p
        self.m_p = m_p
        self.l = l
        self.mu_0 = mu_0
        self.g = g
        self.alpha = alpha
        self.magnets = []

    def add_magnet(self, magnet: Magnet):
        self.magnets.append(magnet)

    def simulate(self, time_interval: tuple[float,float], initial_condition: tuple[float, float, float, float], show_plots=True):
        ts = None
        if show_plots == True:
            t_num_eval = int(10*(time_interval[1] - time_interval[0]))
            ts = np.linspace(time_interval[0],time_interval[1],t_num_eval)

        # def event(t, y):
        #     if t > 0.5:
        #         return np.abs(y[1]) + np.abs(y[3]) - 0.5
        #     else:
        #         return 1

        # event.terminal = True  # Stop integration when event is triggered
        # event.direction = 0    # Trigger on either direction

        sol = solve_ivp(self.f, (time_interval[0], time_interval[1]), np.array([initial_condition[0], initial_condition[1], initial_condition[2], initial_condition[3]]), t_eval=ts)

        ts = sol.t
        phis  = sol.y[0]
        d_phis = sol.y[1]

        theta  = sol.y[2]
        d_theta = sol.y[3]

        if show_plots:
            xs = self.l*np.sin(theta)*np.cos(phis)
            ys = self.l*np.sin(theta)*np.sin(phis)
            zs = self.l*np.cos(theta)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            line, = ax.plot([], [], [], 'o-', lw=2, alpha=0.5)
            trace, = ax.plot([], [], [], '-', lw=2, alpha=1)

            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            
            def init():
                line.set_data([], [])
                line.set_3d_properties([])
                trace.set_data([], [])
                trace.set_3d_properties([])
                return line, trace

            trace_len = 10
            def update(frame):
                line.set_data([0, ys[frame]], [0, zs[frame]])
                line.set_3d_properties([0, xs[frame]])
                trace.set_data(ys[max(0, frame-trace_len):frame], zs[max(0, frame-trace_len):frame])
                trace.set_3d_properties([2]*len(xs[max(0, frame-trace_len):frame]))
                return line, trace
            
            for i, magnet in enumerate(self.magnets):
                ax.scatter([magnet.position[1]], [magnet.position[2]], [magnet.position[0]], label=f"Magnet {i}")
                ax.quiver(magnet.position[1], magnet.position[2], magnet.position[0], 
                          magnet.moment[1], magnet.moment[2], magnet.moment[0], normalize=True)
                
            roots = self.roots()
            roots_phi = roots[:,0]
            roots_theta = roots[:,2]
            
            roots_x = self.l * np.sin(roots_theta) * np.cos(roots_phi)
            roots_y = self.l * np.sin(roots_theta) * np.sin(roots_phi)
            roots_z = self.l * np.cos(roots_theta)

            ax.scatter(roots_y, roots_z,roots_x, color='r', label='Roots')

            ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
            ax.set_aspect('equal')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_zlabel('X')
            ax.invert_zaxis()
            plt.show()

        return sol.t, sol.y

    def f(self, t:float,y:np.ndarray) -> np.ndarray:
        sol_g_phi = -(2.0*y[1]*y[3]*self.l*np.cos(y[2]) + self.g*np.sin(y[0]))/(self.l*np.sin(y[2]))

        sol_m_phi = np.sum(np.array([
            (-self.M_p*y[1]*y[3]*self.l**2*np.sin(2.0*y[2]) - 1/4*np.pi*self.m_p*self.mu_0*(-5*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[2])*np.cos(y[0]) + 5*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.sin(y[0])*np.sin(y[2]))*((magnet.moment[0]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.cos(y[0]) + (magnet.moment[1]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.sin(y[0]) + (magnet.moment[2]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.cos(y[2]) - magnet.position[2])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.cos(y[2]))/((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2)**(7/2) - 1/4*np.pi*self.m_p*self.mu_0*(-(magnet.moment[0]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.sin(y[0]) + (magnet.moment[1]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.cos(y[0]) + (magnet.moment[2]*(2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[2])*np.cos(y[0]) - 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.sin(y[0])*np.sin(y[2])) - 3*(self.l*np.cos(y[2]) - magnet.position[2])*(-self.l*magnet.moment[0]*np.sin(y[0])*np.sin(y[2]) + self.l*magnet.moment[1]*np.sin(y[2])*np.cos(y[0])))*np.cos(y[2]) + (3*self.l*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2]))*np.sin(y[0])*np.sin(y[2]) + magnet.moment[0]*(2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[2])*np.cos(y[0]) - 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.sin(y[0])*np.sin(y[2])) - 3*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*(-self.l*magnet.moment[0]*np.sin(y[0])*np.sin(y[2]) + self.l*magnet.moment[1]*np.sin(y[2])*np.cos(y[0])))*np.cos(y[0]) + (-3*self.l*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2]))*np.sin(y[2])*np.cos(y[0]) + magnet.moment[1]*(2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[2])*np.cos(y[0]) - 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.sin(y[0])*np.sin(y[2])) - 3*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*(-self.l*magnet.moment[0]*np.sin(y[0])*np.sin(y[2]) + self.l*magnet.moment[1]*np.sin(y[2])*np.cos(y[0])))*np.sin(y[0]))/((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2)**(5/2))/(self.M_p*self.l**2*np.sin(y[2])**2)
            for magnet in self.magnets
        ]), axis=0)

        sol_m_theta = np.sum(np.array([
            (1.0*self.M_p*y[1]**2*self.l**2*np.sin(y[2])*np.cos(y[2]) - 1/4*np.pi*self.m_p*self.mu_0*((magnet.moment[0]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.cos(y[0]) + (magnet.moment[1]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.sin(y[0]) + (magnet.moment[2]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.cos(y[2]) - magnet.position[2])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.cos(y[2]))*(5*self.l*(self.l*np.cos(y[2]) - magnet.position[2])*np.sin(y[2]) - 5*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[0])*np.cos(y[2]) - 5*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.cos(y[0])*np.cos(y[2]))/((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2)**(7/2) - 1/4*np.pi*self.m_p*self.mu_0*(-(magnet.moment[2]*((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2) - 3*(self.l*np.cos(y[2]) - magnet.position[2])*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2])))*np.sin(y[2]) + (3*self.l*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2]))*np.sin(y[2]) + magnet.moment[2]*(-2*self.l*(self.l*np.cos(y[2]) - magnet.position[2])*np.sin(y[2]) + 2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[0])*np.cos(y[2]) + 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.cos(y[0])*np.cos(y[2])) - 3*(self.l*np.cos(y[2]) - magnet.position[2])*(self.l*magnet.moment[0]*np.cos(y[0])*np.cos(y[2]) + self.l*magnet.moment[1]*np.sin(y[0])*np.cos(y[2]) - self.l*magnet.moment[2]*np.sin(y[2])))*np.cos(y[2]) + (-3*self.l*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2]))*np.sin(y[0])*np.cos(y[2]) + magnet.moment[1]*(-2*self.l*(self.l*np.cos(y[2]) - magnet.position[2])*np.sin(y[2]) + 2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[0])*np.cos(y[2]) + 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.cos(y[0])*np.cos(y[2])) - 3*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*(self.l*magnet.moment[0]*np.cos(y[0])*np.cos(y[2]) + self.l*magnet.moment[1]*np.sin(y[0])*np.cos(y[2]) - self.l*magnet.moment[2]*np.sin(y[2])))*np.sin(y[0]) + (-3*self.l*(magnet.moment[0]*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0]) + magnet.moment[1]*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1]) + magnet.moment[2]*(self.l*np.cos(y[2]) - magnet.position[2]))*np.cos(y[0])*np.cos(y[2]) + magnet.moment[0]*(-2*self.l*(self.l*np.cos(y[2]) - magnet.position[2])*np.sin(y[2]) + 2*self.l*(self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])*np.sin(y[0])*np.cos(y[2]) + 2*self.l*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*np.cos(y[0])*np.cos(y[2])) - 3*(self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])*(self.l*magnet.moment[0]*np.cos(y[0])*np.cos(y[2]) + self.l*magnet.moment[1]*np.sin(y[0])*np.cos(y[2]) - self.l*magnet.moment[2]*np.sin(y[2])))*np.cos(y[0]))/((self.l*np.cos(y[2]) - magnet.position[2])**2 + (self.l*np.sin(y[0])*np.sin(y[2]) - magnet.position[1])**2 + (self.l*np.sin(y[2])*np.cos(y[0]) - magnet.position[0])**2)**(5/2))/(self.M_p*self.l**2)
            for magnet in self.magnets
        ]), axis=0)

        sol_g_theta = (self.M_p*self.g*self.l*np.cos(y[0])*np.cos(y[2]) + 0.5*self.M_p*(2*y[3]**2*self.l**2*np.sin(y[2])*np.cos(y[2]) + (-y[1]*self.l*np.sin(y[0])*np.sin(y[2]) + y[3]*self.l*np.cos(y[0])*np.cos(y[2]))*(-2*y[1]*self.l*np.sin(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[2])*np.cos(y[0])) + (y[1]*self.l*np.sin(y[2])*np.cos(y[0]) + y[3]*self.l*np.sin(y[0])*np.cos(y[2]))*(2*y[1]*self.l*np.cos(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[0])*np.sin(y[2]))))/(self.M_p*self.l**2)

        sol_alpha_phi = (-self.M_p*y[1]*y[3]*self.l**2*np.sin(2.0*y[2]) + 0.5*self.M_p*((-2*y[1]*self.l*np.sin(y[0])*np.sin(y[2]) + 2*y[3]*self.l*np.cos(y[0])*np.cos(y[2]))*(y[1]*self.l*np.sin(y[2])*np.cos(y[0]) + y[3]*self.l*np.sin(y[0])*np.cos(y[2])) + (-y[1]*self.l*np.sin(y[0])*np.sin(y[2]) + y[3]*self.l*np.cos(y[0])*np.cos(y[2]))*(-2*y[1]*self.l*np.sin(y[2])*np.cos(y[0]) - 2*y[3]*self.l*np.sin(y[0])*np.cos(y[2]))) - self.alpha*y[1])/(self.M_p*self.l**2*np.sin(y[2])**2)

        sol_alpha_theta = (0.5*self.M_p*(2*y[3]**2*self.l**2*np.sin(y[2])*np.cos(y[2]) + (-y[1]*self.l*np.sin(y[0])*np.sin(y[2]) + y[3]*self.l*np.cos(y[0])*np.cos(y[2]))*(-2*y[1]*self.l*np.sin(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[2])*np.cos(y[0])) + (y[1]*self.l*np.sin(y[2])*np.cos(y[0]) + y[3]*self.l*np.sin(y[0])*np.cos(y[2]))*(2*y[1]*self.l*np.cos(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[0])*np.sin(y[2]))) - self.alpha*y[3])/(self.M_p*self.l**2)

        return np.array([y[1], sol_g_phi + sol_m_phi + sol_alpha_phi, y[3], sol_g_theta + sol_m_theta + sol_alpha_theta])
    
    def roots(self):
        phis = np.linspace(-2*np.pi, 2*np.pi, 20)
        d_phis = [0]
        thetas = np.linspace(0.1, np.pi, 20)
        d_thetas = [0]

        guesses = list(product(phis, d_phis, thetas, d_thetas))

        roots = []
        F = lambda y: self.f(0, y)

        for guess in guesses:
            sol = root(F, guess)
            if sol.success:
                if not any(np.allclose(sol.x, r) for r in roots):
                    roots.append(sol.x)
        return np.array(roots)

if __name__ == "__main__0":
    # Sim 1: Pendulum should end in the left hand side of the plot
    sim = Pendulum(M_p=0.1,m_p=1,l=1,mu_0=1,g=50,alpha=0.2)
    sim.add_magnet(Magnet(np.array([1.5,1,1]), np.array([1,0,0])))
    sim.add_magnet(Magnet(np.array([1.5,-1,1]), np.array([1,0,0])))
    sim.add_magnet(Magnet(np.array([1.5,1,-1]), np.array([1,0,0])))
    sim.add_magnet(Magnet(np.array([1.5,-1,-1]), np.array([1,0,0])))

    n = 100
    phis = np.linspace(-1.9,1.9,n)
    d_phis = [0]
    thetas = np.pi/2  + np.linspace(-1.5,1.5,n)
    d_thetas = [0]

    initial_conditions = np.array(list(product(phis, d_phis, thetas, d_thetas)))

    roots = sim.roots()

    closest_roots = np.zeros(len(initial_conditions))
    for i, initial_condition in enumerate(initial_conditions):
        print(i,initial_condition)
        ts, ys = sim.simulate((0,3), initial_condition, show_plots=False)

        first = ys[:,0]
        last = ys[:,-1]

        distances = np.linalg.norm(roots - last, axis=1)
        closest_roots[i] = np.argmin(distances)

    phis_plot = initial_conditions[:,0]
    thetas_plot = initial_conditions[:,2]

    # plt.scatter(phis_plot, thetas_plot, c=closest_roots)
    # plt.show()

    plt.matshow(closest_roots.reshape((n,n)).T)
    plt.gca().invert_yaxis()
    plt.xlabel("Phi")
    plt.ylabel("Theta")
    plt.xticks(ticks=np.arange(0, n, n//10), labels=np.round(phis[::n//10], 2))
    plt.yticks(ticks=np.arange(0, n, n//10), labels=np.round(thetas[::n//10], 2))
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig("./figures/chaos_plot.pdf")
