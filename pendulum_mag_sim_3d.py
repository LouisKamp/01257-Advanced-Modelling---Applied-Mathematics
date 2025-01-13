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

    def simulate(self, time_interval: tuple[float,float], initial_condition: tuple[float, float], t_num_eval=None, show_plots=True):
        if t_num_eval == None:
            t_num_eval = 10*(time_interval[1] - time_interval[0])
        
        # setup
        ts = np.linspace(time_interval[0],time_interval[1],t_num_eval)

        # solve
        sol = solve_ivp(self.f, (time_interval[0],time_interval[1]), np.array([initial_condition[0],initial_condition[1],initial_condition[2],initial_condition[3]]), t_eval=ts)
        phis  = sol.y[0]
        d_phis = sol.y[1]

        theta  = sol.y[2]
        d_theta = sol.y[3]

        xs = self.l*np.sin(theta)*np.cos(phis)
        ys = self.l*np.sin(theta)*np.sin(phis)
        zs = self.l*np.cos(theta)

        if show_plots:
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

            def update(frame):
                line.set_data([0, xs[frame]], [0, ys[frame]])
                line.set_3d_properties([0, zs[frame]])
                trace.set_data(xs[:frame], ys[:frame])
                trace.set_3d_properties([2]*len(zs[:frame]))
                return line, trace
                
            ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.invert_zaxis()
            plt.show()

        return phis, d_phis, theta, d_theta

    def f(self, t:float,y:np.ndarray) -> np.ndarray:
        sol_g_phi = -(2.0*y[1]*y[3]*self.l*np.cos(y[2]) + self.g*np.sin(y[0]))/(self.l*np.sin(y[2]))

        sol_g_theta = (self.M_p*self.g*self.l*np.cos(y[0])*np.cos(y[2]) + 0.5*self.M_p*(2*y[3]**2*self.l**2*np.sin(y[2])*np.cos(y[2]) + (-y[1]*self.l*np.sin(y[0])*np.sin(y[2]) + y[3]*self.l*np.cos(y[0])*np.cos(y[2]))*(-2*y[1]*self.l*np.sin(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[2])*np.cos(y[0])) + (y[1]*self.l*np.sin(y[2])*np.cos(y[0]) + y[3]*self.l*np.sin(y[0])*np.cos(y[2]))*(2*y[1]*self.l*np.cos(y[0])*np.cos(y[2]) - 2*y[3]*self.l*np.sin(y[0])*np.sin(y[2]))))/(self.M_p*self.l**2)

        return np.array([y[1], sol_g_phi, y[3], sol_g_theta])
    
if __name__ == "__main__":
    # Sim 1: Pendulum should end in the left hand side of the plot
    sim = Pendulum(M_p=1,m_p=1,l=1,mu_0=1,g=1,alpha=0.)
    # sim.add_magnet(Magnet(np.array([1,0,0]), np.array([0.0,0,0])))

    sim.simulate((0,100), (1,0,np.pi/2,0))
