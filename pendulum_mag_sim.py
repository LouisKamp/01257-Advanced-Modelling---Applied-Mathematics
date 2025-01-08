from dataclasses import dataclass
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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

    def simulate(self, time_interval: tuple[float,float], initial_condition: tuple[float, float], t_num_eval=None):
        if t_num_eval == None:
            t_num_eval = 10*(time_interval[1] - time_interval[0])
        
        # setup
        f = self.create_f()
        ts = np.linspace(time_interval[0],time_interval[1],t_num_eval)

        # solve
        sol = solve_ivp(f, (time_interval[0],time_interval[1]), np.array([initial_condition[0],initial_condition[1]]), t_eval=ts)
        phis  = sol.y[0]
        d_phis = sol.y[1]

        xs = np.cos(phis)
        ys = np.sin(phis)

        # plot
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'o-', lw=2)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.invert_yaxis()

        for magnet in self.magnets:
            ax.scatter([magnet.position[1]], [magnet.position[0]])
            ax.quiver(magnet.position[1], magnet.position[0], magnet.moment[1], magnet.moment[0], angles='xy', scale_units='xy', scale=1)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data([0, ys[frame]], [0, xs[frame]])
            return line,
            
        ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
        plt.grid()
        plt.xlabel("y")
        plt.ylabel("x")
        ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True, interval=10)
        ax.set_aspect('equal')
        plt.show()

    def create_f(self):
        def f(t:float,y:np.ndarray) -> np.ndarray:
            sol_g = -self.g*np.sin(y[0]) / self.l
            sol_m = np.sum([
                0.0694200459087245*self.m_p*self.mu_0*(-2.0*self.l**4*magnet.moment[0]*np.sin(y[0]) + 2.0*self.l**4*magnet.moment[1]*np.cos(y[0]) - 2.0*self.l**3*magnet.moment[0]*magnet.position[0]*np.sin(2.0*y[0]) + 2.0*self.l**3*magnet.moment[0]*magnet.position[1]*np.cos(2.0*y[0]) + 7.0*self.l**3*magnet.moment[0]*magnet.position[1] + 2.0*self.l**3*magnet.moment[1]*magnet.position[0]*np.cos(2.0*y[0]) - 7.0*self.l**3*magnet.moment[1]*magnet.position[0] + 2.0*self.l**3*magnet.moment[1]*magnet.position[1]*np.sin(2.0*y[0]) + 11.25*self.l**2*magnet.moment[0]*magnet.position[0]**2*np.sin(y[0]) + 0.25*self.l**2*magnet.moment[0]*magnet.position[0]**2*np.sin(3.0*y[0]) - 14.5*self.l**2*magnet.moment[0]*magnet.position[0]*magnet.position[1]*np.cos(y[0]) - 0.5*self.l**2*magnet.moment[0]*magnet.position[0]*magnet.position[1]*np.cos(3.0*y[0]) - 3.25*self.l**2*magnet.moment[0]*magnet.position[1]**2*np.sin(y[0]) - 0.25*self.l**2*magnet.moment[0]*magnet.position[1]**2*np.sin(3.0*y[0]) + 3.25*self.l**2*magnet.moment[1]*magnet.position[0]**2*np.cos(y[0]) - 0.25*self.l**2*magnet.moment[1]*magnet.position[0]**2*np.cos(3.0*y[0]) + 14.5*self.l**2*magnet.moment[1]*magnet.position[0]*magnet.position[1]*np.sin(y[0]) - 0.5*self.l**2*magnet.moment[1]*magnet.position[0]*magnet.position[1]*np.sin(3.0*y[0]) - 11.25*self.l**2*magnet.moment[1]*magnet.position[1]**2*np.cos(y[0]) + 0.25*self.l**2*magnet.moment[1]*magnet.position[1]**2*np.cos(3.0*y[0]) - 2.0*self.l*magnet.moment[0]*magnet.position[0]**3*np.sin(2.0*y[0]) + 6.5*self.l*magnet.moment[0]*magnet.position[0]**2*magnet.position[1]*np.cos(2.0*y[0]) - 3.5*self.l*magnet.moment[0]*magnet.position[0]**2*magnet.position[1] + 7.0*self.l*magnet.moment[0]*magnet.position[0]*magnet.position[1]**2*np.sin(2.0*y[0]) - 2.5*self.l*magnet.moment[0]*magnet.position[1]**3*np.cos(2.0*y[0]) - 3.5*self.l*magnet.moment[0]*magnet.position[1]**3 - 2.5*self.l*magnet.moment[1]*magnet.position[0]**3*np.cos(2.0*y[0]) + 3.5*self.l*magnet.moment[1]*magnet.position[0]**3 - 7.0*self.l*magnet.moment[1]*magnet.position[0]**2*magnet.position[1]*np.sin(2.0*y[0]) + 6.5*self.l*magnet.moment[1]*magnet.position[0]*magnet.position[1]**2*np.cos(2.0*y[0]) + 3.5*self.l*magnet.moment[1]*magnet.position[0]*magnet.position[1]**2 + 2.0*self.l*magnet.moment[1]*magnet.position[1]**3*np.sin(2.0*y[0]) - 2.0*magnet.moment[0]*magnet.position[0]**4*np.sin(y[0]) + 3.0*magnet.moment[0]*magnet.position[0]**3*magnet.position[1]*np.cos(y[0]) - magnet.moment[0]*magnet.position[0]**2*magnet.position[1]**2*np.sin(y[0]) + 3.0*magnet.moment[0]*magnet.position[0]*magnet.position[1]**3*np.cos(y[0]) + magnet.moment[0]*magnet.position[1]**4*np.sin(y[0]) - magnet.moment[1]*magnet.position[0]**4*np.cos(y[0]) - 3.0*magnet.moment[1]*magnet.position[0]**3*magnet.position[1]*np.sin(y[0]) + magnet.moment[1]*magnet.position[0]**2*magnet.position[1]**2*np.cos(y[0]) - 3.0*magnet.moment[1]*magnet.position[0]*magnet.position[1]**3*np.sin(y[0]) + 2.0*magnet.moment[1]*magnet.position[1]**4*np.cos(y[0]))/(self.M_p*self.l**2*(0.5*self.l**2 - self.l*magnet.position[0]*np.cos(y[0]) - self.l*magnet.position[1]*np.sin(y[0]) + 0.5*magnet.position[0]**2 + 0.5*magnet.position[1]**2)**(7/2))
                for magnet in self.magnets
            ])
            sol_alpha = -(self.alpha * y[1]) / (self.M_p * self.l**2)
            return np.array([y[1], sol_g + sol_alpha + sol_m])

        return f

sim = Pendulum(M_p=1,m_p=0.5,l=1,mu_0=1,g=1,alpha=0.1)
sim.add_magnet(Magnet(np.array([1.5,1]), np.array([0.1,0])))
sim.add_magnet(Magnet(np.array([1.5,-1]), np.array([-0.1,0])))
sim.simulate((0,50), (0,-1))
