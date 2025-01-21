#%%
import numpy as np
import matplotlib.pyplot as plt
from pendulum_mag_sim import Magnet, Pendulum

def plot_vector_field(magnet_position=[], plot_title="phase_portrait"):
    phis = np.linspace(-np.pi,np.pi,20)
    dphis = np.linspace(-2,2,20)

    Phis, dPhis = np.meshgrid(phis, dphis)

    sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=1)
    if magnet_position:
        sim.add_magnet(Magnet(np.array(magnet_position), np.array([-2,0])))

    y1,y2 = sim.f(0, np.array([Phis, dPhis]))
    steady_states = sim.roots()

    plt.quiver(Phis, dPhis, y1, y2)
    plt.scatter(steady_states[:,0], steady_states[:,1], color='black',label="Equilibrium points")
    plt.legend()
    plt.xlim(-np.pi, np.pi)
    plt.xlabel('Phi')
    plt.ylabel('dPhi')
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\phi '$")
    plt.title('Vector Field')
    plt.savefig("Plots/vector_field_below.pdf")
    plt.show()

    sim.f(0, np.array([-0.091, 0]))
# %%

plot_vector_field([1.5,0])



# %%
