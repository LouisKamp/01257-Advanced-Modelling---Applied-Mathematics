
#%%
import numpy as np
from pendulum_mag_sim import Magnet, Pendulum
import matplotlib.pyplot as plt

def plot_bifurcation(magnet_position=[], plot_title="bifrucation",magnetic_moment=[-1,1]):
    moment = np.linspace(magnetic_moment[0],magnetic_moment[1],600)
    for m_1 in moment:
        sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=0.1)
        if magnet_position:
            sim.add_magnet(Magnet(np.array(magnet_position), np.array([m_1,0])))
        roots = sim.roots()
        eq_points = roots[(roots[:, 0] < np.pi / 2) & (-np.pi / 2 < roots[:, 0])][:, 0]
        for eq_point in eq_points:
            eigenvalues = sim.eigenvalues([eq_point])
            is_stable = np.all(np.real(eigenvalues) < 0)
            color = 'blue' if is_stable else 'red'
            plt.scatter([m_1], eq_point, color=color)

    plt.xlabel("Magnetic moment")
    plt.ylabel("Phi")
    plt.savefig(f"{plot_title}.pdf", format="pdf")

# %%
import numpy as np
from pendulum_mag_sim import Magnet, Pendulum
import matplotlib.pyplot as plt

sideshift = np.array([1])

i = 0
for m_1y in sideshift:
    moment = np.linspace(-4,4,800)
    i += 1
    print(f'Sim {i} of {len(sideshift)}')
    for m_1 in moment:
        sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=0.1)
        sim.add_magnet(Magnet(np.array([1.5,m_1y]), np.array([m_1,0])))
        roots = sim.roots()
        eq_points = roots[(roots[:, 0] < np.pi / 2) & (-np.pi / 2 < roots[:, 0])][:, 0]
        for eq_point in eq_points:
            eigenvalues = sim.eigenvalues([eq_point])
            is_stable = np.all(np.real(eigenvalues) < 0)
            color = 'blue' if is_stable else 'red'
            plt.scatter([m_1], eq_point, color=color)

    plt.xlabel("Magnetic moment")
    plt.ylabel(r"$\phi$")
    plt.ylim(-np.pi/2, np.pi/2)
    plt.grid()
    plt.savefig(f"plotsbifurcation/bifurcation1.pdf")
    plt.show

    
    
# %%

# %%
