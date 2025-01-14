#%%
import numpy as np
import matplotlib.pyplot as plt
from pendulum_mag_sim import Magnet, Pendulum

def plot_phase_portrait(magnet_position=[], plot_title="phase_portrait"):
    phis = np.linspace(-1,1,10)
    # dphis = np.linspace(-1,0.1,10)

    for phi in phis:
        sim = Pendulum(M_p=1,m_p=0.1,l=1,mu_0=1,g=1,alpha=0.1)
        if magnet_position:
            sim.add_magnet(Magnet(np.array(magnet_position), np.array([1,1])))
        phi_res, dphi_res = sim.simulate((0,50), (phi, 0), show_plots=False)
        plt.scatter(phi_res[0], dphi_res[0], color='red', s=10)
        plt.scatter(phi_res[-1], dphi_res[-1], color='blue', s=10)
        plt.plot(phi_res, dphi_res, linewidth=0.5)

    plt.xlabel("Phi")
    plt.ylabel("d_Phi")
    plt.savefig(f"{plot_title}.pdf", format="pdf")
    plt.show()

# %%
