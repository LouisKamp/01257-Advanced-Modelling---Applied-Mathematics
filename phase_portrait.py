import numpy as np
import matplotlib.pyplot as plt
from pendulum_mag_sim import Pendulum

phis = np.linspace(-np.pi,np.pi,20)
dphis = np.linspace(-1,1,10)

for phi, dphi in zip(phis, dphis):
    sim = Pendulum(M_p=1,m_p=1,l=1,mu_0=1,g=1,alpha=0.1)
    phi_res, dphi_res = sim.simulate((0,50), (phi, dphi), show_plots=False)
    plt.scatter(phi_res[0], dphi_res[0], color='red', s=10)
    plt.scatter(phi_res[-1], dphi_res[-1], color='blue', s=10)
    plt.plot(phi_res, dphi_res, linewidth=0.5)

plt.xlabel("Phi")
plt.ylabel("d_Phi")
plt.show()



