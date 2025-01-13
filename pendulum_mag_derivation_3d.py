# %%
from sympy import *
from sympy.printing.pycode import pycode

#%%
t= symbols("t")
phi = Function('phi')(t)
phi_dot = diff(phi,t)

theta = Function('theta')(t)
theta_dot = diff(theta, t)

# %%

l, mu_0, M_p, g, alpha, m_p_abs = symbols("l, mu_0, M_p, g, alpha, m_p_abs", real=True)

x_p = l * sin(theta) * cos(phi)
y_p = l * sin(theta) * sin(phi)
z_p = l * cos(theta)

U_g =  -M_p * g * x_p
# %%

x_1, y_1, z_1 = symbols("x_1, y_1, z_1")
m_1_x, m_1_y, m_1_z = symbols("m_1_x, m_1_y, m_1_z")

r = Matrix([x_p - x_1, y_p - y_1, z_p - z_1])
r_norm = sqrt(r[0]**2 + r[1]**2 + r[2]**2)
r_hat = r / r_norm

m_1 = Matrix([m_1_x, m_1_y, m_1_z])
m_p = m_p_abs*Matrix([cos(phi), sin(phi), cos(theta)])

B_1 = (mu_0 / 4*pi)*( 3*r_hat * r_hat.dot(m_1) - m_1 ) / (r_norm**3)
U_1 = -m_p.dot(B_1)

#%%
V = U_g + 0*U_1
T = 0.5 * M_p * ( diff(x_p,t)**2 + diff(y_p,t)**2 + diff(z_p,t)**2 )
L = T - V

# %%
P_phi =  diff(L, phi)
P_phi_sym = symbols("P_{phi}")

sol_phi = Eq(P_phi_sym - diff(diff(L, phi_dot),t), 0)
phi_t_t = solve(sol_phi, diff(phi,t,t))[0].subs({P_phi_sym: P_phi})

#%%

sol_phi = Eq(diff(L, phi) - diff(diff(L, phi_dot),t), 0)
phi_t_t = solve(sol_phi, diff(phi,t,t))[0]

#%%
P_theta =  diff(L, theta)
P_theta_sym = symbols("P_{theta}")
sol_theta = Eq(P_theta_sym - diff(diff(L, theta_dot),t), 0*alpha*theta_dot)

theta_t_t = solve(sol_theta, diff(theta,t,t))[0].subs({P_theta_sym: P_theta})

# %%
phi_1, d_phi = symbols("phi d_{phi}")
theta_1, d_theta = symbols("theta d_{theta}")

# %%
pycode(phi_t_t.subs({phi: phi_1, phi_dot: d_phi, theta: theta_1, theta_dot: d_theta}))

# %%
pycode(theta_t_t.subs({phi: phi_1, phi_dot: d_phi, theta: theta_1, theta_dot: d_theta}))
# %%
