# %%
from sympy import *
t= symbols("t")
phi = Function('phi')(t)
phi_dot = diff(phi,t)

# %%
M_p = 1
g = 0*2.7
l = 1
mu_0 = 1
m_p = Matrix([sin(phi), cos(phi)])
m_1 = 10*Matrix([-1,-1])

# pendulum placement
x_p = l*sin(phi)
y_p = -l*cos(phi)

# magnetic 1 placement
x_1 = -1
y_1 = -1


r_1 = Matrix([x_p - x_1, y_p - y_1])
r_1_norm = sqrt( r_1[0]**2 + r_1[1]**2 )
r_1_hat = r_1 / r_1_norm

# %%
B_1 = (mu_0 / (4 * pi)) * (3*r_1_hat * ( r_1_hat.T @ m_1 ) - m_1) / (r_1_norm**3)
U_1 = simplify(-m_p.T @ B_1)[0]

# %%
T = simplify(0.5*M_p*( diff(x_p, t)**2 + diff(y_p,t)**2))

# %%
V = M_p*g*y_p + U_1

# %%
L = T - V

# %%
sol = Eq(diff(L, phi) - diff(diff(L, phi_dot),t),0)

# %%
phi_t_t = solve(sol, diff(phi,t,t))[0]

# %%

phi_t_t_func = lambdify(phi, phi_t_t, modules="numpy")

