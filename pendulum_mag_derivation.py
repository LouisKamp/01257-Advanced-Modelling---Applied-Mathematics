# %%
from sympy import *
t= symbols("t")
phi = Function('phi')(t)
phi_dot = diff(phi,t)

# %%
l = 1
mu_0 = 1
M_p = 1
g = 1
alpha = 0.1

x_p = l * cos(phi)
y_p = l * sin(phi)

x_1 = 1
y_1 = 1

m_p = Matrix([cos(phi), sin(phi)])
m_1 = 1*Matrix([1,1])

r = Matrix([ x_p - x_1, y_p - y_1 ])
r_norm = sqrt( r[0]**2 + r[1]**2 )
r_hat = r / r_norm

B_1 = (mu_0 / 4*pi) * ((3*r_hat * ( r_hat.T @ m_1 ) - m_1) / (r_norm**3))
U_1 = simplify((-m_p.T @ B_1)[0])
U_g =  M_p * g * l*(l-x_p)
# %%
V = U_g + U_1
T = 0.5 * M_p * ( diff(x_p,t)**2 + diff(y_p,t)**2 )
L = simplify(T - V)

# %%

sol = simplify(Eq(diff(L, phi) - diff(diff(L, phi_dot),t), 0))
sol
# %%
phi_t_t = solve(sol, diff(phi,t,t))[0]

# %%

phi_t_t_func = lambdify(phi, phi_t_t, modules="numpy")
# %%
