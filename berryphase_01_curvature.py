


##### INITIALIZATION

### PACKAGES

import math, cmath
import numpy as np
from numpy import linalg as LA
from numba import jit, njit
from numba.np.extensions import cross2d
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import sympy as sym
import matplotlib.colors as col
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


### PARAMETERS

# Mutable
#n = 500 # number of division increments in each direction of the Brillouin zone
n = 200 
phi = 1 # selection of the current pattern to observe
# phi = -1
xi = 10**-5 # mass term
#n_b = 2000 # number of division increments in each direction of the Brillouin zone to calculate the Berry phase
n_b = 200
side_b = [10**-3, 10**-2] # length of the sides around the degeneracy points to calculate the Berry phase
mag = 40 # magnification factor onto the Berry curvature around the degeneracy point

# Non-mutable
eps_d, eps_p = 2.5, 0
t_pd, t_pp = 1, -0.5
V_pd, V_pp = 2.2, 1
z_pd, z_pp = 0.04, 0.04/3

# Inferred & more
k_x, k_y = sym.symbols('k_x k_y', real=True)
x = y = np.linspace(-np.pi, np.pi, n)
X, Y = np.meshgrid(x, y)
Q_x = Q_y = np.pi
R_pd, R_pp = V_pd*z_pd/t_pd, V_pp*z_pp/t_pp
vareps_d, vareps_p = eps_d+V_pd+2*V_pp, eps_p+2*V_pd
p = [[-1,1],[1,1]]
side = 2*10**-4
zoom = side*np.pi/2*np.linspace(-1, 1, n)
cLim = np.zeros(2)
x_b, y_b = side_b[0]*np.pi/2*np.linspace(-1, 1, n_b), side_b[1]*np.pi/2*np.linspace(-1, 1, n_b)
X_b_ini, Y_b_ini = np.meshgrid(x_b, y_b)
p_b = [[1,1],[-1,1]]
X_b = p_b[0][0]*np.pi/2 + (X_b_ini - Y_b_ini)/np.sqrt(2)
Y_b = p_b[0][1]*np.pi/2 + (X_b_ini + Y_b_ini)/np.sqrt(2)


### FUNCTIONS

# To allow easy access to the energy bands at each point
def clean_bands(E_k):
    n_B = E_k[0,0].shape[0]
    E_r, E_f, E = E_k.flatten(), np.zeros((n_B, E_k.size)), np.zeros((n_B, E_k.shape[0], E_k.shape[1]))
    for i in range(n_B):
        for j in range(E_k.size):
            E_f[i][j] = E_r[j][(n_B-1)-i]
        E[i] = np.reshape(E_f[i], (E_k.shape[0], E_k.shape[1]))
    return E



##### MEAN-FIELD HAMILTONIAN

### MATRICES AS SYMBOLIC STATEMENTS

# H_0
def H_0_sym(k_x, k_y):
    s_x, s_y = sym.sin(k_x/2), sym.sin(k_y/2)
    return sym.Matrix([[vareps_d, 2*t_pd*s_x, -2*t_pd*s_y],
                     [2*t_pd*s_x, vareps_p, 4*t_pp*s_x*s_y],
                     [-2*t_pd*s_y, 4*t_pp*s_x*s_y, vareps_p]])

# H_1
def H_1_sym(k_x, k_y):
    s_x, s_y, c_x, c_y = sym.sin(k_x/2), sym.sin(k_y/2), sym.cos(k_x/2), sym.cos(k_y/2)
    return sym.Matrix([[0, 2j*R_pd*c_x, 2j*R_pd*c_y],
                     [-2j*R_pd*s_x, 0, -4j*phi*R_pp*s_x*c_y],
                     [-2j*R_pd*s_y, 4j*phi*R_pp*c_x*s_y, 0]])

# H_MF
def H_MF_sym(k_x, k_y):
    H_0, H_0_Q, H_1 = H_0_sym(k_x, k_y), H_0_sym(k_x+sym.pi, k_y+sym.pi), H_1_sym(k_x, k_y)
    return sym.simplify(sym.Matrix([[H_0, H_1],
                                    [-H_1.T,H_0_Q]]))


### OPTIMIZED COMPUTATION

# Matrices connected to the mean-field Hamiltonian as optimized JIT functions
H_0_k = jit(sym.lambdify([k_x, k_y], H_0_sym(k_x, k_y), 'numpy'))
H_1_k = jit(sym.lambdify([k_x, k_y], H_1_sym(k_x, k_y), 'numpy'))
H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y), 'numpy'))

# Energy bands of the mean-field Hamiltonian as a vectorized optimized JIT function
@njit
def E_MF_k(k_x, k_y):
    return LA.eigvalsh(H_MF_k(k_x, k_y))
E_MF_k = np.vectorize(E_MF_k, otypes=[np.ndarray])

# Evaluation of the mean-field energy bands
E_MF = clean_bands(E_MF_k(X, Y))


### VISUALIZATION

fig = plt.figure(figsize=(10,5))
fig.suptitle('Energy Bands of the Mean-Field Hamiltonian')

# 1st plot: all mean-field energy bands
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.title.set_text('All Energy Bands')
for i in range(6):
    ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
    
# 2nd plot: upper mean-Field energy bands
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.title.set_text('Upper Energy Bands')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')

plt.show()



##### FERMI SURFACE

### COMPUTATION

E_sort = np.sort(E_MF[:2].flatten())
E_f = E_sort[len(E_sort)//2]
E_max = E_sort[-1]
E_min = E_MF[5].min()

### VISUALIZATION

fig = plt.figure(figsize=(10,5))
fig.suptitle('Fermi Energy of the Mean-Field Hamiltonian')

# 1st plot: Fermi pockets
ax = fig.add_subplot(1, 2, 1)
ax.title.set_text('Fermi Pockets of the Upper Energy Bands')
ax.contourf(X/np.pi, Y/np.pi, E_MF[0], levels=[E_min,E_f], colors=('C0'), alpha=0.75)
ax.contourf(X/np.pi, Y/np.pi, E_MF[1], levels=[E_f,E_max], colors=('C1'), alpha=0.75)
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$')
ax.grid(alpha=0.75)
    
# 2nd plot: upper mean-field energy bands with the Fermi energy represented as a plane
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.title.set_text('Upper Energy Bands')
for i in range(2): ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.plot_surface(X/np.pi, Y/np.pi, E_f*np.ones((n,n)), alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')

plt.show()



##### PROJECTED HAMILTONIAN

### OPTIMIZED COMPUTATION

# Projected Hamiltonian as an optimized JIT function
@njit
def H_U_k(k_x, k_y):
    H_0, H_0_Q = H_0_k(k_x, k_y), H_0_k(k_x+Q_x, k_y+Q_y)
    (E_0, n_0), (E_0_Q, n_0_Q) = LA.eigh(H_0), LA.eigh(H_0_Q)
    n_0, n_0_Q = n_0.astype(np.complex128), n_0_Q.astype(np.complex128)
    n_up, n_down = np.concatenate((n_0[:,2], np.array([0,0,0]))), np.concatenate((np.array([0,0,0]), n_0_Q[:,2]))
    return np.array([[n_up.conj()@H_MF_k(k_x, k_y)@n_up, n_up.conj()@H_MF_k(k_x, k_y)@n_down],
                     [n_down.conj()@H_MF_k(k_x, k_y)@n_up, n_down.conj()@H_MF_k(k_x, k_y)@n_down]])

# Energy bands of the projected Hamiltonian as a vectorized optimized JIT function
@njit
def E_U_k(k_x, k_y):
    return LA.eigvalsh(H_U_k(k_x, k_y))
E_U_k = np.vectorize(E_U_k, otypes=[np.ndarray])

# Evaluation of the projected energy bands
E_U = clean_bands(E_U_k(X, Y))


### VISUALIZATION

fig = plt.figure(figsize=(10,5))
fig.suptitle('Projection of the Mean-Field Hamiltonian on the Upper Energy Bands Subspace')

# 1st plot: projected energy bands
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.title.set_text('Projection of the Upper Mean-Field Energy Bands')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_U[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
    
# 2nd plot: upper mean-field energy bands
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.title.set_text('Upper Mean-Field Energy Bands')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')

plt.show()



##### MEAN-FIELD HAMILTONIAN WITH MASS TERM TO LIFT THE DEGENERACIES IN THE UPPER BANDS

### OPTIMIZED COMPUTATION

# Mean-field Hamiltonian with mass term as an optimized JIT function
@njit
def H_xi_k(k_x, k_y):
    (E_0, n_0), (E_0_Q, n_0_Q) = LA.eigh(H_0_k(k_x, k_y)), LA.eigh(H_0_k(k_x+Q_x, k_y+Q_y))
    m = xi*np.outer(n_0[:,2],n_0_Q[:,2].conj())
    M = np.concatenate((np.concatenate((np.zeros((3,3)), m), axis=1),
                        np.concatenate((m.conj().T, np.zeros((3,3))), axis=1)), axis=0)
    return H_MF_k(k_x, k_y) + M

# Evaluation of the mean-field energy bands with mass term as a vectorized optimized JIT function
@njit
def E_xi_k(k_x, k_y):
    return LA.eigvalsh(H_xi_k(k_x, k_y))
E_xi_k = np.vectorize(E_xi_k, otypes=[np.ndarray])

# Evaluation of the mean-field energy bands with mass term
E_xi = clean_bands(E_xi_k(X, Y))


### VISUALIZATION

fig = plt.figure(figsize=(10,5))
fig.suptitle('Lifting the Degeneracies of the Mean-Field Hamiltonian')

# 1st plot: upper mean-field energy bands with mass term
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.title.set_text('Upper Energy Bands with Mass Term')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_xi[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
    
# 2nd plot: upper mean-field energy bands
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.title.set_text('Upper Energy Bands without Mass Term')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')

plt.show()



##### PROJECTED MEAN-FIELD HAMILTONIAN WITH MASS TERM

### OPTIMIZED COMPUTATION

# Projected Hamiltonian with mass term as an optimized JIT function
@njit
def H_U_xi_k(k_x, k_y):
    H_0, H_0_Q = H_0_k(k_x, k_y), H_0_k(k_x+Q_x, k_y+Q_y)
    (E_0, n_0), (E_0_Q, n_0_Q) = LA.eigh(H_0), LA.eigh(H_0_Q)
    n_0, n_0_Q = n_0.astype(np.complex128), n_0_Q.astype(np.complex128)
    n_up, n_down = np.concatenate((n_0[:,2], np.array([0,0,0]))), np.concatenate((np.array([0,0,0]), n_0_Q[:,2]))
    return np.array([[n_up.conj()@H_xi_k(k_x, k_y)@n_up, n_up.conj()@H_xi_k(k_x, k_y)@n_down],
                     [n_down.conj()@H_xi_k(k_x, k_y)@n_up, n_down.conj()@H_xi_k(k_x, k_y)@n_down]])

# Evaluation of the projected energy bands with mass term as a vectorized optimized JIT function
@njit
def E_U_xi_k(k_x, k_y):
    return LA.eigvalsh(H_U_xi_k(k_x, k_y))
E_U_xi_k = np.vectorize(E_U_xi_k, otypes=[np.ndarray])

# Evaluation of the projected energy bands with mass term
E_U_xi = clean_bands(E_U_xi_k(X, Y))


### VISUALIZATION

fig = plt.figure(figsize=(10,5))
fig.suptitle('Projection of the Mean-Field Hamiltonian with Mass Term on the Higher Energy Subspace')

# 1st plot: projected energy bands with mass term
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.title.set_text('Projection of the Upper Energy Bands with Mass Term')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_U_xi[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
    
# 2nd plot: upper mean-field energy bands
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.title.set_text('Upper Energy Bands without Mass Term')
for i in range(2):
    ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$k_x/\pi$'), ax.set_ylabel('$k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')

plt.show()



##### BERRY PHASE

### OPTIMIZED COMPUTATION

# Gradient of the mean-field Hamiltonian as optimized JIT functions
Dx_H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y).diff(k_x), 'numpy'))
Dy_H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y).diff(k_y), 'numpy'))

# Evaluation of the z-component of the Berry curvature as a vectorized optimized JIT function
@njit
def Bz_k(k_x, k_y):
    (E_MF, n_MF), (E_xi, n_xi) = LA.eigh(H_MF_k(k_x, k_y)), LA.eigh(H_xi_k(k_x, k_y))
    DH_MF = [Dx_H_MF_k(k_x, k_y), Dy_H_MF_k(k_x, k_y)]
    DH_MF_ij = np.zeros((6,6,2), dtype='complex_')
    for i in range(6):
        for j in range(6):
            DH_MF_ij[i,j] = [n_xi[:, i].conj()@DH_MF[0]@n_xi[:, j], n_xi[:, i].conj()@DH_MF[1]@n_xi[:, j]]
    B_z = np.zeros((6), dtype='complex_')
    for i in range(6):
        for j in list(range(0,i))+list(range(i+1,6)):
            if E_xi[i] == E_xi[j]: continue # it may happen with the lower bands
            B_z[i] -= cross2d(DH_MF_ij[i,j], DH_MF_ij[j,i]).imag/(E_xi[i]-E_xi[j])**2
    return B_z.real
Bz_k = np.vectorize(Bz_k, otypes=[np.ndarray])


### VISUALIZATION

fig, axs = plt.subplots(1, 2, figsize=(12,5), sharey=True)
fig.suptitle('Z-Component of the Berry Curvature of the Lowest Upper Energy Band')

# 1st plot: z-component of the Berry curvature of the lowest upper energy band around (-pi/2a, pi/2a)
x_zoom, y_zoom = p[0][0]*np.pi/2+zoom, p[0][1]*np.pi/2+zoom
X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
Bz = clean_bands(Bz_k(X_zoom, Y_zoom))
c0 = axs[0].pcolormesh(Bz[1], cmap ='coolwarm')
cLim[0] = np.max(np.abs(c0.get_clim()))
axs[0].set_xticks(np.linspace(0, n, 5)), axs[0].set_yticks(np.linspace(0, n, 5))
xTick = (p[0][0]/2+side/2*np.linspace(-1, 1, 5)).tolist()
xTick[0] = xTick[-1] = ''
yTick = (p[0][1]/2+side/2*np.linspace(-1, 1, 5)).tolist()
axs[0].set_xticklabels(xTick), axs[0].set_yticklabels(yTick)
axs[0].set_xlabel(r'$k_x/\pi$'), axs[0].set_ylabel(r'$k_y/\pi$')

# 2nd plot: z-component of the Berry curvature of the lowest upper energy band around (pi/2a, pi/2a)
x_zoom, y_zoom = p[1][0]*np.pi/2+zoom, p[1][1]*np.pi/2+zoom
X_zoom, Y_zoom = np.meshgrid(x_zoom, y_zoom)
Bz = clean_bands(Bz_k(X_zoom, Y_zoom))
c1 = axs[1].pcolormesh(Bz[1], cmap ='coolwarm')
cLim[1] = np.max(np.abs(c1.get_clim()))
axs[1].set_xticks(np.linspace(0, n, 5))
xTick = (p[1][0]/2+side/2*np.linspace(-1, 1, 5)).tolist()
xTick[0] = xTick[-1] = ''
axs[1].set_xticklabels(xTick)
axs[1].set_xlabel(r'$k_x/\pi$')

cMax = np.max(cLim)
c0.set_clim(-cMax, cMax), c1.set_clim(-cMax, cMax)
fig.colorbar(c1, ax=axs)
plt.show()



##### BERRY PHASE

### COMPUTATION

# Evaluation of the Berry phase of the lowest upper energy band around (pi/2a, pi/2a)
# Counterclockwise orbit in percentage of pi
# For simplicity, momentum space has been translated to (pi/2a, pi/2a) and rotated by pi/4 clockwise
Bz = clean_bands(Bz_k(X_b, Y_b))
dA = side_b[0]*side_b[1]*(np.pi/(n_b-1))**2
berry_phase = [100*sum(Bz[1].flatten())*dA/np.pi, 100/np.sqrt(((side_b[0]/2)/xi)**2+1)/np.pi]


### VISUALIZATION

fig, axs = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle('Z-Component of the Berry Curvature of the Lowest Upper Energy Band')

# 1st plot: z-component of the Berry curvature of the lowest upper energy band around (pi/2a, pi/2a)
c0 = axs[0].pcolormesh(Bz[1], cmap='hot')
cLim[0] = np.max(np.abs(c0.get_clim()))
axs[0].set_xticks(np.linspace(0,n_b,5)), axs[0].set_yticks(np.linspace(0,n_b,5))
xTick = (side_b[0]/2*np.linspace(-1, 1, 5)).tolist()
xTick[0] = xTick[-1] = ''
yTick = (side_b[1]/2*np.linspace(-1, 1, 5)).tolist()
axs[0].set_xticklabels(xTick), axs[0].set_yticklabels(yTick)
axs[0].set_xlabel(r'$k_x/\pi$'), axs[0].set_ylabel(r'$k_y/\pi$')

# 2nd plot: zoom on the z-component of the Berry curvature of the lowest upper energy band around (pi/2a, pi/2a)
c1 = axs[1].pcolormesh(Bz[1], cmap='hot')
cLim[1] = np.max(np.abs(c1.get_clim()))
axs[1].set_xlim(n_b/2*(1-1/mag), n_b/2*(1+1/mag)), axs[1].set_ylim(n_b/2*(1-1/mag), n_b/2*(1+1/mag))
axs[1].set_xticks(np.linspace(n_b/2*(1-1/mag), n_b/2*(1+1/mag), 5))
axs[1].set_yticks(np.linspace(n_b/2*(1-1/mag), n_b/2*(1+1/mag), 5))
xTick = [-side_b[0]/(2*mag), -side_b[0]/(4*mag), 0, side_b[0]/(4*mag), side_b[0]/(2*mag)]
xTick[0] = xTick[-1] = ''
yTick = [-side_b[1]/(2*mag), -side_b[1]/(4*mag), 0, side_b[1]/(4*mag), side_b[1]/(2*mag)]
axs[1].set_xticklabels(xTick), axs[1].set_yticklabels(yTick)
axs[1].set_xlabel(r'$k_x/\pi$'), axs[1].set_ylabel(r'$k_y/\pi$')

cMax = np.max(cLim)
c0.set_clim(0, cMax), c1.set_clim(0, cMax)
fig.colorbar(c1, ax=axs)
plt.show()



##### BERRY PHASE IN PERCENTAGE OF PI

# Rounding of the Berry phase in terms of the uncertainty
dec = 0
while (round(np.trunc(berry_phase[1]*10**dec)) == 0):
    dec += 1
berry_phase = [round(berry_phase[0], dec), round(berry_phase[1], dec)]

# Berry phase in percentage of pi for the lowest upper energy band around (pi/2a, pi/2a)
print(berry_phase)


