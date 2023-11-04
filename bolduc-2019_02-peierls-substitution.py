


##### INITIALIZATION

### PACKAGES

import math, cmath
import numpy as np
from numpy import linalg as LA
from numba import jit, njit
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
import sympy as sym
from scipy.sparse import linalg
import matplotlib.pyplot as plt


### PARAMETERS

# Mutable
#n = 1000 # number of division increments in each direction of the Brillouin zone
n = 200
n_B = 2 # number of division increments in each direction of the magnetic field Brillouin zone
phi = 1  # selection of the current pattern to observe
# phi = -1
p = 1 # numerator of the normalized magnetic flux
#q = 1500 # denominator of the normalized magnetic flux
q = 200

# Non-mutable
eps_d, eps_p = 2.5, 0
t_pd, t_pp = 1, -0.5
V_pd, V_pp = 2.2, 1
z_pd, z_pp = 0.04, 0.04/3

# Inferred & more
chi = p/q
R_pd, R_pp = V_pd*z_pd/t_pd, V_pp*z_pp/t_pp
vareps_d, vareps_p = eps_d+V_pd+2*V_pp, eps_p+2*V_pd
k_x, k_y = symbols('k_x k_y', real=True)
x = y = np.linspace(0, np.pi, n)
X, Y = np.meshgrid(x, y)
L = np.linspace(0, np.pi/2, n)
x_B = y_B = np.linspace(0, np.pi/np.sqrt(2), n_B)
y_B = y_B/q
X_B, Y_B = np.meshgrid(x_B, y_B)


### FUNCTIONS

# To allow easy access to the energy bands at each point on a surface
def clean_bands(E_k):
    n_B = E_k[0,0].shape[0]
    E_r, E_f, E = E_k.flatten(), np.zeros((n_B, E_k.size)), np.zeros((n_B, E_k.shape[0], E_k.shape[1]))
    for i in range(n_B):
        for j in range(E_k.size):
            E_f[i][j] = E_r[j][(n_B-1)-i]
        E[i] = np.reshape(E_f[i], (E_k.shape[0], E_k.shape[1]))
    return E

# To allow easy access to the energy bands at each point on a line
def clean_bands_line(E_k):
    n_B = E_k[0].shape[0]
    E_r, E_f, E = E_k.flatten(), np.zeros((n_B, E_k.size)), np.zeros((n_B, E_k.shape[0]))
    for i in range(n_B):
        for j in range(E_k.size):
            E_f[i][j] = E_r[j][(n_B-1)-i]
        E[i] = np.reshape(E_f[i], (E_k.shape[0]))
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


### ENERGETIC MILESTONES

# Evaluation of the Fermi energy
E_sort = np.sort(E_MF[:2].flatten())
E_f = E_sort[len(E_sort)//2]

# Evaluation of the energy at which the electron & hole pockets appear
E_flat = E_MF_k(np.pi, 0)
E_elec = E_flat[-1]
E_hole = E_flat[-2]

# Evaluation of the energy at which both upper bands meet & hole pockets disappear
E_deg = E_MF_k(np.pi/2, np.pi/2)
E_meet = E_deg[-2]

# Evaluation of the energy at which the electron pockets disappear
E_max = max(clean_bands_line(E_MF_k(np.pi-L, L)).flatten())



##### MAGNETIC FIELD HAMILTONIAN

### OPTIMIZED COMPUTATION

# Diagonal sub-block matrices as an optimized JIT functions
@njit
def H_nu_rr_k(nu: int, r: int, k_x: float, k_y: float):
    f_xd = -(t_pd+1j*R_pd*(-1)**nu)*np.exp(1j*2*np.pi*chi*(r+(4*nu-1)/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x-k_y))
    f_yd = (t_pd-1j*R_pd*(-1)**nu)*np.exp(1j*2*np.pi*chi*(r+(4*nu+1)/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x+k_y))
    f_yx = (t_pp-1j*phi*R_pp*(-1)**nu)*np.exp(-1j/np.sqrt(2)*k_y)
    return np.array([[vareps_d, np.conjugate(f_xd), np.conjugate(f_yd)],
                     [f_xd, vareps_p, np.conjugate(f_yx)],
                     [f_yd, f_yx, vareps_p]])

@njit
def H_nu_r1r_k(nu: int, r: int, k_x: float, k_y: float):
    f_xy = (t_pp-1j*phi*R_pp*(-1)**nu)*np.exp(-1j/np.sqrt(2)*k_y)
    return np.array([[0, 0, 0.0j],
                     [0, 0, f_xy],
                     [0, 0, 0.0j]])

# Off-diagonal sub-block matrices as an optimized JIT functions
@njit
def H_21_rr_k(r: int, k_x: float, k_y: float):
    f_xd = (t_pd-1j*R_pd)*np.exp(-1j*2*np.pi*chi*(r+5/8))*np.exp(1j/(2*np.sqrt(2))*(k_x-k_y))
    f_dy = -(t_pd+1j*R_pd)*np.exp(1j*2*np.pi*chi*(r+7/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x+k_y))
    f_xy = -2*(t_pp*np.cos(k_x/np.sqrt(2)-4*np.pi*chi*(r+3/4))+phi*R_pp*np.sin(k_x/np.sqrt(2)-4*np.pi*chi*(r+3/4)))
    return np.array([[0, 0, f_dy],
                     [f_xd, 0, f_xy],
                     [0, 0, 0.0j]])

@njit
def H_21_r1r_k(r: int, k_x: float, k_y: float):
    f_yd = -(t_pd+1j*R_pd)*np.exp(-1j*2*np.pi*chi*(r+3/8))*np.exp(1j/(2*np.sqrt(2))*(k_x+k_y))
    f_dx = (t_pd-1j*R_pd)*np.exp(1j*2*np.pi*chi*(r+1/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x-k_y))
    f_yx = -2*(t_pp*np.cos(k_x/np.sqrt(2)-4*np.pi*chi*(r+1/4))-phi*R_pp*np.sin(k_x/np.sqrt(2)-4*np.pi*chi*(r+1/4)))
    return np.array([[0, f_dx, 0],
                     [0, 0, 0.0j],
                     [f_yd, f_yx, 0]])

# Diagonal block matrices as an optimized JIT function
@njit
def H_nu_k(nu: int, k_x: float, k_y: float):
    row = np.concatenate((H_nu_rr_k(nu, 1, k_x, k_y), H_nu_r1r_k(nu, 1, k_x, k_y).T.conj()), axis=1)
    row = np.concatenate((row, np.zeros((3,3*(q-3)))), axis=1)
    row = np.concatenate((row, H_nu_r1r_k(nu, q, k_x, k_y)), axis=1)
    H_nu = row
    for r in range(2, q):
        row = np.concatenate((np.zeros((3,3*(r-2))), H_nu_r1r_k(nu, r-1, k_x, k_y)), axis=1)
        row = np.concatenate((row, H_nu_rr_k(nu, r, k_x, k_y)), axis=1)
        row = np.concatenate((row, H_nu_r1r_k(nu, r, k_x, k_y).T.conj()), axis=1)
        row = np.concatenate((row, np.zeros((3,3*(q-1-r)))), axis=1)
        H_nu = np.concatenate((H_nu, row), axis=0)
    row = np.concatenate((H_nu_r1r_k(nu, q, k_x, k_y).T.conj(), np.zeros((3,3*(r-2)))), axis=1)
    row = np.concatenate((row, H_nu_r1r_k(nu, q-1, k_x, k_y)), axis=1)
    row = np.concatenate((row, H_nu_rr_k(nu, q, k_x, k_y)), axis=1)
    H_nu = np.concatenate((H_nu, row), axis=0)
    return H_nu

# Off-diagonal block matrices as an optimized JIT function
@njit
def H_21_k(k_x: float, k_y: float):
    row = np.concatenate((H_21_rr_k(1, k_x, k_y), H_21_r1r_k(2, k_x, k_y)), axis=1)
    row = np.concatenate((row, np.zeros((3,3*(q-2)))), axis=1)
    H_21 = row
    for r in range(2, q):
        row = np.concatenate((np.zeros((3,3*(r-1))), H_21_rr_k(r, k_x, k_y)), axis=1)
        row = np.concatenate((row, H_21_r1r_k(r+1, k_x, k_y)), axis=1)
        row = np.concatenate((row, np.zeros((3,3*(q-1-r)))), axis=1)
        H_21 = np.concatenate((H_21, row), axis=0)
    row = np.concatenate((H_21_r1r_k(1, k_x, k_y), np.zeros((3,3*(q-2)))), axis=1)
    row = np.concatenate((row, H_21_rr_k(q, k_x, k_y)), axis=1)
    H_21 = np.concatenate((H_21, row), axis=0)
    return H_21

# Magnetic field Hamiltonian as an optimized JIT function
@njit
def H_B_k(k_x: float, k_y: float):
    H_B = np.concatenate((H_nu_k(1, k_x, k_y), H_21_k(k_x, k_y).T.conj()), axis=1)
    row = np.concatenate((H_21_k(k_x, k_y), H_nu_k(2, k_x, k_y)), axis=1)
    H_B = np.concatenate((H_B, row), axis=0)
    return H_B

# Energy bands of the magnetic field Hamiltonian as a vectorized optimized JIT function
def E_B_k(k_x, k_y):
    E_B, n_B = linalg.eigsh(H_B_k(k_x, k_y), k=2*q)
    E_B = E_B[::-1]
    return E_B
E_B_k = np.vectorize(E_B_k, otypes=[np.ndarray])

# Evaluation of the magnetic field energy bands
E_B = np.sort(clean_bands(E_B_k(X_B, Y_B)).flatten())
N = len(E_B)


### VISUALIZATION

fig = plt.figure(figsize=(10,10))
plt.title('Energy Distribution of the Upper Energy Bands of the Mean-Field Hamiltonian under a Magnetic Field')


# Magnetic field Hamiltonian upper energy bands distribution
plt.plot(np.array(range(N))*100/N, E_B, 'k.', markersize=1)
plt.axhline(y=E_hole, color='C1', linestyle='--'), plt.axhline(y=E_meet, color='C1', linestyle='--')
plt.axhline(y=E_elec, color='C0', linestyle='--'), plt.axhline(y=E_max, color='C0', linestyle='--')
plt.xlim(0, 100)
plt.xlabel(r'State Index (%)'), plt.ylabel(r'$E/t_{pd}$')
plt.show()



##### BOUND ON THE BERRY PHASE

### COMPUTATION

# Isolating the energy distribution of the hole & electron pockets
ind, E_sub = np.array(range(N*3//10, N*6//10)), E_B[N*3//10:N*6//10]

# To allow easy access to the energy plateaux
def isolate_plateaux(E_min, E_max):
    
    findLim = np.where((E_min <= E_sub) & (E_sub <= E_max))
    ind_lim, E_lim = ind[findLim], E_sub[findLim]
    
    findJump = np.where(np.diff(E_lim) >= np.mean(np.diff(E_lim)))
    ind_L, E_L = ind_lim[findJump], E_lim[findJump]
    ind_L, E_L = np.append(ind_L, ind_lim[-1]), np.append(E_L, E_lim[-1])
    
    ind_plat, E_plat = [[]]*(len(ind_L)-1), [[]]*(len(ind_L)-1)
    for i in range(len(ind_L)-1):
        findLimits = np.where((ind_L[i] <= ind_lim) & (ind_lim < ind_L[i+1]))
        ind_plat[i] = ind_lim[findLimits]
        E_plat[i] = E_lim[findLimits]
            
    return (ind_plat, E_plat)

# Evalutation of the energy plateaux of the hole & electron pockets
(ind_low, E_low) = isolate_plateaux(E_hole+(E_elec-E_hole)/10, E_elec)
(ind_mix, E_mix) = isolate_plateaux(E_elec, E_meet)

# Isolation of the energy plateaux of the hole pockets
ind_plat, E_plat = ind_low, E_low
plat_lim = np.mean([np.mean([len(E) for E in E_low]), np.mean([len(E) for E in E_mix])])
ind_temp, E_temp = [], []
for (i, E) in [(i, E) for (i, E) in enumerate(E_mix) if len(E) > plat_lim]:
    ind_temp.append(ind_mix[i]), E_temp.append(E)
ind_plat.extend(ind_temp), E_plat.extend(E_temp)
ind_plat, E_plat = ind_plat[::-1], E_plat[::-1]

# Evaluation of the bound on a Berry phase of pi in percentage of pi
dA = (np.pi/(n-1))**2
n_SC = [0]*len(E_plat)
for (i, E) in [(i, np.mean(E)) for (i, E) in enumerate(E_plat)]:
    n_SC[i] = len(E_MF[1][E_MF[1] >= E])*dA/(2*chi*(2*np.pi)**2)
delta = [100*2*np.mean(n_SC-np.round(n_SC)), 100*2*np.std(n_SC-np.round(n_SC))]


### VISUALIZATION
fig = plt.figure(figsize=(10,10))
plt.title('Energy Distribution of the Upper Energy Bands of the Mean-Field Hamiltonian under a Magnetic Field')

# Magnetic field Hamiltonian energy distribution of the hole & electron pockets
plt.plot(ind*100/N, E_sub, 'k.', markersize=1)
for i in range(len(ind_plat)): plt.plot(ind_plat[i]*100/N, E_plat[i], 'C1.', markersize=1)
plt.axhline(y=E_hole, color='C1', linestyle='--'), plt.axhline(y=E_meet, color='C1', linestyle='--')
plt.axhline(y=E_elec, color='C0', linestyle='--'), plt.axhline(y=E_max, color='C0', linestyle='--')
plt.xlim(30, 60)
plt.xlabel(r'State Index (%)'), plt.ylabel(r'$E/t_{pd}$')
plt.show()



##### DIFFERENCE FROM A BERRY PHASE OF PI IN PERCENTAGE OF PI

# Rounding of the delta term in terms of the uncertainty
dec = 0
while (round(np.trunc(delta[1]*10**dec)) == 0):
    dec += 1
delta = [round(delta[0], dec), round(delta[1], dec)]

# Difference from a Berry phase of pi in percentage of pi for the lowest upper energy band around (pi/2a, pi/2a)
print(delta)


