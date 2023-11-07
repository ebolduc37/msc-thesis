#!/usr/bin/env python
# coding: utf-8

"""Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates:
Numerical Evaluation - Peierls Substitution

This script generates plots and performs the numerical evaluation of the bound on the Berry phase
by comparing a semiclassical approach with a Peierls substitution approach as described in the thesis.
"""


#===============================================================
# INITIALIZATION
#===============================================================

#--------------
# USER SELECTION
#--------------

# Choice of current pattern
Bulut_current = True

# Precision of the discrete grid
high_precision = False

#--------------
# PACKAGES
#--------------

import math, cmath
import numpy as np
from numpy import linalg as LA
from numba import jit, njit
from numba.core.errors import NumbaPerformanceWarning
import sympy as sym
from scipy.sparse import coo_matrix, hstack, vstack, bmat, bsr_matrix, linalg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

#-----------------------------------
# VARIABLES
#-----------------------------------

# Numerical analysis parameters
n = 1000 if high_precision else 200 # increments in each direction of the Brillouin zone
n_B = 1 if high_precision else 3 # increments in each direction of the magnetic field Brillouin zone
phi = 1 if Bulut_current else -1 # current pattern
p = 1 # numerator of the normalized magnetic flux
q = 1500 if high_precision else 100 # denominator of the normalized magnetic flux

# Experimental parameters
eps_d, eps_p = 2.5, 0
t_pd, t_pp = 1, -0.5
V_pd, V_pp = 2.2, 1
z_pd, z_pp = 0.04, 0.04/3
vareps_d, vareps_p = eps_d+V_pd+2*V_pp, eps_p+2*V_pd
R_pd, R_pp = V_pd*z_pd/t_pd, phi*V_pp*z_pp/t_pp

# Others
chi = p/q
k_x, k_y = sym.symbols('k_x k_y', real=True)

#-----------------------------------
# UTIL FUNCTIONS
#-----------------------------------

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


#===============================================================
# MEAN-FIELD HAMILTONIAN
#===============================================================

#-----------------------------------
# MATRICES AS SYMBOLIC STATEMENT
#-----------------------------------

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
                     [-2j*R_pd*s_x, 0, -4j*R_pp*s_x*c_y],
                     [-2j*R_pd*s_y, 4j*R_pp*c_x*s_y, 0]])

# H_MF
def H_MF_sym(k_x, k_y):
    H_0, H_0_Q, H_1 = H_0_sym(k_x, k_y), H_0_sym(k_x+sym.pi, k_y+sym.pi), H_1_sym(k_x, k_y)
    return sym.simplify(sym.Matrix([[H_0, H_1],
                                    [-H_1.T,H_0_Q]]))

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

# Matrices connected to the mean-field Hamiltonian as optimized JIT functions
H_0_k = jit(sym.lambdify([k_x, k_y], H_0_sym(k_x, k_y), 'numpy'))
H_1_k = jit(sym.lambdify([k_x, k_y], H_1_sym(k_x, k_y), 'numpy'))
H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y), 'numpy'))

# Energy bands of the mean-field Hamiltonian as a vectorized optimized JIT function
@njit
def E_MF_k(k_x, k_y):
    return LA.eigvalsh(H_MF_k(k_x, k_y))
E_MF_k = np.vectorize(E_MF_k, otypes=[np.ndarray])

#-----------------------------------
# COMPUTATION
#-----------------------------------

x = y = np.linspace(0, np.pi, n)
X, Y = np.meshgrid(x, y)

# Energy bands
E_MF = clean_bands(E_MF_k(X, Y))

# Fermi energy
E_sort = np.sort(E_MF[:2].flatten())
E_f = E_sort[len(E_sort)//2]

# Energy at which the electron & hole pockets appear
E_flat = E_MF_k(np.pi, 0)
E_elec = E_flat[-1]
E_hole = E_flat[-2]

# Energy at which both high-energy bands meet & the hole pockets disappear
E_deg = E_MF_k(np.pi/2, np.pi/2)
E_meet = E_deg[-2]

# Energy at which the electron pockets disappear
L = np.linspace(0, np.pi/2, n)
E_max = max(clean_bands_line(E_MF_k(np.pi-L, L)).flatten())


#===============================================================
# MEAN-FIELD HAMILTONIAN UNDER A MAGNETIC FIELD
#===============================================================

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

# Diagonal sub-block matrices
def H_nu_rr_k(nu: int, r: int, k_x: float, k_y: float):
    f_xd = -(t_pd+1j*R_pd*(-1)**nu)*np.exp(1j*2*np.pi*chi*(r+(4*nu-1)/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x-k_y))
    f_yd = (t_pd-1j*R_pd*(-1)**nu)*np.exp(1j*2*np.pi*chi*(r+(4*nu+1)/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x+k_y))
    f_yx = (t_pp-1j*R_pp*(-1)**nu)*np.exp(-1j/np.sqrt(2)*k_y)
    return coo_matrix([[vareps_d, np.conjugate(f_xd), np.conjugate(f_yd)],
                       [f_xd, vareps_p, np.conjugate(f_yx)],
                       [f_yd, f_yx, vareps_p]])

def H_nu_r1r_k(nu: int, r: int, k_x: float, k_y: float):
    f_xy = (t_pp-1j*R_pp*(-1)**nu)*np.exp(-1j/np.sqrt(2)*k_y)
    return coo_matrix([[0, 0, 0.0j],
                       [0, 0, f_xy],
                       [0, 0, 0.0j]])

# Off-diagonal sub-block matrices
def H_21_rr_k(r: int, k_x: float, k_y: float):
    f_xd = (t_pd-1j*R_pd)*np.exp(-1j*2*np.pi*chi*(r+5/8))*np.exp(1j/(2*np.sqrt(2))*(k_x-k_y))
    f_dy = -(t_pd+1j*R_pd)*np.exp(1j*2*np.pi*chi*(r+7/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x+k_y))
    f_xy = -2*(t_pp*np.cos(k_x/np.sqrt(2)-4*np.pi*chi*(r+3/4))+R_pp*np.sin(k_x/np.sqrt(2)-4*np.pi*chi*(r+3/4)))
    return coo_matrix([[0, 0, f_dy],
                       [f_xd, 0, f_xy],
                       [0, 0, 0.0j]])

def H_21_r1r_k(r: int, k_x: float, k_y: float):
    f_yd = -(t_pd+1j*R_pd)*np.exp(-1j*2*np.pi*chi*(r+3/8))*np.exp(1j/(2*np.sqrt(2))*(k_x+k_y))
    f_dx = (t_pd-1j*R_pd)*np.exp(1j*2*np.pi*chi*(r+1/8))*np.exp(-1j/(2*np.sqrt(2))*(k_x-k_y))
    f_yx = -2*(t_pp*np.cos(k_x/np.sqrt(2)-4*np.pi*chi*(r+1/4))-R_pp*np.sin(k_x/np.sqrt(2)-4*np.pi*chi*(r+1/4)))
    return coo_matrix([[0, f_dx, 0],
                       [0, 0, 0.0j],
                       [f_yd, f_yx, 0]])

# Diagonal block matrices
def H_nu_k(nu: int, k_x: float, k_y: float):
    row = hstack([H_nu_rr_k(nu, 1, k_x, k_y), H_nu_r1r_k(nu, 1, k_x, k_y).T.conj()])
    row = hstack([row, coo_matrix((3,3*(q-3)))])
    row = hstack([row, H_nu_r1r_k(nu, q, k_x, k_y)])
    H_nu = row
    for r in range(2, q):
        row = hstack([coo_matrix((3,3*(r-2))), H_nu_r1r_k(nu, r-1, k_x, k_y)])
        row = hstack([row, H_nu_rr_k(nu, r, k_x, k_y)])
        row = hstack([row, H_nu_r1r_k(nu, r, k_x, k_y).T.conj()])
        row = hstack([row, coo_matrix((3,3*(q-1-r)))])
        H_nu = vstack([H_nu, row])
    row = hstack([H_nu_r1r_k(nu, q, k_x, k_y).T.conj(), coo_matrix((3,3*(r-2)))])
    row = hstack([row, H_nu_r1r_k(nu, q-1, k_x, k_y)])
    row = hstack([row, H_nu_rr_k(nu, q, k_x, k_y)])
    H_nu = vstack([H_nu, row])
    return bsr_matrix(H_nu)

# Off-diagonal block matrices
def H_21_k(k_x: float, k_y: float):
    row = hstack([H_21_rr_k(1, k_x, k_y), H_21_r1r_k(2, k_x, k_y)])
    row = hstack([row, coo_matrix((3,3*(q-2)))])
    H_21 = row
    for r in range(2, q):
        row = hstack([coo_matrix((3,3*(r-1))), H_21_rr_k(r, k_x, k_y)])
        row = hstack([row, H_21_r1r_k(r+1, k_x, k_y)])
        row = hstack([row, coo_matrix((3,3*(q-1-r)))])
        H_21 = vstack([H_21, row])
    row = hstack([H_21_r1r_k(1, k_x, k_y), coo_matrix((3,3*(q-2)))])
    row = hstack([row, H_21_rr_k(q, k_x, k_y)])
    H_21 = vstack([H_21, row])
    return bsr_matrix(H_21)

# Magnetic field Hamiltonian
def H_B_k(k_x: float, k_y: float):
    H_B = bmat([[H_nu_k(1, k_x, k_y),H_21_k(k_x, k_y).T.conj()],
                [H_21_k(k_x, k_y),H_nu_k(2, k_x, k_y)]])
    return H_B

# Energy bands of the magnetic field Hamiltonian
def E_B_k(k_x, k_y):
    E_B = linalg.eigsh(H_B_k(k_x, k_y), k=2*q, return_eigenvectors=False)
    return E_B
E_B_k = np.vectorize(E_B_k, otypes=[np.ndarray])

#-----------------------------------
# COMPUTATION
#-----------------------------------

x_B = y_B = np.pi/(2*np.sqrt(2)) if high_precision else np.linspace(0, np.pi/np.sqrt(2), n_B)
y_B = y_B/q
X_B, Y_B = np.meshgrid(x_B, y_B)

E_B = np.sort(clean_bands(E_B_k(X_B, Y_B)).flatten())
N = len(E_B)


#-----------------------------------
# VISUALIZATION
#-----------------------------------

fig = plt.figure(figsize=(5,5))
plt.title('Energy Distribution of the High-Energy Subspace of the\n'
          'Mean-Field Hamiltonian ($\phi =$' + str(phi) + ') under a Magnetic\n'
          'Field ($\chi =$' + str(p) + '/' + str(q) + ') and the Energies within which the\n'
          'Hole (Orange) and Electron (Blue) Pockets Reside')
plt.plot(np.array(range(N))*100/N, E_B, 'k.', markersize=1)
plt.axhline(y=E_hole, color='C1', linestyle='--'), plt.axhline(y=E_meet, color='C1', linestyle='--')
plt.axhline(y=E_elec, color='C0', linestyle='--'), plt.axhline(y=E_max, color='C0', linestyle='--')
plt.xlim(100, 0)
plt.xlabel(r'State Index (%)'), plt.ylabel(r'$E/t_{pd}$')
plt.tight_layout()
plt.show()


#===============================================================
# ENERGY PLATEAUX
#===============================================================

#-----------------------------------
# PRELIMINARY COMPUTATION
#-----------------------------------

# Isolating the energy distribution of the hole & electron pockets
ind, E_sub = np.array(range(N*3//10, N*6//10)), E_B[N*3//10:N*6//10]

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

# To allow easy access to the energy plateaux
def isolate_plateaux(E_min, E_max):
    
    findLim = np.where((E_min <= E_sub) & (E_sub <= E_max))
    ind_lim, E_lim = ind[findLim], E_sub[findLim]
    
    findJump = np.array(np.where(np.diff(E_lim) >= np.mean(np.diff(E_lim))/2))+1
    ind_L, E_L = ind_lim[findJump], E_lim[findJump]
    ind_L, E_L = np.append(ind_L, ind_lim[-1]), np.append(E_L, E_lim[-1])
    
    ind_plat, E_plat = [[]]*(len(ind_L)-1), [[]]*(len(ind_L)-1)
    for i in range(len(ind_L)-1):
        findLimits = np.where((ind_L[i] <= ind_lim) & (ind_lim < ind_L[i+1]))
        ind_plat[i] = ind_lim[findLimits]
        E_plat[i] = E_lim[findLimits]
            
    return (ind_plat, E_plat)

#-----------------------------------
# COMPUTATION
#-----------------------------------

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

#-----------------------------------
# VISUALIZATION
#-----------------------------------

fig = plt.figure(figsize=(5,5))
plt.title('Energy Distribution of the High-Energy Subspace of the\n'
          'Mean-Field Hamiltonian ($\phi =$' + str(phi) + ') under a Magnetic\n'
          'Field ($\chi =$' + str(p) + '/' + str(q) + ') and the Energies within which the\n'
          'Hole (Orange) and Electron (Blue) Pockets Reside with\n'
          'Corresponding Energy Plateaux within the Hole Pockets')
plt.plot(ind*100/N, E_sub, 'k.', markersize=1)
for i in range(len(ind_plat)): plt.plot(ind_plat[i]*100/N, E_plat[i], 'C1.', markersize=1)
plt.axhline(y=E_hole, color='C1', linestyle='--'), plt.axhline(y=E_meet, color='C1', linestyle='--')
plt.axhline(y=E_elec, color='C0', linestyle='--'), plt.axhline(y=E_max, color='C0', linestyle='--')
if phi == 1: plt.xlim(58, 48), plt.ylim(8.03, 8.19)
else: plt.xlim(58, 46), plt.ylim(7.975, 8.17)
plt.xlabel(r'State Index (%)'), plt.ylabel(r'$E/t_{pd}$')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(5,5))
plt.title('Energy Distribution of the High-Energy Subspace of the\n'
          'Mean-Field Hamiltonian ($\phi =$' + str(phi) + ') under a Magnetic\n'
          'Field ($\chi =$' + str(p) + '/' + str(q) + ') and the Energies within which the\n'
          'Hole (Orange) and Electron (Blue) Pockets Reside with\n'
          'Corresponding Energy Plateaux within the Hole Pockets')
plt.plot(ind*100/N, E_sub, 'k.', markersize=1)
for i in range(len(ind_plat)): plt.plot(ind_plat[i]*100/N, E_plat[i], 'C1.', markersize=1)
plt.axhline(y=E_hole, color='C1', linestyle='--'), plt.axhline(y=E_meet, color='C1', linestyle='--')
plt.axhline(y=E_elec, color='C0', linestyle='--'), plt.axhline(y=E_max, color='C0', linestyle='--')
if phi == 1: plt.xlim(50, 34), plt.ylim(7.675, 8.075)
else: plt.xlim(48, 34), plt.ylim(7.725, 8.000)
plt.xlabel(r'State Index (%)'), plt.ylabel(r'$E/t_{pd}$')
plt.tight_layout()
plt.show()


#===============================================================
# BOUND ON THE BERRY PHASE
#===============================================================

#-----------------------------------
# COMPUTATION
#-----------------------------------

# Evaluation of the mismatch between n_SC and n_PS
dA = (np.pi/(n-1))**2
n_SC = [0]*len(E_plat)
for (i, E) in [(i, np.mean(E)) for (i, E) in enumerate(E_plat)]:
    n_SC[i] = len(E_MF[1][E_MF[1] >= E])*dA/(2*chi*(2*np.pi)**2)
delta = [np.mean(n_SC-np.round(n_SC)), np.std(n_SC-np.round(n_SC),ddof=1)]

# Rounding of the mismatch in terms of the uncertainty
dec = 0
while (round(np.trunc(delta[1]*10**dec)) == 0):
    dec += 1
delta = [round(delta[0], dec), round(delta[1], dec)]

#-----------------------------------
# PRINT
#-----------------------------------

# Mismatch between n_SC and n_PS
print(delta)
