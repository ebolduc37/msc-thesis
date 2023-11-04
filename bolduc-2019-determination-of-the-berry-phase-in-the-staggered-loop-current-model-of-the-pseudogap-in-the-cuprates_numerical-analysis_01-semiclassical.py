#!/usr/bin/env python
# coding: utf-8

"""Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates:
Numerical Evaluation - Semiclassical Approach

This script generates plots and performs the numerical evaluation of the Berry phase
following a semiclassical approach as described in the thesis.
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
high_precision = True

#--------------
# PACKAGES
#--------------

import math, cmath
import numpy as np
from numpy import linalg as LA
from numba import jit, njit
from numba.np.extensions import cross2d
from numba.core.errors import NumbaPerformanceWarning
import sympy as sym
import matplotlib.colors as col
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=UserWarning)

#-----------------------------------
# VARIABLES
#-----------------------------------

# Numerical analysis parameters
n = 500 if high_precision else 200 # increments in each direction of the Brillouin zone
phi = 1 if Bulut_current else -1 # current pattern
xi = 10**-5 # mass term
n_B = 2000 if high_precision else 200 # increments in each direction to calculate the Berry phase
side_B = [10**-3, 10**-2] # side lengths around the degeneracy point to calculate the Berry phase
mag = 40 # magnification factor onto the Berry curvature around the degeneracy point

# Experimental parameters
eps_d, eps_p = 2.5, 0
t_pd, t_pp = 1, -0.5
V_pd, V_pp = 2.2, 1
z_pd, z_pp = 0.04, 0.04/3
vareps_d, vareps_p = eps_d+V_pd+2*V_pp, eps_p+2*V_pd
R_pd, R_pp = V_pd*z_pd/t_pd, phi*V_pp*z_pp/t_pp

# Others
Q_x = Q_y = np.pi
k_x, k_y = sym.symbols('k_x k_y', real=True)
cLim = np.zeros(2)
eps_E = 10**-4
side = 2*10**-4
point_L, point_R = [-1,1], [1,1]
point_B = [1,1]

#-----------------------------------
# UTIL FUNCTION
#-----------------------------------

# To allow easy access to the energy bands at each point
def clean_bands(E_k):
    n_b = E_k[0,0].shape[0]
    E_r, E_f, E = E_k.flatten(), np.zeros((n_b, E_k.size)), np.zeros((n_b, E_k.shape[0], E_k.shape[1]))
    for i in range(n_b):
        for j in range(E_k.size):
            E_f[i][j] = E_r[j][(n_b-1)-i]
        E[i] = np.reshape(E_f[i], (E_k.shape[0], E_k.shape[1]))
    return E


#===============================================================
# KINETIC ENERGY PART OF THE MEAN-FIELD HAMILTONIAN
#===============================================================

#-----------------------------------
# MATRIX AS SYMBOLIC STATEMENT
#-----------------------------------

# H_0
def H_0_sym(k_x, k_y):
    s_x, s_y = sym.sin(k_x/2), sym.sin(k_y/2)
    return sym.Matrix([[vareps_d, 2*t_pd*s_x, -2*t_pd*s_y],
                     [2*t_pd*s_x, vareps_p, 4*t_pp*s_x*s_y],
                     [-2*t_pd*s_y, 4*t_pp*s_x*s_y, vareps_p]])

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

# Matrix H_0 connected to the kinetic energy as optimized JIT functions
H_0_k = jit(sym.lambdify([k_x, k_y], H_0_sym(k_x, k_y), 'numpy'))

# Energy bands of H_0 as a vectorized optimized JIT function
@njit
def E_0_k(k_x, k_y):
    return LA.eigvalsh(H_0_k(k_x, k_y))
E_0_k = np.vectorize(E_0_k, otypes=[np.ndarray])

#-----------------------------------
# COMPUTATION
#-----------------------------------

x = y = np.linspace(-np.pi, np.pi, n)
X, Y = np.meshgrid(x, y)

E_0 = clean_bands(E_0_k(X, Y))
E_0_Q = clean_bands(E_0_k(X+Q_x, Y+Q_y))

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Energy bands of kinetic part
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('Energy Bands of the\nKinetic Part of the\nMean-Field Hamiltonian');
for i in range(3): ax.plot_surface(X/np.pi, Y/np.pi, E_0[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xticks(np.linspace(-1, 1, 5)), ax.set_yticks(np.linspace(-1, 1, 5))
ax.view_init(5, 60)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('$|E_n(k) - E_n(k+Q)| < $' + str(eps_E))
for i in range(3): ax.contourf(X/np.pi, Y/np.pi, E_0[i]-E_0_Q[i], levels=[-eps_E, eps_E], alpha=0.75)
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$')
ax.grid(alpha=0.75)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()
plt.show()


#===============================================================
# MEAN-FIELD HAMILTONIAN
#===============================================================

#-----------------------------------
# MATRICES AS SYMBOLIC STATEMENTS
#-----------------------------------

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

x_Q1 = y_Q1 = np.linspace(0, np.pi, n)
X_Q1, Y_Q1 = np.meshgrid(x_Q1, y_Q1)

E_MF = clean_bands(E_MF_k(X, Y))
E_MF_Q1 = clean_bands(E_MF_k(X_Q1, Y_Q1))

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Energy bands
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('Energy Bands of the\nMean-Field Hamiltonian ($\phi =$' + str(phi) + ')')
for i in range(6): ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xticks(np.linspace(-1, 1, 5)), ax.set_yticks(np.linspace(-1, 1, 5))
ax.view_init(5, 60)
plt.tight_layout()
plt.show()
    
# High-energy subspace
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('High-Energy Subspace of the\nMean-Field Hamiltonian ($\phi =$' + str(phi) + ')')
for i in range(2): ax.plot_surface(X/np.pi, Y/np.pi, E_MF[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xticks(np.linspace(-1, 1, 5)), ax.set_yticks(np.linspace(-1, 1, 5))
ax.view_init(5, 60)
plt.tight_layout()
plt.show()

# First quadrant of the high-energy subspace
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('First Quadrant of the\nHigh-Energy Subspace of the\nMean-Field Hamiltonian ($\phi =$' + str(phi) + ')')
for i in range(2):
    ax.plot_surface(X_Q1/np.pi, Y_Q1/np.pi, E_MF_Q1[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(0,1), ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 1, 5)), ax.set_yticks(np.linspace(0, 1, 5))
ax.view_init(5, 60)
plt.tight_layout()
plt.show()


#===============================================================
# FERMI SURFACE
#===============================================================

#-----------------------------------
# COMPUTATION
#-----------------------------------

E_sort = np.sort(E_MF_Q1[:2].flatten())
E_f = (E_sort[n**2-1]+E_sort[n**2])/2
E_max = E_MF_Q1[0].max()
E_min = E_MF_Q1[1].min()

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Fermi energy in the first quadrant of the high-energy subspace
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('Fermi Energy in the First Quadrant\nof the High-Energy Subspace of the\n'
             'Mean-Field Hamiltonian ($\phi =$' + str(phi) + ')')
for i in range(2): ax.plot_surface(X_Q1/np.pi, Y_Q1/np.pi, E_MF_Q1[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.plot([0,1], [1,0], [E_f,E_f], label='parametric curve', color='r')
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(0,1), ax.set_ylim(0,1)
ax.set_xticks(np.linspace(0, 1, 5)), ax.set_yticks(np.linspace(0, 1, 5))
ax.view_init(0, 45)
plt.tight_layout()
plt.show()

# Fermi pockets
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('Fermi Pockets of the\nMean-Field Hamiltonian ($\phi =$' + str(phi) + ')')
ax.contourf(X/np.pi, Y/np.pi, E_MF[0], levels=[E_min,E_f], colors=('C0'), alpha=0.75)
ax.contourf(X/np.pi, Y/np.pi, E_MF[1], levels=[E_f,E_max], colors=('C1'), alpha=0.75)
ax.set_xlim(-1,1), ax.set_ylim(-1,1)
ax.set_xlabel('$a k_x/\pi$'), ax.set_ylabel('$a k_y/\pi$')
ax.grid(alpha=0.75)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tight_layout()
plt.show()


#===============================================================
# MEAN-FIELD HAMILTONIAN WITH MASS TERM
#===============================================================

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

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

#-----------------------------------
# COMPUTATION
#-----------------------------------

x_xi = y_xi = np.linspace((1-side)*np.pi/2, (1+side)*np.pi/2, n)
X_xi, Y_xi = np.meshgrid(x_xi, y_xi)

E_xi = clean_bands(E_xi_k(X_xi, Y_xi))

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Lifting the degeneracy in the high-energy subspace with a small mass term
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_title('Degeneracy Point in the High-Energy Subspace of the\nMean-Field '
             'Hamiltonian ($\phi =$' + str(phi) + ') with a Small Mass Term')
for i in range(2):
    ax.plot_surface(X_xi/np.pi-0.5,Y_xi/np.pi-0.5,E_xi[i], alpha=0.75,rstride=5,cstride=5,linewidth=0)
ax.set_xlabel('$0.5 + a k_x/\pi$'), ax.set_ylabel('$0.5 + a k_y/\pi$'), ax.set_zlabel('$E/t_{pd}$')
ax.set_xlim(-side/2,side/2), ax.set_ylim(-side/2,side/2)
ax.set_xticks(np.linspace(-side/2, side/2, 5))
ax.set_yticks(np.linspace(-side/2, side/2, 5))
ax.ticklabel_format(style='sci', scilimits=(0,0))
ax.view_init(1, 45)
plt.tight_layout()
plt.show()


#===============================================================
# BERRY CURVATURE
#===============================================================

#-----------------------------------
# OPTIMIZING COMPUTATION
#-----------------------------------

# Gradient of the mean-field Hamiltonian as optimized JIT functions
Dx_H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y).diff(k_x), 'numpy'))
Dy_H_MF_k = jit(sym.lambdify([k_x, k_y], H_MF_sym(k_x, k_y).diff(k_y), 'numpy'))

# Evaluation of the z-component of the Berry curvature as a vectorized optimized JIT function
@njit
def Bz_k(k_x, k_y):
    (E_xi, n_xi) = LA.eigh(H_xi_k(k_x, k_y))
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

#-----------------------------------
# COMPUTATION
#-----------------------------------

zoom = side*np.pi/2*np.linspace(-1, 1, n)
x_L, y_L = point_L[0]*np.pi/2+zoom, point_L[1]*np.pi/2+zoom
X_L, Y_L = np.meshgrid(x_L, y_L)
x_R, y_R = point_R[0]*np.pi/2+zoom, point_R[1]*np.pi/2+zoom
X_R, Y_R = np.meshgrid(x_R, y_R)

Bz_L = clean_bands(Bz_k(X_L, Y_L))
Bz_R = clean_bands(Bz_k(X_R, Y_R))

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Z-component of the Berry curvature of the lowest band of the high-energy subspace around pi/2*(-1,1)
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('Z-Component of the Berry Curvature of the\nLowest Band of the '
             'High-Energy Subspace of the\nMean-Field '
             'Hamiltonian ($\phi =$' + str(phi) + ') with a Small Mass Term\n')
c = ax.pcolormesh(Bz_L[1], cmap ='coolwarm')
ax.set_xticks(np.linspace(0, n, 5)), ax.set_yticks(np.linspace(0, n, 5))
xTick = yTick = ["%.1e" % x for x in side/2*np.linspace(-1, 1, 5)]
xTick[2] = yTick[2] = 0.0
ax.set_xticklabels(xTick), ax.set_yticklabels(yTick)
ax.set_xlabel(r'$-0.5+ak_x/\pi$'), ax.set_ylabel(r'$0.5+ak_y/\pi$')
cMax = np.max(np.abs(c.get_clim()))
c.set_clim(-cMax, cMax)
fig.colorbar(c)
plt.tight_layout()
plt.show()

# Z-component of the Berry curvature of the lowest band of the high-energy subspace around pi/2*(1,1)
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('Z-Component of the Berry Curvature of the\nLowest Band of the '
             'High-Energy Subspace of the\nMean-Field '
             'Hamiltonian ($\phi =$' + str(phi) + ') with a Small Mass Term\n')
c = ax.pcolormesh(Bz_R[1], cmap ='coolwarm')
ax.set_xticks(np.linspace(0,n,5)), ax.set_yticks(np.linspace(0,n,5))
xTick = yTick = ["%.1e" % x for x in side/2*np.linspace(-1, 1, 5)]
xTick[2] = yTick[2] = 0.0
ax.set_xticklabels(xTick), ax.set_yticklabels(yTick)
ax.set_xlabel(r'$0.5+ak_x/\pi$'), ax.set_ylabel(r'$0.5+ak_y/\pi$')
cMax = np.max(np.abs(c.get_clim()))
c.set_clim(-cMax, cMax)
fig.colorbar(c)
plt.tight_layout()
plt.show()


#===============================================================
# BERRY CURVATURE
#===============================================================

#-----------------------------------
# COMPUTATION
#-----------------------------------

x_B, y_B = side_B[0]*np.pi/2*np.linspace(-1, 1, n_B), side_B[1]*np.pi/2*np.linspace(-1, 1, n_B)
X_B_ini, Y_B_ini = np.meshgrid(x_B, y_B)
X_B = point_B[0]*np.pi/2 + (X_B_ini - Y_B_ini)/np.sqrt(2)
Y_B = point_B[1]*np.pi/2 + (X_B_ini + Y_B_ini)/np.sqrt(2)

Bz = clean_bands(Bz_k(X_B, Y_B))
sg = np.sign(Bz_k(point_B[0]*np.pi/2,point_B[1]*np.pi/2)[1])

#-----------------------------------
# VISUALIZATION
#-----------------------------------

# Z-component of the Berry curvature of the lowest band of the high-energy subspace around pi/2*(1,1)
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('Z-Component of the Berry Curvature of the\nLowest Band of the '
             'High-Energy Subspace of the\nMean-Field '
             'Hamiltonian ($\phi =$' + str(phi) + ') with a Small Mass Term\n'
             'After Translating, Rotating, and Scaling\n')
c = ax.pcolormesh(Bz[1], cmap='hot' if sg == 1 else 'hot_r')
ax.set_xticks(np.linspace(0,n_B,5)), ax.set_yticks(np.linspace(0,n_B,5))
xTick = ["%.1e" % x for x in side_B[0]*np.linspace(-1, 1, 5)]
yTick = ["%.1e" % x for x in side_B[1]*np.linspace(-1, 1, 5)]
xTick[2] = yTick[2] = 0.0
ax.set_xticklabels(xTick), ax.set_yticklabels(yTick)
ax.set_xlabel(r'$ak_x/\pi$'), ax.set_ylabel(r'$ak_y/\pi$')
cMaxAbs = np.max(np.abs(c.get_clim()))
c.set_clim(0, cMaxAbs)
fig.colorbar(c)
plt.tight_layout()
plt.show()

# Zoom on the z-component of the Berry of the lowest band of the high-energy subspace curvature around pi/2*(1,1)
fig = plt.figure(figsize=(5,5))
ax = plt.axes()
ax.set_title('Z-Component of the Berry Curvature of the\nLowest Band of the '
             'High-Energy Subspace of the\nMean-Field '
             'Hamiltonian ($\phi =$' + str(phi) + ') with a Small Mass Term\n'
             'After Translating, Rotating, and Scaling\n')
c = ax.pcolormesh(Bz[1], cmap='hot' if sg == 1 else 'hot_r')
ax.set_xlim(n_B/2*(1-1/mag), n_B/2*(1+1/mag)), ax.set_ylim(n_B/2*(1-1/mag), n_B/2*(1+1/mag))
ax.set_xticks(np.linspace(n_B/2*(1-1/mag), n_B/2*(1+1/mag), 5))
ax.set_yticks(np.linspace(n_B/2*(1-1/mag), n_B/2*(1+1/mag), 5))
xTick = ["%.2e" % x for x in side_B[0]/(2*mag)*np.linspace(-1, 1, 5)]
yTick = ["%.2e" % x for x in side_B[1]/(2*mag)*np.linspace(-1, 1, 5)]
xTick[2] = yTick[2] = 0.0
ax.set_xticklabels(xTick), ax.set_yticklabels(yTick)
ax.set_xlabel(r'$ak_x/\pi$'), ax.set_ylabel(r'$ak_y/\pi$')
cMaxAbs = np.max(np.abs(c.get_clim()))
c.set_clim(0, cMaxAbs)
fig.colorbar(c)
plt.tight_layout()
plt.show()


#===============================================================
# BERRY PHASE
#===============================================================

#-----------------------------------
# COMPUTATION
#-----------------------------------

# Evaluation of the Berry phase of the lowest upper energy band around point_B*(pi/2, pi/2)
# for a counterclockwise orbit in percentage of pi
dA = side_B[0]*side_B[1]*(np.pi/(n_B-1))**2
berry_phase = [100*sum(Bz[1].flatten())*dA/np.pi, 100/np.sqrt(((side_B[0]/2)/xi)**2+1)/np.pi]

# Rounding of the Berry phase in terms of the uncertainty
dec = 0
while (round(np.trunc(berry_phase[1]*10**dec)) == 0):
    dec += 1
berry_phase = [round(berry_phase[0], dec), round(berry_phase[1], dec)]

#-----------------------------------
# PRINT
#-----------------------------------

# Berry phase in percentage of pi for the lowest upper energy band around (pi/2a, pi/2a)
print(berry_phase)

