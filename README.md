# Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates.

By Etienne Bolduc, Department of Physics, McGill University, Montreal, Canada. (August 2019)  
A thesis submitted to McGill University in partial fulfillment of the requirements of the degree of Master of Science.

## Abstract

High-temperature superconductivity in the cuprates has been at the heart of many debates since its discovery more than 30 years ago. No consensus has been reached yet about the underlying physics, but plausible descriptions usually fall into two categories each carrying various propositions. The quantum oscillations data acquired over the past few years for the normal state of the cuprates under a strong magnetic field has recently been used to obtain the electronic Berry phase of different compounds, which manifests through the phase mismatch in quantum oscillations. This analysis revealed an electronic Berry phase of $0 \pmod{2\pi}$ in three hole-doped compounds and $1.4\pi \pmod{2\pi}$ in one electron-doped compound. To investigate the mysterious pseudogap phase of the cuprates, the theoretical candidate known as the circulating current state of Varma as approached by Bulut is analyzed to numerically evaluate through a semiclassical approach the electronic Berry phase in this normal state. Under a typical parameter set in line with experimental data, a phase of $\pi$ is found. A comparison of the semiclassical approach with the Peierls substitution applied to this model confirms this result and further leads to an uncertainty on the phase of order $0.01\pi$. Hence, the circulating current state is incompatible with quantum oscillation data according to the Berry phase.

## Numerical Analysis

The present section is a description of the numerical analysis behind this thesis. While numeric computing—including plotting and graphing—was initially performed on MATLAB, it was converted to Python for portability and shareability. The images used here come from the figures generated in Python. For more information on the theoretical background of this work and to see all derivations in detail, please consult the [full thesis](bolduc-2019-determination-of-the-berry-phase-in-the-staggered-loop-current-model-of-the-pseudogap-in-the-cuprates.pdf).

### 1　Semiclassical approach

Bla

#### 1.1　Mean-Field Hamiltonian

```math
\hat{H}_{MF} = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger \textbf{H}_{MF}(\textbf{k}) \overline{\Psi}_\textbf{k}
```

```math
\textbf{H}_{MF}(\textbf{k}) =
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & \textbf{H}_{1}(\textbf{k}) \\
\textbf{H}_{MF}^\dagger(\textbf{k}) & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}
```

#### 1.2　Fermi Surface

Bla

#### 1.3　Mean-Field Hamiltonian with Mass Term

Bla

#### 1.4　Berry Curvature

Bla

#### 1.5　Berry Phase

Bla

### 2　Peierls Substitution
