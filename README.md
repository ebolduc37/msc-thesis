# Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates.

By Etienne Bolduc, Department of Physics, McGill University, Montreal, Canada. (August 2019)  
A thesis submitted to McGill University in partial fulfillment of the requirements of the degree of Master of Science.

## Abstract

High-temperature superconductivity in the cuprates has been at the heart of many debates since its discovery more than 30 years ago. No consensus has been reached yet about the underlying physics, but plausible descriptions usually fall into two categories each carrying various propositions. The quantum oscillations data acquired over the past few years for the normal state of the cuprates under a strong magnetic field has recently been used to obtain the electronic Berry phase of different compounds, which manifests through the phase mismatch in quantum oscillations. This analysis revealed an electronic Berry phase of $0 \pmod{2\pi}$ in three hole-doped compounds and $1.4\pi \pmod{2\pi}$ in one electron-doped compound. To investigate the mysterious pseudogap phase of the cuprates, the theoretical candidate known as the circulating current state of Varma as approached by Bulut is analyzed to numerically evaluate through a semiclassical approach the electronic Berry phase in this normal state. Under a typical parameter set in line with experimental data, a phase of $\pi$ is found. A comparison of the semiclassical approach with the Peierls substitution applied to this model confirms this result and further leads to an uncertainty on the phase of order $0.01\pi$. Hence, the circulating current state is incompatible with quantum oscillation data according to the Berry phase.

## Numerical Analysis

The present section is a description of the numerical analysis behind this thesis. While numeric computing—including plotting and graphing—was initially performed in MATLAB, it was converted to Python for portability and shareability. The plots below were generated in Python. For more information on the theoretical background of this work and to find all detailed derivations, consult the [full thesis](bolduc-2019-determination-of-the-berry-phase-in-the-staggered-loop-current-model-of-the-pseudogap-in-the-cuprates.pdf).

### 1　Semiclassical approach

In an effort to describe the pseudogap phase of cuprates, Varma suggested in 1997 a competing order model, a three-band model with the particularity of having current circulating in each unit cell as in Figure 4.2, leading to this phase being referred to as the circulating current phase [50]. It was argued then and shown later that the properties of this phase are similar to those of the pseudogap phase [51].

More recently, Bulut _et al._ have investigated a phase with a staggered pattern of intra-unit cell loop currents (LCs) called $\pi\textrm{LC}$ phase [7]. It features the ordering wave vector $\textbf{Q} = \tfrac{1}{a} (\pi,\pi)$ where $a$ is the lattice spacing, one that is relevant to cuprates [29]. As demonstrated in Figure 4.1, it plays an essential role in the Fermi surface reconstruction suggested to be behind the small Fermi surface of the pseudogap phase and the hole and electron pockets observed in experiments [8]. This section shows that a d-wave-symmetric gap will be maintained in the energy spectrum of the $\pi\textrm{LC}$ state, in agreement with the pseudogap phase [29]. Additionally, the $\pi\textrm{LC}$ state breaks time-reversal symmetry and could explain the Kerr effect observed in the pseudogap phase through experiments [7].

The $\pi\textrm{LC}$ Hamiltonian is written on a square lattice and each site corresponds to a unit cell, i.e., a $\textrm{CuO}_ 2$ plane containing a copper $d_{x^2-y^2}$ orbital and oxygen $p_x$ and $p_y$ orbitals, denoted by $\textrm{Cu}d_{x^2-y^2}$, $\textrm{O}p_x$, and $\textrm{O}p_y$. The relevant bonds are the nearest neighbor $p–d$ and $p–p$ bonds. As discussed before, these bonds exhibit intra-unit cell loop currents, equivalent to directional hopping. Additionally, the current must switch direction between unit cells like in Figure 4.2 to obtain the Fermi surface reconstruction found in Figure 4.1. Any state under such considerations breaks both time-reversal and lattice-translation symmetries. The specific staggered pattern of intertwined LCs studied by Bulut _et al._ [7] is shown in Figure 4.2. Note that this state has 4-fold rotational symmetry and conserves current. The $\pi\textrm{LC}$ model analyzed below is the same as the one explored by Bulut _et al._ [7], but an alternate current pattern is also investigated—ultimately leading to the same Berry phase.

The $\pi\textrm{LC}$ Hamiltonian denoted by $\hat{H}$ can be broken down into two parts: the kinetic energy $\hat{H}_0$ and the charge order $\hat{H}^\prime$. Each one can be solved separately. However, a mean-field approach is needed to solve the charge order, which will lead to the final diagonalized Hamiltonian being a mean-field approximation. Since we are looking for solutions that do not break spin-rotational symmetries, the spin labels are omitted going forward.

#### 1.1　Mean-Field Hamiltonian

The choice of the unit cell for a single $\textrm{CuO}_ 2$ plane and of orbital phase convention can be found in Figure 4.3. The inequivalent bonds are numbered to distinguish them from one another. We assume a total of $N^2$ unit cells with periodic boundary conditions and label the orbitals as $d$ for $\textrm{Cu}d_{x^2-y^2}$, $x$ for $\textrm{O}p_x$, and $y$ for $\textrm{O}p_y$.

##### 1.1.1　Kinetic Energy

The kinetic energy in momentum space can be obtained from its position-space expression through a Fourier transformation. Let $\hat{c}_ {\textbf{k}\alpha}$ / $\hat{c}_ {\textbf{k}\alpha}^\dagger$ / $\hat{n}_ {\textbf{k}\alpha}$ be the annihilation/creation/number operator for an electron in orbital $\alpha$ with crystal momentum $\hbar \textbf{k}$ with $\textbf{k} = ( k_x, k_y ) \in \tfrac{2\pi}{aN} \mathbb{Z}_ N^2$. By periodicity, the Brillouin zone $BZ$ is the set $\tfrac{1}{a} \[−\pi, \pi\] \times \tfrac{1}{a} \[−\pi, \pi\]$. Introducing the ordering wave vector $\textbf{Q} = \tfrac{1}{a} (\pi,\pi)$ for the Fermi surface reconstruction, we obtain after some calculations that

```math
\hat{H}_{0} = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & \textbf{0} \\
\textbf{0} & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}
\overline{\Psi}_\textbf{k}
```

where $BZ^\prime$ is the reduced Brillouin zone pictured in Figure 4.4 consisting of the closed ball of radius $\tfrac{1}{a} \pi$ with p-norm $L_1$ centered at the origin. We defined $\overline{\Psi}_ \textbf{k}^\dagger \equiv \[ \Psi_ \textbf{k}^\dagger \Psi_{\textbf{k} + \textbf{Q}}^\dagger \]$ with $\Psi_\textbf{k}^\dagger \equiv \[ \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \]$ and applied a gauge transformation such that $\hat{c}_ {\textbf{k} x} \rightarrow i\hat{c}_ {\textbf{k} x}$ and $\hat{c}_ {\textbf{k} y} \rightarrow i\hat{c}_ {\textbf{k} y}$. By letting $s_x \equiv \sin{(\tfrac{a}{2} k_x)}$ and $s_y \equiv \sin{(\tfrac{a}{2} k_y)}$, we have that

```math
\begin{align}
& \textbf{H}_{0}(\textbf{k}) \equiv \\
& \begin{bmatrix}
\epsilon_d & 2 t_{pd} s_x & -2 t_{pd} s_y \\
2 t_{pd} s_x & \epsilon_p & 4 t_{pp} s_x s_y \\
-2 t_{pd} s_y & 4 t_{pp} s_x s_y & \epsilon_p
\end{bmatrix}.
\end{align}
```

Under a typical parameter set, $t_{pd} = 1$ to define the unit energy, $t_{pp} = −0.5$ and $\epsilon_d − \epsilon_p = 2.5$, in accordance with experimental data [7, 19]. The energy gap between the highest energy band of $\textbf{H}_{0}(\textbf{k})$ and the two other bands can be observed from Figure ??. Additionally, the property that $E_n (\textbf{k} + \textbf{Q}) = E_n (\textbf{k})$ if and only if $\textbf{k} \in \partial BZ^\prime$ can be observed by inspecting the points where $| E_n (\textbf{k}) - E_n (\textbf{k} + \textbf{Q}) | < 0.01$ as seen in Figure ??.

![H_0](https://github.com/ebolduc37/msc-thesis/assets/44382376/117cb73c-5d12-43aa-ae1d-01d2ab6c5566)
![diff_E_n](https://github.com/ebolduc37/msc-thesis/assets/44382376/805d5a02-5b1e-4fc0-a4bc-66390c9cbbb5)

##### 1.1.2　Mean-Field Decomposition of the Charge Order

From the charge order $\hat{H}^\prime$ taking into consideration the intraorbital and interorbital interactions in position space, it is possible to obtain the mean-field version $\hat{H}^\prime_{MF}$ in terms of circulating currents in momentum-space. On the one hand, the intraorbital interactions lead to Hartree shifts to the orbital energies which merely renormalize in $\epsilon_d$ and $\epsilon_p$. On the other hand, the interorbital interactions can be decomposed in terms of the circulating currents where the sign depends on the direction of the current flow. After some calculations, the mean-field decomposition of the charge order can be expressed as

```math
\hat{H}_{MF}^\prime = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger
\begin{bmatrix}
\widetilde{\boldsymbol{\epsilon}} & \textbf{H}_{1}(\textbf{k}) \\
\textbf{H}_{1}^\dagger(\textbf{k}) & \widetilde{\boldsymbol{\epsilon}}
\end{bmatrix}
\overline{\Psi}_\textbf{k}
```

where an explicit expression for $\textbf{H}_ {1}(\textbf{k})$ depends on the current pattern. Note that $\widetilde{\boldsymbol{\epsilon}}$ simply represents a shift in the orbital energies equivalent to $V_ {pd} + 2V_ {pp} \equiv \widetilde{\epsilon}_ d$ in $\textrm{Cu}d_{x^2-y^2}$ and $2V_{pd} \equiv \widetilde{\epsilon}_ p$ in $\textrm{O}p_x$ and $\textrm{O}p_y$, where $V_{\beta \alpha}$ is the interorbital Coulomb interaction energy.

The focus is made on _physical_ current patterns—meaning that the current is conserved on each orbital site—with 4-fold rotational symmetry. There are only two possible inequivalent physical current patterns with 4-fold rotational symmetry. The one investigated by Bulut _et al._ [7] is shown in Figure 4.2. The second current pattern can be obtained from the one in Figure 4.2 by inverting the current along $p–p$ bonds. Throughout the rest of this work, the two current patterns will be distinguished by $\phi \in \\{ \pm 1 \\}$ with the current pattern investigated by Bulut _et al._ [7] corresponding to $\phi = 1$. Explicitly, we have for these two current patterns under the gauge transformation from earlier that $\widetilde{\boldsymbol{\epsilon}}$ stays invariant while, if we let $R_{pd} \equiv V_{pd} z_{pd} / t_{pd}$, $R_{pp} \equiv V_{pp} z_{pp} / t_{pp}$, $c_x \equiv \cos{(\tfrac{a}{2} k_x)}$, and $c_y \equiv \cos{(\tfrac{a}{2} k_y)}$,

```math
\begin{align}
& \textbf{H}_{1}(\textbf{k}) = \\
& \begin{bmatrix}
0 & 2 i R_{pd} c_x & 2i R_{pd} c_y \\
-2i R_{pd} s_x & 0 & -4i \phi R_{pp} s_x c_y \\
-2i R_{pd} s_y & 4i \phi R_{pp} c_x s_y & 0
\end{bmatrix}.
\end{align}
```

Under a typical parameter set, $V_ {pd} = 2.2$, $V_ {pp} = 1$, $z_ {pd} = 0.04$, and $z_ {pp} = z_ {pd} / 3$, in accordance with experimental data [7, 19].

##### 1.1.3　Full Mean-Field Hamiltonian

Combining both (4.7) and (4.11) yields the effective mean-field $\pi\textrm{LC}$ Hamiltonian

```math
\hat{H}_{MF} = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger \textbf{H}_{MF}(\textbf{k}) \overline{\Psi}_\textbf{k}
```
where letting $\epsilon_ d \rightarrow \epsilon_ \alpha + \widetilde{\epsilon}_ \alpha \equiv \varepsilon_ \alpha$ in $\textbf{H}_ {0}(\textbf{k})$ and real matrix $\textbf{V}(\textbf{k}) \equiv i \lambda^{-1} \textbf{H}_ {1}(\textbf{k})$ for unitless parameter $\lambda = z_ {pd}/t_ {pd}$ results in

```math
\textbf{H}_{MF}(\textbf{k}) =
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & - i \lambda \textbf{V}(\textbf{k}) \\
i \lambda \textbf{V}^T (\textbf{k}) & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}.
```

Four points are of particular interest here: the points $\textbf{k}^ *$ such that $|k^ *_ x| = |k^ *_ y| = \tfrac{\pi}{2a}$, a set that we denote by $D$. These momenta correspond to the 2-fold degeneracy points of the high-energy subspace of $\textbf{H}_{MF}(\textbf{k})$ as seen in Figure ??.

![H_MF_bands_all](https://github.com/ebolduc37/msc-thesis/assets/44382376/6a7fc6f8-8d2e-4199-a2bf-50c97aa9d7f4)
![H_MF_bands_high](https://github.com/ebolduc37/msc-thesis/assets/44382376/57951b05-93c6-4da7-a747-0a3bf1bea5cd)

##### 1.1.4　Fermi Surface

We put our focus on the two highest energy bands which are half-filled and related to the energy of the $\textrm{Cu}d_{x^2-y^2}$ [49]; the other energy bands are irrelevant because they are restricted to energies well below the Fermi energy. To find the Fermi energy numerically, we list the energies from the high-energy subspace of $\textbf{H}_{MF}(\textbf{k})$ for all $N^2$ momenta, sort the resulting list of $2 N^2$ items, and return the $N^2$ item. The resulting Fermi surface is shown in Figure ?? with hole and electron pockets in orange and blue respectively.

![H_MF_bands_high_F](https://github.com/ebolduc37/msc-thesis/assets/44382376/d29dbfce-ed17-4bcc-9974-a275da116828)
![Fermi_pockets](https://github.com/ebolduc37/msc-thesis/assets/44382376/3497f477-5264-4094-bae2-5e944335f97f)

#### 1.2　Berry Phase

It is possible to evaluate numerically the Berry phase accumulated by an electron orbiting a hole pocket. The method is straightforward: first, the Hamiltonian in momentum space in its matrix form is diagonalized numerically on a discrete grid of points; then, the Berry curvature is calculated at all points; finally, the Berry phase is evaluated by integrating over the area of the hole pocket which is enclosed by the electron's orbit.

##### 1.2.1　Mass Term

Special considerations must be taken in the presence of a Dirac point because numerical methods do not adequately work at discontinuous points. In the case of a two-level system in 2D, a mass term must be added to the Hamiltonian in order to evaluate the Berry phase. Given a small, finite value, the mass term provides a way to approximate the Dirac delta function located at the Dirac points in the Berry curvature. In such a way, numerical methods can approximate the Berry phase without any problem. In order to make this precise, the mass term should be small enough to make the delta function's weight negligible outside the area of integration whereas the grid resolution must be taken high enough to approximate around the peak accurately.

To approach this problem, we must rely on perturbation theory. Considering that the matrix elements of $\textbf{V}(\textbf{k})$ are of the order of magnitude of 1 or lower for any $\textbf{k}$ and that $\lambda \ll 1$ under a typical parameter set, we take the unperturbed Hamiltonian to be

```math
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & \textbf{0} \\
\textbf{0} & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}.
```

Let the energy eigenvalues and corresponding eigenstates off $\textbf{H}_ {0}(\textbf{k})$ be denoted by $\textbf{E}_ {n}(\textbf{k})$ and $\ket{n(\textbf{k})}$ for $n \in \\{ \pm, 0 \\}$ such that $E_+(\textbf{k}) \geq E_ {0, -}(\textbf{k})$. Hence, the unperturbed energy eigenvalues are $\textbf{E}_ {n}(\textbf{k})$ and $\textbf{E}_ {n}(\textbf{k}+\textbf{Q})$ with respective corresponding eigenstates

```math
\begin{gather}
\ket{n_\uparrow (\textbf{k})} =
\begin{bmatrix}
\ket{n(\textbf{k})} \\
\textbf{0}
\end{bmatrix},
\\
\ket{n_\downarrow (\textbf{k})} =
\begin{bmatrix}
\textbf{0} \\
\ket{n(\textbf{k}+\textbf{Q})}
\end{bmatrix}.
\end{gather}
```

The two high-energy eigenstates $\ket{n_ \uparrow (\textbf{k})}$ and $\ket{n_ \downarrow (\textbf{k})}$ form a subspace that is separated enough energetically from the rest of the Hilbert space to apply perturbation theory and effectively project the mean-field Hamiltonian onto this subspace. Thus, the projected Hamiltonian at any point $\textbf{k}$ can be expressed as

```math
\textbf{H}_{U}(\textbf{k}) =
\bar{E}(\textbf{k})\textbf{I}
+ \lambda \Delta(\textbf{k}) \boldsymbol{\sigma}_2
+ \varepsilon(\textbf{k}) \boldsymbol{\sigma}_3
```

with $\boldsymbol{\sigma}_ i$ the Pauli matrices and where $\bar{E}(\textbf{k}) \equiv \tfrac{1}{2} \[ \textbf{E}_ {+}(\textbf{k}) + \textbf{E}_ {+}(\textbf{k}+\textbf{Q}) \]$, $\Delta(\textbf{k}) \equiv \bra{+(\textbf{k})} \textbf{V}(\textbf{k}) \ket{+(\textbf{k}+\textbf{Q})}$, and $\varepsilon(\textbf{k}) \equiv \tfrac{1}{2} \[ \textbf{E}_ {+}(\textbf{k}) - \textbf{E}_ {+}(\textbf{k}+\textbf{Q}) \]$. The high-energy subspace of the mean-field Hamiltonian and of the projected Hamiltonian share the exact same degeneracy points from the set $D$.

We introduce the mass term $\xi > 0$ to first order in perturbation theory by taking $\textbf{H}_ {U}(\textbf{k}) \rightarrow \textbf{H}_ {U}(\textbf{k}) + \alpha \xi \boldsymbol{\sigma}_ 1$ for the unitary dimensionful constant $\alpha$ carrying units of energy times length. For the mean-field Hamiltonian, it translates to

```math
\textbf{H}_{MF}(\textbf{k}) \rightarrow
\textbf{H}_{MF}(\textbf{k})
+ \alpha \xi \begin{bmatrix}
\textbf{0} & \textbf{M}(\textbf{k}) \\
\textbf{M}^\dagger (\textbf{k}) & \textbf{0}
\end{bmatrix}
```

for $\textbf{M}(\textbf{k}) \equiv \ket{+(\textbf{k})} \bra{+(\textbf{k}+\textbf{Q})}$.

##### 1.2.2　Numerical Evaluation

Given a Hamiltonian $H(\textbf{k})$, the Berry curvature $\textbf{B}_ {n}(\textbf{k})$ can be calculated at any point $\textbf{k}$ as

```math
\textbf{B}_{n}(\textbf{k}) =
i \sum_{m \neq n} \frac{ \langle \nabla H (\textbf{k}) \rangle_{nm} \times \langle \nabla H (\textbf{k})\rangle_{mn}}
{[ E_n(\textbf{k}) - E_m(\textbf{k}) ]^2}
```

where $\langle \nabla H (\textbf{k})\rangle_ {mn} = \bra{m(\textbf{k})} \nabla H (\textbf{k}) \ket{n(\textbf{k})}$ for $\\{ \ket{n (\textbf{k})} \\}$ the (orthonormal) set of energy eigenstates of $H(\textbf{k})$.

### 2　Peierls Substitution
