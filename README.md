# Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates.

By Etienne Bolduc, Department of Physics, McGill University, Montreal, Canada (August 2019).  
A thesis submitted to McGill University in partial fulfillment of the requirements of the degree of Master of Science.

## Abstract

High-temperature superconductivity in the cuprates has been at the heart of many debates since its discovery more than 30 years ago. No consensus has been reached yet about the underlying physics, but plausible descriptions usually fall into two categories each carrying various propositions. The quantum oscillations data acquired over the past few years for the normal state of the cuprates under a strong magnetic field has recently been used to obtain the electronic Berry phase of different compounds, which manifests through the phase mismatch in quantum oscillations. This analysis revealed an electronic Berry phase of $0 \pmod{2\pi}$ in three hole-doped compounds and $1.4\pi \pmod{2\pi}$ in one electron-doped compound. To investigate the mysterious pseudogap phase of the cuprates, the theoretical candidate known as the circulating current state of Varma as approached by Bulut is analyzed to numerically evaluate through a semiclassical approach the electronic Berry phase in this normal state. Under a typical parameter set in line with experimental data, a phase of $\pi$ is found. A comparison of the semiclassical approach with the Peierls substitution applied to this model confirms this result and further leads to an uncertainty on the phase of order $0.01\pi$. Hence, the circulating current state is incompatible with quantum oscillation data according to the Berry phase.

## Numerical Analysis

The present section is a short summary of the numerical analysis behind this thesis. While the numerical computing—including plotting and graphing—was performed in MATLAB initially, it was converted to Python for portability and shareability after the thesis was submitted. For the sake of consistency, the plots found below are generated in Python. For more information, consult the [full thesis](bolduc-2019-determination-of-the-berry-phase-in-the-staggered-loop-current-model-of-the-pseudogap-in-the-cuprates.pdf).

### 1　Semiclassical approach

In an effort to describe the pseudogap phase of cuprates, Varma suggested in 1997 a competing order model, a three-band model with the particularity of having current circulating in each unit cell as in the figure below, leading to this phase being referred to as the circulating current phase [50]. It was argued then and shown later that the properties of this phase are similar to those of the pseudogap phase [51].

![Color-online-Staggered-pattern-of-spontaneous-loop-currents-Open-circle-x-and-y](https://github.com/ebolduc37/msc-thesis/assets/44382376/5d20d3f6-394a-4f8d-bcdd-83936f4a8cea)

More recently, Bulut _et al._ have investigated a phase with a staggered pattern of intra-unit cell loop currents (LCs) called $\pi\textrm{LC}$ phase [7]. It features the ordering wave vector $\boldsymbol{Q} = \tfrac{1}{a} (\pi,\pi)$ where $a$ is the lattice spacing, one that is relevant to cuprates [29], which plays an essential role in the Fermi surface reconstruction suggested to be behind the small Fermi surface of the pseudogap phase and the hole and electron pockets observed in experiments [8].

The $\pi\textrm{LC}$ Hamiltonian is written on a square lattice and each site corresponds to a unit cell, i.e., a $\textrm{CuO}_ 2$ plane containing a copper $d_{x^2-y^2}$ orbital and oxygen $p_x$ and $p_y$ orbitals, denoted by $\textrm{Cu}d_{x^2-y^2}$, $\textrm{O}p_x$, and $\textrm{O}p_y$. The relevant bonds are the nearest neighbor $p–d$ and $p–p$ bonds. As discussed above, these bonds exhibit intra-unit cell loop currents, equivalent to directional hopping. Additionally, the direction of the current must be inverted between unit cells to obtain the right Fermi surface reconstruction. Any state under such considerations breaks both time-reversal and lattice-translation symmetries. The specific staggered pattern of intertwined LCs studied by Bulut _et al._ [7] is shown in the figure above. Note that this state has 4-fold rotational symmetry and conserves current. The $\pi\textrm{LC}$ model analyzed thereafter is the same as the one explored by Bulut _et al._ [7], but an alternate current pattern is also investigated—ultimately leading to the same Berry phase.

The $\pi\textrm{LC}$ Hamiltonian denoted by $\hat{H}$ can be broken down into two parts: the kinetic energy $\hat{H}_0$ and the charge order $\hat{H}^\prime$. Each one can be solved separately. However, a mean-field approach is needed to solve the charge order, which will lead to the final diagonalized Hamiltonian being a mean-field approximation. Since we are looking for solutions that do not break spin-rotational symmetries, the spin labels are omitted going forward.

#### 1.1　Mean-Field Hamiltonian

The choice of the unit cell for a single $\textrm{CuO}_ 2$ plane and of orbital phase convention can be found in Figure 4.3. The inequivalent bonds are numbered to distinguish them from one another. We assume a total of $N^2$ unit cells with periodic boundary conditions and label the orbitals as $d$ for $\textrm{Cu}d_{x^2-y^2}$, $x$ for $\textrm{O}p_x$, and $y$ for $\textrm{O}p_y$.

##### 1.1.1　Kinetic Energy

The kinetic energy in momentum space can be obtained from its position-space expression through a Fourier transformation. Let $\hat{c}_ {\boldsymbol{k}\alpha}$ / $\hat{c}_ {\boldsymbol{k}\alpha}^\dagger$ be the annihilation/creation operator for an electron in orbital $\alpha$ with crystal momentum $\hbar \boldsymbol{k}$ with $\boldsymbol{k} = ( k_x, k_y ) \in \tfrac{2\pi}{aN} \mathbb{Z}_ N^2$. By periodicity, the Brillouin zone $BZ$ is the set $\tfrac{1}{a} \[−\pi, \pi\] \times \tfrac{1}{a} \[−\pi, \pi\]$. Introducing the ordering wave vector $\boldsymbol{Q} = \tfrac{1}{a} (\pi,\pi)$ for the Fermi surface reconstruction, we have that
```math
\hat{H}_{0} = \sum_{\boldsymbol{k} \in BZ'} \overline{\Psi}_\boldsymbol{k}^\dagger
\begin{bmatrix}
\boldsymbol{H}_{0}(\boldsymbol{k}) & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{H}_{0}(\boldsymbol{k} + \boldsymbol{Q})
\end{bmatrix}
\overline{\Psi}_\boldsymbol{k}
```
where $BZ^\prime$ is the reduced Brillouin zone pictured in Figure 4.4 consisting of the closed ball of radius $\tfrac{1}{a} \pi$ with p-norm $L_1$ centered at the origin. We defined $\overline{\Psi}_ \boldsymbol{k}^\dagger \equiv \[ \Psi_ \boldsymbol{k}^\dagger \Psi_{\boldsymbol{k} + \boldsymbol{Q}}^\dagger \]$ with $\Psi_\boldsymbol{k}^\dagger \equiv \[ \hat{c}_ {\boldsymbol{k}\alpha\sigma}^\dagger \hat{c}_ {\boldsymbol{k}\alpha\sigma}^\dagger \hat{c}_ {\boldsymbol{k}\alpha\sigma}^\dagger \]$ and applied a gauge transformation such that $\hat{c}_ {\boldsymbol{k} x} \rightarrow i\hat{c}_ {\boldsymbol{k} x}$ and $\hat{c}_ {\boldsymbol{k} y} \rightarrow i\hat{c}_ {\boldsymbol{k} y}$. By letting $s_x \equiv \sin{(\tfrac{a}{2} k_x)}$ and $s_y \equiv \sin{(\tfrac{a}{2} k_y)}$, we have that
```math
\begin{align}
& \boldsymbol{H}_{0}(\boldsymbol{k}) \equiv \\
& \begin{bmatrix}
\epsilon_d & 2 t_{pd} s_x & -2 t_{pd} s_y \\
2 t_{pd} s_x & \epsilon_p & 4 t_{pp} s_x s_y \\
-2 t_{pd} s_y & 4 t_{pp} s_x s_y & \epsilon_p
\end{bmatrix}.
\end{align}
```

Under a typical parameter set, $t_{pd} = 1$ to define the unit energy, $t_{pp} = −0.5$ and $\epsilon_d − \epsilon_p = 2.5$, in accordance with experimental data [7, 19]. The energy gap between the highest energy band of $\boldsymbol{H}_{0}(\boldsymbol{k})$ and the two other bands can be observed in the first figure below. Additionally, the property that $E_n (\boldsymbol{k} + \boldsymbol{Q}) = E_n (\boldsymbol{k})$ if and only if $\boldsymbol{k} \in \partial BZ^\prime$ can be observed by inspecting the points where $| E_n (\boldsymbol{k}) - E_n (\boldsymbol{k} + \boldsymbol{Q}) | < 0.01$ as seen in the second figure below.

![01_bands-kinetic](https://github.com/ebolduc37/msc-thesis/assets/44382376/c96208a9-4ff9-4621-8813-d0aa49c8f931)
![02_bands-diff](https://github.com/ebolduc37/msc-thesis/assets/44382376/8ee27d77-f888-4957-9c7a-93af0379a11f)


##### 1.1.2　Mean-Field Decomposition of the Charge Order

From the charge order $\hat{H}^\prime$ taking into consideration the intraorbital and interorbital interactions in position space, it is possible to obtain the mean-field version $\hat{H}^\prime_{MF}$ in terms of circulating currents in momentum-space. On the one hand, the intraorbital interactions lead to Hartree shifts to the orbital energies which merely renormalize in $\epsilon_d$ and $\epsilon_p$. On the other hand, the interorbital interactions can be decomposed in terms of the circulating currents where the sign depends on the direction of the current flow. The mean-field decomposition of the charge order is expressed as
```math
\hat{H}_{MF}^\prime = \sum_{\boldsymbol{k} \in BZ'} \overline{\Psi}_\boldsymbol{k}^\dagger
\begin{bmatrix}
\widetilde{\boldsymbol{\epsilon}} & \boldsymbol{H}_{1}(\boldsymbol{k}) \\
\boldsymbol{H}_{1}^\dagger(\boldsymbol{k}) & \widetilde{\boldsymbol{\epsilon}}
\end{bmatrix}
\overline{\Psi}_\boldsymbol{k}
```
where an explicit expression for $\boldsymbol{H}_ {1}(\boldsymbol{k})$ depends on the current pattern. Note that $\widetilde{\boldsymbol{\epsilon}}$ simply represents a shift in the orbital energies equivalent to $\widetilde{\epsilon}_ d \equiv V_ {pd} + 2V_ {pp}$ in $\textrm{Cu}d_{x^2-y^2}$ and $\widetilde{\epsilon}_ p \equiv 2V_{pd}$ in $\textrm{O}p_x$ and $\textrm{O}p_y$, where $V_{\beta \alpha}$ is the interorbital Coulomb interaction energy.

The focus is made on _physical_ current patterns—meaning that the current is conserved on each orbital site—with 4-fold rotational symmetry. There are only two possible inequivalent physical current patterns with 4-fold rotational symmetry. The one investigated by Bulut _et al._ [7] is shown at the beginning. The second current pattern can be obtained from the former by inverting the current along $p–p$ bonds. Throughout the rest of this work, the two current patterns will be denoted by $\phi \in \\{ \pm 1 \\}$, where the current pattern investigated by Bulut _et al._ [7] corresponds to $\phi = 1$ and the other to $\phi = -1$. Explicitly, we have for these two current patterns under the gauge transformation from earlier that $\widetilde{\boldsymbol{\epsilon}}$ stays invariant while, if we let $R_{pd} \equiv V_{pd} z_{pd} / t_{pd}$, $R_{pp} \equiv \phi V_{pp} z_{pp} / t_{pp}$, $c_x \equiv \cos{(\tfrac{a}{2} k_x)}$, and $c_y \equiv \cos{(\tfrac{a}{2} k_y)}$,
```math
\begin{align}
& \boldsymbol{H}_{1}(\boldsymbol{k}) \equiv \\
& \begin{bmatrix}
0 & 2 i R_{pd} c_x & 2i R_{pd} c_y \\
-2i R_{pd} s_x & 0 & -4i R_{pp} s_x c_y \\
-2i R_{pd} s_y & 4i R_{pp} c_x s_y & 0
\end{bmatrix}.
\end{align}
```

Under a typical parameter set, $V_ {pd} = 2.2$, $V_ {pp} = 1$, $z_ {pd} = 0.04$, and $z_ {pp} = z_ {pd} / 3$, in accordance with experimental data [7, 19].

##### 1.1.3　Full Mean-Field Hamiltonian

Combining both parts yields the effective mean-field $\pi\textrm{LC}$ Hamiltonian
```math
\hat{H}_{MF} = \sum_{\boldsymbol{k} \in BZ'} \overline{\Psi}_\boldsymbol{k}^\dagger \boldsymbol{H}_{MF}(\boldsymbol{k}) \overline{\Psi}_\boldsymbol{k}
```
where letting $\epsilon_ d \rightarrow \epsilon_ \alpha + \widetilde{\epsilon}_ \alpha \equiv \varepsilon_ \alpha$ in $\boldsymbol{H}_ {0}(\boldsymbol{k})$ and real matrix $\boldsymbol{V}(\boldsymbol{k}) \equiv i \lambda^{-1} \boldsymbol{H}_ {1}(\boldsymbol{k})$ for unitless parameter $\lambda = z_ {pd}/t_ {pd}$ results in
```math
\boldsymbol{H}_{MF}(\boldsymbol{k}) =
\begin{bmatrix}
\boldsymbol{H}_{0}(\boldsymbol{k}) & - i \lambda \boldsymbol{V}(\boldsymbol{k}) \\
i \lambda \boldsymbol{V}^T (\boldsymbol{k}) & \boldsymbol{H}_{0}(\boldsymbol{k} + \boldsymbol{Q})
\end{bmatrix}.
```

Four points are of particular interest here: the points $\boldsymbol{k}^ *$ such that $|k^ *_ x| = |k^ *_ y| = \tfrac{\pi}{2a}$, a set that we denote by $D$. These momenta correspond to the 2-fold degeneracy points of the high-energy subspace of $\boldsymbol{H}_{MF}(\boldsymbol{k})$ as seen in the second figure below for $\phi = 1$. Identical results are obtained when $\phi = -1$.

![03_bands-MF](https://github.com/ebolduc37/msc-thesis/assets/44382376/ed1beec7-799a-437e-949d-708124f19a4c)
![04_high-energy-bands-MF](https://github.com/ebolduc37/msc-thesis/assets/44382376/4691dd4e-5612-4bcb-9099-ff5efb04ab90)

##### 1.1.4　Fermi Surface

We put our focus on the two highest energy bands which are half-filled and related to the energy of the $\textrm{Cu}d_{x^2-y^2}$ [49]; the other energy bands are irrelevant because they are restricted to energies well below the Fermi energy. To find the Fermi energy numerically, we list the energies from the high-energy subspace of $\boldsymbol{H}_{MF}(\boldsymbol{k})$ for all $N^2$ momenta, sort the resulting list of $2 N^2$ items, and return the average of the $(N^2)^{th}$ and $(N^2+1)^{th}$ item. The resulting Fermi surface is shown in the second figure below with hole and electron pockets in orange and blue respectively.

![06_fermi-energy-Q1](https://github.com/ebolduc37/msc-thesis/assets/44382376/fc70a8e7-3fdd-4a14-89d1-6b6940bd2a23)
![07_fermi-pockets](https://github.com/ebolduc37/msc-thesis/assets/44382376/bbe0f307-008e-4d8d-8084-2fdb168707e9)

#### 1.2　Berry Phase

It is possible to evaluate numerically the Berry phase accumulated by an electron orbiting a hole pocket. The method is straightforward: first, the Hamiltonian in momentum space in its matrix form is diagonalized numerically on a discrete grid of points; then, the Berry curvature is calculated at all points; finally, the Berry phase is evaluated by integrating over the area of the hole pocket which is enclosed by the electron's orbit.

##### 1.2.1　Mass Term

Special considerations must be taken in the presence of a Dirac point because numerical methods do not adequately work at discontinuous points. In the case of a two-level system in 2D, a mass term must be added to the Hamiltonian in order to evaluate the Berry phase. Given a small, finite value, the mass term provides a way to approximate the Dirac delta function located at the Dirac points in the Berry curvature. In such a way, numerical methods can approximate the Berry phase without any problem. In order to make this precise, the mass term should be small enough to make the delta function's weight negligible outside the area of integration whereas the grid resolution must be taken high enough to approximate around the peak accurately.

To approach this problem, we must rely on perturbation theory. Considering that the matrix elements of $\boldsymbol{V}(\boldsymbol{k})$ are of the order of magnitude of 1 or lower for any $\boldsymbol{k}$ and that $\lambda \ll 1$ under a typical parameter set, we take the unperturbed Hamiltonian to be
```math
\begin{bmatrix}
\boldsymbol{H}_{0}(\boldsymbol{k}) & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{H}_{0}(\boldsymbol{k} + \boldsymbol{Q})
\end{bmatrix}.
```

Let the energy eigenvalues and corresponding eigenstates off $\boldsymbol{H}_ {0}(\boldsymbol{k})$ be denoted by $\boldsymbol{E}_ {n}(\boldsymbol{k})$ and $\ket{n(\boldsymbol{k})}$ for $n \in \\{ \pm, 0 \\}$ such that $E_+(\boldsymbol{k}) \geq E_ {0, -}(\boldsymbol{k})$. Hence, the unperturbed energy eigenvalues are $\boldsymbol{E}_ {n}(\boldsymbol{k})$ and $\boldsymbol{E}_ {n}(\boldsymbol{k}+\boldsymbol{Q})$ with respective corresponding eigenstates
```math
\begin{gather}
\ket{n_\uparrow (\boldsymbol{k})} =
\begin{bmatrix}
\ket{n(\boldsymbol{k})} \\
\boldsymbol{0}
\end{bmatrix},
\\
\ket{n_\downarrow (\boldsymbol{k})} =
\begin{bmatrix}
\boldsymbol{0} \\
\ket{n(\boldsymbol{k}+\boldsymbol{Q})}
\end{bmatrix}.
\end{gather}
```

The two high-energy eigenstates $\ket{n_ \uparrow (\boldsymbol{k})}$ and $\ket{n_ \downarrow (\boldsymbol{k})}$ form a subspace that is separated enough energetically from the rest of the Hilbert space to apply perturbation theory and effectively project the mean-field Hamiltonian onto this subspace. Thus, the projected Hamiltonian at any point $\boldsymbol{k}$ can be expressed as
```math
\boldsymbol{H}_{U}(\boldsymbol{k}) =
\bar{E}(\boldsymbol{k})\boldsymbol{I}
+ \lambda \Delta(\boldsymbol{k}) \boldsymbol{\sigma}_2
+ \varepsilon(\boldsymbol{k}) \boldsymbol{\sigma}_3
```
with $\boldsymbol{\sigma}_ i$ the Pauli matrices and where $\bar{E}(\boldsymbol{k}) \equiv \tfrac{1}{2} \[ \boldsymbol{E}_ {+}(\boldsymbol{k}) + \boldsymbol{E}_ {+}(\boldsymbol{k}+\boldsymbol{Q}) \]$, $\Delta(\boldsymbol{k}) \equiv \bra{+(\boldsymbol{k})} \boldsymbol{V}(\boldsymbol{k}) \ket{+(\boldsymbol{k}+\boldsymbol{Q})}$, and $\varepsilon(\boldsymbol{k}) \equiv \tfrac{1}{2} \[ \boldsymbol{E}_ {+}(\boldsymbol{k}) - \boldsymbol{E}_ {+}(\boldsymbol{k}+\boldsymbol{Q}) \]$. The high-energy subspace of the mean-field Hamiltonian and of the projected Hamiltonian share the same degeneracy points from the set $D$.

We introduce the mass term $\xi > 0$ to first order in perturbation theory by taking $\boldsymbol{H}_ {U}(\boldsymbol{k}) \rightarrow \boldsymbol{H}_ {U}(\boldsymbol{k}) + \alpha \xi \boldsymbol{\sigma}_ 1$ for the unitary dimensionful constant $\alpha$ carrying units of energy times length. For the mean-field Hamiltonian, this translates to $\boldsymbol{H}_ {MF}(\boldsymbol{k}) \rightarrow \boldsymbol{H}_ {MF}^\xi(\boldsymbol{k})$ for $\boldsymbol{H}_ {MF}^\xi(\boldsymbol{k}) \equiv \boldsymbol{H}_ {MF}(\boldsymbol{k}) + \alpha \xi \boldsymbol{M}(\boldsymbol{k})$ where $\boldsymbol{M}_ {11}(\boldsymbol{k}) = \boldsymbol{M}_ {22}(\boldsymbol{k}) = \boldsymbol{0}$ and $\boldsymbol{M}_ {12}(\boldsymbol{k}) = \boldsymbol{M}^\dagger_ {21}(\boldsymbol{k}) = \ket{+(\boldsymbol{k})} \bra{+(\boldsymbol{k}+\boldsymbol{Q})}$.

By introducing a mass term $\xi = 1 \cdot 10^{-5} \[a/\pi\]^{-1}$ as defined later on, the energy bands stay identical except near the degeneracy points where the degeneracy is lifted, as shown around the point $(\tfrac{\pi}{2a}, \tfrac{\pi}{2a})$ in the figure below.

![08_degeneracy-with-mass](https://github.com/ebolduc37/msc-thesis/assets/44382376/552ad311-5c74-499f-9bb6-0eaae96f09d2)

##### 1.2.2　Numerical Evaluation

Given a Hamiltonian $H(\boldsymbol{k})$, the Berry curvature $\boldsymbol{B}_ {n}(\boldsymbol{k})$ can be evaluated at any point $\boldsymbol{k}$ with the following equation:
```math
\boldsymbol{B}_{n}(\boldsymbol{k}) =
i \sum_{m \neq n} \frac{ \langle \nabla H (\boldsymbol{k}) \rangle_{nm} \times \langle \nabla H (\boldsymbol{k})\rangle_{mn}}
{[ E_n(\boldsymbol{k}) - E_m(\boldsymbol{k}) ]^2}
```
where $\langle \nabla H (\boldsymbol{k})\rangle_ {mn} = \bra{m(\boldsymbol{k})} \nabla H (\boldsymbol{k}) \ket{n(\boldsymbol{k})}$ for $\\{ \ket{n (\boldsymbol{k})} \\}$ the (orthonormal) set of energy eigenstates of $H(\boldsymbol{k})$. Thus, we apply the above equation to $\boldsymbol{H}_ {MF}^\xi(\boldsymbol{k})$, the mean-field Hamiltonian with the mass term.

At each point $\boldsymbol{k}$, we calculate numerically the eigenstates of $\boldsymbol{H}_ {0}(\boldsymbol{k})$ and $\boldsymbol{H}_ {0}(\boldsymbol{k}+\boldsymbol{Q})$ to derive $\ket{+(\boldsymbol{k})}$ and $\ket{+(\boldsymbol{k}+\boldsymbol{Q})}$ and obtain $\boldsymbol{H}_ {MF}^\xi(\boldsymbol{k})$. Then, we calculate numerically the eigenvalues and eigenstates of $\boldsymbol{H}_ {MF}^\xi(\boldsymbol{k})$ to use in the equation of the Berry curvature above. Take note that we must take the gradient of $\boldsymbol{H}_ {MF}(\boldsymbol{k})$ as we do not have a closed-form expression of the mass term in terms of $\boldsymbol{k}$. Fortunately, this is not a problem because the contribution of the gradient of the mass term in the total is negligible.

The mass term needs to be chosen appropriately for the numerical evaluation of the Berry phase. For a simple linear dispersion, a fraction $\[1 + (\rho/\xi)^2 \]^{-\tfrac{1}{2}}$ of the delta function's weight is lost outside of a radius $\rho$ around the Dirac point. Hence, $\rho$ and $xi$ must be chosen for $\rho / \xi$ to be large enough. In particular, about $1%$ of the delta function's weight is lost when taking $\rho/\xi = 100$. Asymmetry in the growth rate of the gap needs to be considered when choosing the mass term and the grid. We can assume the loss from a more general dispersion to be negligible because the correction is of order $\lambda \ll 1$ under a typical parameter set.

It is enough to only compute the Berry phase of the hole pocket surrounding $(\tfrac{\pi}{2a}, \tfrac{\pi}{2a})$ for an electron in the lowest energy band of the high-energy subspace. Additionally, we need to take a few things specific to our case into consideration. It can be seen numerically that the distance between the Dirac point and the boundary of the surrounding hole pocket ranges from approximately $0.025\[a/\pi\]^{-1}$ to $0.16\[a/\pi\]^{-1}$ for both current patterns.

To make the calculation more efficient, the elongated shape of the hole pocket and its positioning are taken into account. Thus, the origin of the momentum space is translated to $(\tfrac{\pi}{2a}, \tfrac{\pi}{2a})$ to correspond to the degeneracy point; the space is then rotated clockwise by $\pi/4$. Furthermore, a rectangular grid centered at the origin of this transformed momentum space is taken with the horizontal 10 times shorter than the vertical. The same number of discrete points is taken horizontally and vertically. Specifically, the vertical side is set to have a length of $1 \cdot 10^{-2} \[a/\pi\]^{-1}$ with a grid spacing of $5 \cdot 10^{-6} \[a/\pi\]^{-1}$ along this direction. These numbers differ from the thesis to give more precise results. Finally, a mass term of $\xi = 1 \cdot 10^{-5} \[a/\pi\]^{-1}$ is chosen.

![11_berry-curvature](https://github.com/ebolduc37/msc-thesis/assets/44382376/6c3e35db-0f61-4818-a7b1-965700d3b903)
![12_berry-curvature-zoom](https://github.com/ebolduc37/msc-thesis/assets/44382376/df9e9102-f8d3-46bc-8a57-17694e1519a3)

Finally, we perform a numerical integration over the area of the hole pocket to obtain the Berry phase. The results are listed in the table below.

| $\phi$ | $\gamma_-(C)/\pi$ | $\[1 + (\rho/\xi)^2 \]^{-\tfrac{1}{2}}$ |
| :----: | :---------------: | :--: |
| $+1$   | 99.4%             | 0.6% |
| $-1$   | 99.3%             | 0.6% |

For both possible current patterns, a Berry phase nearly equal to $\pi$ is obtained, all within expectations: almost equal to $\pi$, but not exactly because of the delta function's weight that is lost outside of the grid. In fact, we obtain a total value of $\pi$ within the rounding of significant figures if we add the weight loss to the Berry phase. Unknown discretization errors may be at play in the small discrepancy.

### 2　Peierls Substitution

_Work in progress._




