# Determination of the Berry Phase in the Staggered Loop Current Model of the Pseudogap in the Cuprates.

By Etienne Bolduc, Department of Physics, McGill University, Montreal, Canada. (August 2019)  
A thesis submitted to McGill University in partial fulfillment of the requirements of the degree of Master of Science.

## Abstract

High-temperature superconductivity in the cuprates has been at the heart of many debates since its discovery more than 30 years ago. No consensus has been reached yet about the underlying physics, but plausible descriptions usually fall into two categories each carrying various propositions. The quantum oscillations data acquired over the past few years for the normal state of the cuprates under a strong magnetic field has recently been used to obtain the electronic Berry phase of different compounds, which manifests through the phase mismatch in quantum oscillations. This analysis revealed an electronic Berry phase of $0 \pmod{2\pi}$ in three hole-doped compounds and $1.4\pi \pmod{2\pi}$ in one electron-doped compound. To investigate the mysterious pseudogap phase of the cuprates, the theoretical candidate known as the circulating current state of Varma as approached by Bulut is analyzed to numerically evaluate through a semiclassical approach the electronic Berry phase in this normal state. Under a typical parameter set in line with experimental data, a phase of $\pi$ is found. A comparison of the semiclassical approach with the Peierls substitution applied to this model confirms this result and further leads to an uncertainty on the phase of order $0.01\pi$. Hence, the circulating current state is incompatible with quantum oscillation data according to the Berry phase.

## Numerical Analysis

The present section is a description of the numerical analysis behind this thesis. While numeric computing—including plotting and graphing—was initially performed on MATLAB, it was converted to Python for portability and shareability. The images used here come from the figures generated in Python. For more information on the theoretical background of this work and to see all derivations in detail, consult the [full thesis](bolduc-2019-determination-of-the-berry-phase-in-the-staggered-loop-current-model-of-the-pseudogap-in-the-cuprates.pdf).

### 1　Semiclassical approach

In an effort to describe the pseudogap phase of cuprates, Varma suggested in 1997 a competing order model, a three-band model with the particularity of having current circulating in each unit cell as in Figure 4.2, leading to this phase being referred to as the circulating current phase [50]. It was argued then and shown later that the properties of this phase are similar to those of the pseudogap phase [51]. A few years later, weak magnetic moments were detected below $T^*$ through spin-polarized neutron scattering experiments [15, 24, 25, 45], influencing Varma to put forward the idea of intra-unit cell loop currents (LCs) [52].

More recently, Bulut _et al._ have investigated a phase with a staggered pattern of LCs called $\pi\textrm{LC}$ phase [7]. It features the ordering wave vector $\textbf{Q} = \tfrac{1}{a} (\pi,\pi)$ where $a$ is the lattice spacing. This wave vector is the one relevant to cuprates [29] and can also be found in other proposals for competing order, such as the d-density wave ($\textrm{DDW}$) state [9]. As demonstrated in Figure 4.1, it plays an essential role in the Fermi surface reconstruction suggested to be behind the small Fermi surface of the pseudogap phase and the hole and electron pockets observed in experiments [8]. This section shows that a d-wave-symmetric gap will be maintained in the energy spectrum of the $\pi\textrm{LC}$ state, similar to the $\textrm{DDW}$ state and in agreement with the pseudogap phase [29]. Additionally, the $\pi\textrm{LC}$ state breaks time-reversal symmetry and could explain the Kerr effect observed in the pseudogap phase through experiments [7].

The $\pi\textrm{LC}$ model analyzed in this [thesis](bolduc-2019-determination-of-the-berry-phase-in-the-staggered-loop-current-model-of-the-pseudogap-in-the-cuprates.pdf) is the same as the one explored by Bulut _et al._ [7], but an alternate current pattern is also investigated—ultimately leading to the same Berry phase.

The $\pi\textrm{LC}$ Hamiltonian is written on a square lattice and each site corresponds to a unit cell, i.e., a $\textrm{CuO}_ 2$ plane containing a copper $d_{x^2-y^2}$ orbital and oxygen $p_x$ and $p_y$ orbitals, denoted by $\textrm{Cu}d_{x^2-y^2}$, $\textrm{O}p_x$, and $\textrm{O}p_y$. The relevant bonds are the nearest neighbor $p–d$ and $p–p$ bonds. As discussed before, these bonds exhibit intra-unit cell loop currents, equivalent to directional hopping. Additionally, the current must switch direction between unit cells like in Figure 4.2 to obtain the Fermi surface reconstruction found in Figure 4.1. Any state under such considerations breaks both time-reversal and lattice-translation symmetries. The specific staggered pattern of intertwined LCs studied by Bulut _et al._ [7] is shown in Figure 4.2. Note that this state has 4-fold rotational symmetry and conserves current.

The $\pi\textrm{LC}$ Hamiltonian denoted by $\hat{H}$ can be broken down into two parts: the kinetic energy $\hat{H}_0$ and the charge order $\hat{H}^\prime$. Each one can be solved separately. However, a mean-field approach is needed to solve the charge order, which will lead to the final diagonalized Hamiltonian being a mean-field approximation.

#### 1.1　Mean-Field Hamiltonian

##### Kinetic Energy

The choice of the unit cell for a single $\textrm{CuO}_ 2$ plane and of orbital phase convention can be found in Figure 4.3. The inequivalent bonds are numbered to distinguish them from one another. We assume a total of $N^2$ unit cells with periodic boundary conditions. Each unit cell will be labeled by its position through a vector $\textbf{i} = (i^x, i^y) \in \mathbb{Z}_ N^2$. Also, the orbitals will be labeled by $d$ for $\textrm{Cu}d_{x^2-y^2}$, $x$ for $\textrm{O}p_x$, and $y$ for $\textrm{O}p_y$.

Let $\hat{c}_ {\textbf{i}\alpha\sigma}$ / $\hat{c}_ {\textbf{i}\alpha\sigma}^\dagger$ / $\hat{n}_ {\textbf{i}\alpha\sigma}$ be the annihilation/creation/number operator for an electron in orbital $\alpha$ of unit cell $\textbf{i}$ and spin $\sigma$. Let $\epsilon_\alpha \in \\{ \epsilon_d , \epsilon_p \\}$ be the orbital energy of orbital $\alpha$ and $t_ {\textbf{j}\beta , \textbf{i}\alpha} \in \\{ t_{pd} , t_{pp} \\}$ the tunneling matrix element from orbital $\alpha$ in unit cell $\textbf{i}$ to orbital $\beta$ in unit cell $\textbf{j}$. The kinetic energy is then

```math
\hat{H}_{0} = \sum_{\textbf{i}\alpha} \epsilon_\alpha \hat{n}_ {\textbf{i}\alpha}
+ \sum_{\langle \textbf{i}\alpha ,\, \textbf{j}\beta \rangle}
t_ {\textbf{j}\beta , \textbf{i}\alpha} \hat{c}_ {\textbf{j}\beta}^\dagger \hat{c}_ {\textbf{i}\alpha}
```

where the sum over $\langle \textbf{i}\alpha\sigma , \textbf{j}\beta\sigma \rangle$ includes nearest neighbor $p–d$ and $p–p$ bonds [7]. The spin labels can be omitted here. Note that $t_ {\textbf{j}\beta , \textbf{i}\alpha}$ can be expressed as $t^n_ {\beta \alpha}$, in terms of $\alpha$, $\beta$, and $n$ the inequivalent bond between orbital $\alpha$ in unit cell $\textbf{i}$ and orbital $\beta$ in unit cell $\textbf{j}$. This expression of the tunneling matrix elements has the particularity that it does not depend on $\textbf{i}$ and $\textbf{j}$ anymore: only the inequivalent bond and corresponding orbitals are required.

Let $\hat{c}_ {\textbf{k}\alpha}$ / $\hat{c}_ {\textbf{k}\alpha}^\dagger$ / $\hat{n}_ {\textbf{k}\alpha}$ be the annihilation/creation/number operator for an electron in orbital $\alpha$ with crystal momentum $\hbar \textbf{k}$ with $\textbf{k} = ( k^x, k^y ) \in \tfrac{2\pi}{aN} \mathbb{Z}_ N^2$. By periodicity, the Brillouin zone $BZ$ is taken to be the set $\tfrac{1}{a} \[-\pi , \pi\] \times \tfrac{1}{a} \[-\pi , \pi\]$. The momentum-space operators are then related to their position-space counterparts through the Fourier transform. After some calculations where we replace the position-space operators by their Fourier transform and introduce the ordering wave vector $\textbf{Q} = \tfrac{1}{a} (\pi,\pi)$ for the Fermi surface reconstruction, we obtain that

```math
\hat{H}_{0} = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & \textbf{0} \\
\textbf{0} & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}
\overline{\Psi}_\textbf{k}
```

where $BZ^\prime$ is the reduced Brillouin zone pictured in Figure 4.4, which consists of the closed ball of radius $\tfrac{1}{a} \pi$ centered at the origin with p-norm $L_1$. We defined $\overline{\Psi}_ \textbf{k}^\dagger \equiv \[ \Psi_ \textbf{k}^\dagger \Psi_{\textbf{k} + \textbf{Q}}^\dagger \]$ with $\Psi_\textbf{k}^\dagger \equiv \[ \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \hat{c}_ {\textbf{k}\alpha\sigma}^\dagger \]$ and applied a gauge transformation such that $\hat{c}_ {\textbf{k} x} \rightarrow i\hat{c}_ {\textbf{k} x}$ and $\hat{c}_ {\textbf{k} y} \rightarrow i\hat{c}_ {\textbf{k} y}$. More importantly, we have that

```math
\begin{align}
& \textbf{H}_{0}(\textbf{k}) \equiv \\
& \begin{bmatrix}
\epsilon_d & 2 t_{pd} s_x & -2 t_{pd} s_y \\
2 t_{pd} s_x & \epsilon_p & 4 t_{pp} s_x s_y \\
-2 t_{pd} s_y & 4 t_{pp} s_x s_y & \epsilon_p
\end{bmatrix}
\end{align}
```

for $s_x \equiv \sin{(\tfrac{a}{2} k^x)}$ and $s_y \equiv \sin{(\tfrac{a}{2} k^y)}$.

MAYBE SHOW THE ENERGY GAP? AND E(k) = E(k+Q)

##### Mean-Field Decomposition of the Charge Order

Let $U_\alpha$ and $V_{\beta \alpha}$ be the intraorbital and interorbital Coulomb interactions respectively. The charge order in position space is

```math
\hat{H}^\prime = \sum_{\textbf{i}\alpha} U_\alpha \hat{n}_{\textbf{i}\alpha\uparrow} \hat{n}_{\textbf{i}\alpha\downarrow}
+ \sum_{\langle \textbf{i}\alpha\sigma ,\, \textbf{j}\beta\sigma^\prime \rangle} \frac{1}{2} V_{\beta \alpha} \hat{n}_{\textbf{j}\beta\sigma^\prime} \hat{n}_{\textbf{i}\alpha\sigma}
```

where the sum over $\langle \textbf{i}\alpha\sigma , \textbf{j}\beta\sigma^\prime \rangle$ includes nearest neighbor $p–d$ and $p–p$ bonds [7].

From $\hat{H}^\prime$, it is possible to obtain the mean-field version $\hat{H}^\prime_{MF}$ which includes circulating currents. The intraorbital interactions lead to Hartree shifts to the orbital energies which merely renormalize in $\epsilon_d$ and $\epsilon_p$. Since we are looking for solutions that do not break spin-rotational symmetries, the spin labels are omitted for the rest of this. The interorbital interactions can be decomposed in terms of the circulating current: the Hermitian operator $\hat{J}_ {\textbf{j}\beta , \textbf{i}\alpha} = -it_ {\textbf{j}\beta , \textbf{i}\alpha} \[ \hat{c}_ {\textbf{j}\beta}^\dagger \hat{c}_ {\textbf{i}\alpha} - \hat{c}_ {\textbf{i}\alpha}^\dagger \hat{c}_ {\textbf{j}\beta} \]$ is the current operator for the bond between $\textbf{i}\alpha$ and $\textbf{j}\beta$ such that the current flows from $\textbf{i}\alpha$ to $\textbf{j}\beta$ if $\langle \hat{J}_ {\textbf{j}\beta , \textbf{i}\alpha} \rangle \equiv z_ {\textbf{j}\beta , \textbf{i}\alpha} > 0$ and from $\textbf{j}\beta$ to $\textbf{i}\alpha$ if $z_ {\textbf{j}\beta , \textbf{i}\alpha} < 0$.

After applying the above framework to obtain the mean-field version of $\hat{H}^\prime$, we obtain that

```math
\hat{H}_{MF}^\prime = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger
\begin{bmatrix}
\widetilde{\boldsymbol{\epsilon}} & \textbf{H}_{1}(\textbf{k}) \\
\textbf{H}_{1}^\dagger(\textbf{k}) & \widetilde{\boldsymbol{\epsilon}}
\end{bmatrix}
\overline{\Psi}_\textbf{k}
```

where an explicit expression for $\textbf{H}_ {1}(\textbf{k})$ depends on the current pattern. Note that $\widetilde{\boldsymbol{\epsilon}}$ simply represents a shift in the orbital energies equivalent to $\epsilon_d = V_ {pd} + 2V_ {pp}$ and $\epsilon_p = 2V_{pd}$.

The focus needs to be made on _physical_ current patterns—meaning that the current is conserved on each orbital site—with 4-fold rotational symmetry. Such current patterns have the particularity that all elements of $\textbf{H}_ {1}(\textbf{k})$ _after_ the gauge transformation are purely imaginary. There are only two possible inequivalent physical current patterns with 4-fold rotational symmetry, and the one investigated by Bulut _et al._ [7] shown in Figure 4.2 is one of them. The second current pattern can be obtained from the one in Figure 4.2 by merely inverting the current along $p–p$ bonds.

Throughout the rest of this work, the two current patterns satisfying the above conditions will be distinguished by $\phi \in \\{ \pm 1 \\}$ with the current pattern investigated by Bulut _et al._ [7] corresponding to $\phi = 1$. Explicitly, we have for these two current patterns under the gauge transformation from earlier that $\widetilde{\boldsymbol{\epsilon}}$ stays invariant while

```math
\begin{align}
& \textbf{H}_{1}(\textbf{k}) = \\
& \begin{bmatrix}
0 & 2 i R_{pd} c_x & 2i R_{pd} c_y \\
-2i R_{pd} s_x & 0 & -4i \phi R_{pp} s_x c_y \\
-2i R_{pd} s_y & 4i \phi R_{pp} c_x s_y & 0
\end{bmatrix}
\end{align}
```

for $R_{pd} \equiv \tfrac{V_{pd} z_{pd}}{t_{pd}}$, $R_{pp} \equiv \tfrac{V_{pp} z_{pp}}{t_{pp}}$, $c_x \equiv \cos{(\tfrac{a}{2} k^x)}$, and $c_x \equiv \cos{(\tfrac{a}{2} k^x)}$.

##### Full Mean-Field Hamiltonian

Combining both (4.7) and (4.11) yields the effective mean-field $\pi\textrm{LC}$ Hamiltonian

```math
\hat{H}_{MF} = \sum_{\textbf{k} \in BZ'} \overline{\Psi}_\textbf{k}^\dagger \textbf{H}_{MF}(\textbf{k}) \overline{\Psi}_\textbf{k}
```
with

```math
\textbf{H}_{MF}(\textbf{k}) =
\begin{bmatrix}
\textbf{H}_{0}(\textbf{k}) & \textbf{H}_{1}(\textbf{k}) \\
\textbf{H}_{1}^\dagger(\textbf{k}) & \textbf{H}_{0}(\textbf{k} + \textbf{Q})
\end{bmatrix}
```

where $\epsilon_ d \rightarrow \epsilon_ d + \widetilde{\epsilon}_ d \equiv \varepsilon_ d$ and $\epsilon_ p \rightarrow \epsilon_ p + \widetilde{\epsilon}_ p \equiv \varepsilon_ p$ in $\textbf{H}_{0}(\textbf{k})$.

#### 1.2　Fermi Surface

Bla

#### 1.3　Mean-Field Hamiltonian with Mass Term

Bla

#### 1.4　Berry Curvature

Bla

#### 1.5　Berry Phase

Bla

### 2　Peierls Substitution
