# PaiNN (Polarizable atom Interaction Neural Network) — math-y cheat sheet

## Graph, nodes & edges

* **Atoms (nodes):** Index $i=1,\dots,N$ with nuclear charge $Z_i \in \mathbb{N}$ and position $\mathbf{r}_i \in \mathbb{R}^3$.
* **Neighbors (directed edges):** $j \in \mathcal{N}(i) = \{ j \neq i : \|\mathbf{r}_{ij}\| \le r_{\text{cut}} \}$ with $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, distance $r_{ij} = \|\mathbf{r}_{ij}\|$, and unit direction $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij} / r_{ij}$.

## Features (per atom)

* **Scalar (rotation–invariant):** $q_i \in \mathbb{R}^{F}$.
* **Vector (rotation–equivariant):** $\boldsymbol{\mu}_i \in \mathbb{R}^{3 \times F}$ (3D direction for each of $F$ channels).
* **Init:** $q_i^{(0)} = \mathrm{Embed}(Z_i)$, $\boldsymbol{\mu}_i^{(0)} = \mathbf{0}$.

## Edge feature engineering (radial)

* **Radial basis:** $\boldsymbol{\phi}(r_{ij}) \in \mathbb{R}^{B}$, e.g.
  $$\phi_n(r) = \frac{\sin\!\left(n\pi r / r_{\text{cut}}\right)}{r}, \quad n=1,\dots,B.$$
* **Cutoff:** $f_{\text{cut}}(r)$ (e.g., cosine) to enforce smooth locality.
* **Filters (per edge):**
  $$\mathbf{W}_{ij} = \mathrm{MLP}_{\text{filter}}\big(\boldsymbol{\phi}(r_{ij})\big) \cdot f_{\text{cut}}(r_{ij}).$$
  (Optionally shared across interaction blocks.)

## High-level architecture

Repeat **L** times (interaction + mixing), with residuals:
$$
(q_i,\boldsymbol{\mu}_i) \xrightarrow{\text{Interaction}} (q_i',\boldsymbol{\mu}_i') \xrightarrow{\text{Mixing}} (q_i^{+},\boldsymbol{\mu}_i^{+}).
$$

---

## 1) Interatomic **interaction** (message passing)

A small atomwise MLP produces **3F** per-atom channels:
$$
\mathbf{x}_i = \mathrm{MLP}_{\text{inter}}(q_i) \in \mathbb{R}^{3F}, \qquad
\mathbf{x}_i \equiv \big[\underbrace{\mathbf{x}_i^{(q)}}_{F},\underbrace{\mathbf{x}_i^{(R)}}_{F},\underbrace{\mathbf{x}_i^{(\mu)}}_{F}\big].
$$

For each edge $(i \leftarrow j)$:

* **Gate by filter:** $\tilde{\mathbf{x}}_{ij} = \mathbf{W}_{ij} \odot \mathbf{x}_j$.
* **Split:** $\tilde{\mathbf{x}}_{ij}^{(q)}, \tilde{\mathbf{x}}_{ij}^{(R)}, \tilde{\mathbf{x}}_{ij}^{(\mu)} \in \mathbb{R}^{F}$.

Aggregate to the receiver $i$ (scatter-add over neighbors):

* **Scalar message (to $q_i$):**
  $$\Delta q_i^{\text{(m)}} = \sum_{j \in \mathcal{N}(i)} \tilde{\mathbf{x}}_{ij}^{(q)}.$$
* **Vector message (to $\boldsymbol{\mu}_i$):**
  $$
  \Delta \boldsymbol{\mu}_i^{\text{(m)}} = \sum_{j \in \mathcal{N}(i)} \left(
  \tilde{\mathbf{x}}_{ij}^{(R)} \otimes \hat{\mathbf{r}}_{ij}
  + \tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j
  \right).
  $$
  (First term injects **new** directional info via $\hat{\mathbf{r}}_{ij}$; second term **propagates** existing equivariant info $\boldsymbol{\mu}_j$ across the graph.)

Residual update:
$$
q_i \leftarrow q_i + \Delta q_i^{\text{(m)}}, \qquad
\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i + \Delta \boldsymbol{\mu}_i^{\text{(m)}}.
$$

---

## 2) Intra-atomic **mixing** (equivariant gating)

Linear mix two vector streams from $\boldsymbol{\mu}_i$:
$$
[\boldsymbol{\mu}_{V,i}\ \ \boldsymbol{\mu}_{W,i}] = \mathrm{Linear}(\boldsymbol{\mu}_i), \qquad
\boldsymbol{\mu}_{V,i},\ \boldsymbol{\mu}_{W,i} \in \mathbb{R}^{3 \times F}.
$$
Compute per-feature norms (used as scalar context):
$$
\mathbf{s}_{V,i} = \|\boldsymbol{\mu}_{V,i}\|_2 \in \mathbb{R}^{F} \quad \text{(norm over the 3D axis)}.
$$
Feed $[q_i; \mathbf{s}_{V,i}]$ to an MLP yielding **3F** scalars:
$$
\big[\delta \mathbf{q}^{\text{(intra)}},\ \delta \boldsymbol{\mu}^{\text{(intra)}},\ \delta \mathbf{q\mu}^{\text{(intra)}}\big]
= \mathrm{MLP}_{\text{intra}}\big([q_i; \mathbf{s}_{V,i}]\big).
$$

Form scalar-vector couplings:
$$
\langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle
= \sum_{a=1}^3 \boldsymbol{\mu}_{V,i}^{(a)} \odot \boldsymbol{\mu}_{W,i}^{(a)} \in \mathbb{R}^{F}.
$$

Residual updates (per feature $f=1,\dots,F$):
$$
\begin{aligned}
q_i &\leftarrow q_i + \delta \mathbf{q}^{\text{(intra)}}
+ \delta \mathbf{q\mu}^{\text{(intra)}} \odot \langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle, \\
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i
+ \big(\delta \boldsymbol{\mu}^{\text{(intra)}} \odot \boldsymbol{\mu}_{W,i}\big).
\end{aligned}
$$

These gates are **equivariant**: all nonlinearities are scalar; vectors are only scaled or linearly combined, preserving $R\boldsymbol{\mu}$ under a global rotation $R \in \mathrm{SO}(3)$.

---

## Symmetries & complexity

* **Translation invariance:** only relative positions $\mathbf{r}_{ij}$ are used.
* **Permutation invariance:** atomwise updates + neighbor sums.
* **Rotation properties:** scalars $q_i$ invariant; vectors $\boldsymbol{\mu}_i$ equivariant.
* **Per-block cost:** $\mathcal{O}(|E|F)$ with $|E| = \sum_i |\mathcal{N}(i)| = \mathcal{O}(N)$ for fixed cutoff; avoids $\mathcal{O}(N|\mathcal{N}|^2)$ angle triplets.

---

## Readouts (examples)

### (a) Scalar properties (e.g., energy)

Atomwise head $\varepsilon(\cdot)$ on $q_i$ and sum:
$$
E = \sum_{i=1}^N \varepsilon(q_i).
$$
**Forces** enforced by energy conservation:
$$
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
$$

### (b) Molecular dipole (rank-1 tensor)

Latent atomic **charges** and **dipoles** from final features:
$$
\boldsymbol{\mu} = \sum_{i=1}^N \left(
\underbrace{\boldsymbol{\mu}_{\text{atom}}(\boldsymbol{\mu}_i)}_{\text{local dipole}}
+ \underbrace{q_{\text{atom}}(q_i)}_{\text{charge}} \ \mathbf{r}_i
\right).
$$

### (c) Polarizability (rank-2 tensor)

$$
\boldsymbol{\alpha} = \sum_{i=1}^N \left(
\alpha_0(q_i)\mathbf{I}_3
+ \boldsymbol{\nu}(\boldsymbol{\mu}_i) \otimes \mathbf{r}_i
+ \mathbf{r}_i \otimes \boldsymbol{\nu}(\boldsymbol{\mu}_i)
\right).
$$

(Here, $\varepsilon$, $q_{\text{atom}}$, $\boldsymbol{\mu}_{\text{atom}}$, $\alpha_0$, $\boldsymbol{\nu}$ are small MLP heads; $\boldsymbol{\nu}(\cdot) \in \mathbb{R}^3$.)

---

## Layer shapes & flow (one block)

* **Inputs:** $q \in \mathbb{R}^{N \times F}$, $\boldsymbol{\mu} \in \mathbb{R}^{N \times 3 \times F}$.
* **Edge preprocessing:** $r_{ij}$, $\hat{\mathbf{r}}_{ij}$, $\boldsymbol{\phi}(r_{ij})$, $\mathbf{W}_{ij}$.
* **Interaction MLP:** $F \to 3F$ per atom → gate per edge → neighbor sum.
* **Mixing:** vector linear $F \to 2F$; norm over $3$ → concat $[q; s_V] \in \mathbb{R}^{2F}$ → MLP $2F \to 3F$.
* **Residuals:** add to get next $(q,\boldsymbol{\mu})$.
* **Repeat** $L$ times; final heads read from $q$ and/or $\boldsymbol{\mu}$.

---

## Why it works (core math ideas)

* **Equivariance preserved** by restricting nonlinearities to scalars and using only linear/scalar scaling on vectors.
* **Directional information** is injected via $\hat{\mathbf{r}}_{ij}$ and **propagated** across hops via the $\tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j$ term, achieving angular sensitivity with **linear** neighbor complexity $\mathcal{O}(|\mathcal{N}(i)|)$, unlike angle-triplet models.
* **Scalar–vector coupling** (inner products, norms) lets vectors influence scalars while maintaining invariance of scalar channels.

---

## Minimal set of equations (all together)

**Edge filters**
$$
\mathbf{W}_{ij} = \mathrm{MLP}_{\text{filter}}(\boldsymbol{\phi}(r_{ij})) \cdot f_{\text{cut}}(r_{ij}), \qquad \mathbf{W}_{ij} \in \mathbb{R}^{3F}.
$$

**Interaction**
$$
\begin{aligned}
\mathbf{x}_j &= \mathrm{MLP}_{\text{inter}}(q_j) \in \mathbb{R}^{3F}, \\
\tilde{\mathbf{x}}_{ij} &= \mathbf{W}_{ij} \odot \mathbf{x}_j, \\
\Delta q_i^{\text{(m)}} &= \sum_{j\in\mathcal{N}(i)} \tilde{\mathbf{x}}_{ij}^{(q)}, \\
\Delta \boldsymbol{\mu}_i^{\text{(m)}} &= \sum_{j\in\mathcal{N}(i)} \left(
\tilde{\mathbf{x}}_{ij}^{(R)} \otimes \hat{\mathbf{r}}_{ij}
+ \tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j
\right).
\end{aligned}
$$

**Mixing**
$$
\begin{aligned}
[\boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i}] &= \mathrm{Linear}(\boldsymbol{\mu}_i), \\
\mathbf{s}_{V,i} &= \|\boldsymbol{\mu}_{V,i}\|_2, \\
[\delta\mathbf{q}, \delta\boldsymbol{\mu}, \delta\mathbf{q\mu}] &= \mathrm{MLP}_{\text{intra}}([q_i; \mathbf{s}_{V,i}]), \\
q_i &\leftarrow q_i + \Delta q_i^{\text{(m)}} + \delta\mathbf{q} + \delta\mathbf{q\mu} \odot \langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle, \\
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i + \Delta \boldsymbol{\mu}_i^{\text{(m)}} + \delta\boldsymbol{\mu} \odot \boldsymbol{\mu}_{W,i}.
\end{aligned}
$$

That’s the PaiNN core: scalable equivariant message passing (via $\hat{\mathbf{r}}_{ij}$ and vector transport), plus tight scalar–vector coupling for expressive, symmetry-respecting predictions of both scalar and tensorial molecular properties.
