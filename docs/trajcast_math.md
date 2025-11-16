# TrajCast: Equivariant Trajectory Forecasting in SEGNN Notation

This note reverse-engineers the TrajCast implementation bundled with this repository and reconciles it with the public description of the model in *Force-Free Molecular Dynamics Through Autoregressive Equivariant Networks* (Thiemann et al., 2025) [arXiv:2503.23794](https://arxiv.org/abs/2503.23794). All code references point to `models/trajcast/...` unless stated otherwise. Symbols follow SEGNN-style conventions: irreps are indexed by rotation order $\ell$, parity $p\in\{\pm 1\}$, and channel multiplicity $m$.

---

## 1. Atomic graph state and baseline notation

Each simulation frame is represented as an `AtomicGraph` $(\mathcal{V}, \mathcal{E})$ (see `datasets/MD/atomic_graph/atomic_graph.py`). A node $i\in\mathcal{V}$ stores

- fixed attributes: position $\mathbf{x}_i\in\mathbb{R}^3$, atomic type $Z_i$, atomic mass $m_i$;
- dynamical state: velocity $\mathbf{v}_i$ and (normalised) displacement target $\Delta\mathbf{x}_i$ when present;
- optional metadata: thermostat control, timestep encoding, etc.

Directed edges $(i \leftarrow j)\in\mathcal{E}$ are created for interatomic distances $\|\mathbf{x}_i-\mathbf{x}_j\|\leq r_\text{cut}$ with periodic wrap-around shifts. The helper `compute_edge_vectors()` injects

$$
\mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i+\mathbf{s}_{ij},\qquad r_{ij}=\|\mathbf{r}_{ij}\|,
$$

used for spherical harmonics and radial basis evaluations. The graph keeps track of an empirical average neighbour count

$$
\bar{N} = \frac{1}{|\mathcal{V}|}\sum_{i} \deg(i),
$$

either provided in the config or estimated from the dataset (`Trainer.__init__`).

---

## 2. Feature engineering pipeline

TrajCast builds equivariant node and edge features before message passing (`TrajCastModel._build_model`).

### 2.1 Species embedding

`OneHotAtomTypeEncoding` maps integer species to a channel-wise scalar irrep:

$$
\mathbf{e}_i \in \mathbb{R}^{C_Z} \sim C_Z \times (0^+),
$$

where $C_Z$ is the number of chemical species (Section 2.1 of the paper emphasises element awareness).

### 2.2 Radial distance basis with cutoff

`EdgeLengthEncoding` applies a trainable Bessel basis (`BesselBasisTrainable`) with a DimeNet-style polynomial cutoff,

$$
\boldsymbol{\phi}_r(r_{ij}) = \text{cut}(r_{ij}) \cdot \left[\sin(\beta_k r_{ij})/(r_{ij}+\varepsilon)\right]_{k=1}^{K_r},
$$

with learnable roots $\{\beta_k\}$ and $K_r=\texttt{num\_edge\_rbf}`` (default 8). The multiplicity corresponds to $K_r\times(0^+)$.

### 2.3 Velocity norm basis conditioned on species

`ElementBasedNormEncoding` computes $\|\mathbf{v}_i\|$ and expands it in a fixed Gaussian radial basis (`FixedBasis`) capped at `vel_max`. A fully connected tensor product couples this basis to the species one-hot vector, yielding

$$
\boldsymbol{\phi}_v(\|\mathbf{v}_i\|, Z_i) \sim K_v \times (0^+),
$$

where $K_v=\texttt{num\_vel\_rbf}``.

### 2.4 Spherical harmonic projections

Two projections with component normalisation (`normalization="component"`) are used:

- Edge geometry:
  $$
  \mathbf{Y}_{ij}^{(\ell)} = Y^{(\ell)}(\hat{\mathbf{r}}_{ij}) \in ( \ell^{(-1)^\ell} ).
  $$
- Velocity orientation (per node):
  $$
  \mathbf{Y}_{i}^{(\ell)} = Y^{(\ell)}(\hat{\mathbf{v}}_{i}),\quad \text{with } \ell\leq \ell_{\max}.
  $$

When assembling the initial node feature, TrajCast concatenates species scalars, velocity norms, and *only the $\ell\geq 1$* spherical harmonics of velocities:

$$
\mathbf{h}_i^{(0)} = \left[\mathbf{e}_i \,\|\, \boldsymbol{\phi}_v(\|\mathbf{v}_i\|,Z_i) \,\|\, \mathbf{Y}_{i}^{(\ell)}\big|_{\ell\geq 1}\right].
$$

The omission of $\ell=0$ velocity modes avoids duplicating scalar channels already covered by the norm basis.

### 2.5 Initial tensor mixing

`LinearTensorMixer` lifts $\mathbf{h}_i^{(0)}$ into a structured set of irreps:

$$
\mathbf{H}_i^{(0)} = \bigoplus_{\ell=0}^{\ell_{\max}}\left( \mathbb{R}^{m_\ell}\otimes \ell^+ \right)
\oplus
\bigoplus_{\ell=0}^{\ell_{\max}}\left( \mathbb{R}^{m_\ell}\otimes \ell^- \right),
$$

with $m_\ell=\texttt{num\_hidden\_channels}``. Both even and odd parity copies are retained so later gated activations can choose appropriate channels (`models.py`, `LinearTypeEncoding`). This is analogous to the "feature mixing tensor" in the TrajCast paper (Fig. 2 in [Thiemann et al., 2025]citeturn8search9).

---

## 3. Residual conditioned message passing block

Each layer (`ResidualConditionedMessagePassingLayer`) implements a two-stage tensor product message followed by gating and a ResNet correction.

### 3.1 Edge message construction

For each edge $(i\leftarrow j)$ the layer forms a depthwise tensor product (DTP) between sender features and geometric harmonics:

$$
\mathbf{m}_{ij} = \operatorname{DTP}\!\left(\mathbf{H}_j^{(\ell)},\;\mathbf{Y}_{ij}^{(\ell)};\; \mathbf{w}_{ij}^{(r)}\right),
$$

where $\mathbf{w}_{ij}^{(r)} = \text{MLP}_r(\boldsymbol{\phi}_r(r_{ij}))$ provides scalar weights per tensor path. The multiplicity handling mode `"uvu"` enforces that each output irrep inherits the minimal multiplicity across inputs to avoid overparameterisation (`_tensor_cross_interactions.py`).

### 3.2 Neighbour aggregation with degree normalisation

Messages are degree-normalised using the running average $\bar{N}$:

$$
\bar{\mathbf{m}}_i = \frac{1}{\bar{N}} \sum_{j\in\mathcal{N}(i)} \mathbf{m}_{ij}.
$$

Splitting by irreps, a linear contraction reduces redundant multiplicities:

$$
\tilde{\mathbf{m}}_i = \mathbf{W}_\text{contract} \bar{\mathbf{m}}_i,
$$

acting block-diagonally on each $(\ell,p)$.

### 3.3 Velocity-conditioned update TP

Velocity embeddings influence updates via a second DTP:

$$
\mathbf{u}_i = \operatorname{DTP}\!\left(\tilde{\mathbf{m}}_i,\; \mathbf{Y}_i^{(\ell)};\; \mathbf{w}_i^{(v)}\right),
$$

with weights $\mathbf{w}_i^{(v)} = \text{MLP}_v(\boldsymbol{\phi}_v(\|\mathbf{v}_i\|, Z_i))$. This implements the forecast-conditioning described in Section 3.2 of the paperciteturn8search9: velocities steer the propagation kernel rather than only the readout.

### 3.4 Linear mixing, ResNet, and gated activation

1. **Linear mixing:** A final equivariant linear map produces candidate updates
   $$
   \mathbf{z}_i = \mathbf{W}_\text{update}\,\mathbf{u}_i.
   $$

2. **Species-aware residual:** A fully connected tensor product couples the previous layer’s features with species one-hots:
   $$
   \mathbf{r}_i = \operatorname{FCTP}(\mathbf{H}_i^{\text{prev}}, \mathbf{e}_i).
   $$

3. **Gated nonlinearity:** Scalars and higher-order components are split, activated, and recombined using parity-aware SiLU/tanh gates (`nl_gate_kwargs`):
   $$
   \mathbf{H}_i^{\text{next}} = \text{Gate}\left(\mathbf{z}_i\right).
   $$

4. **Variance preserving blend:** The residual is added and re-normalised,
   $$
   \mathbf{H}_i^{\text{next}} \leftarrow \frac{1}{\sqrt{2}}\left(\mathbf{H}_i^{\text{next}} + \mathbf{r}_i\right),
   $$
   mirroring the $\mathcal{N}(0,1)$ preserving trick from SEGNN.

The first layer uses `ConditionedMessagePassingLayer`, which omits the species ResNet and contraction shortcut, matching the "non-residual bootstrap" described in the paper’s ablation studyciteturn8search9.

---

## 4. Layer stack and readout

- **Stack depth:** `num_mp_layers` (default 4) layers share the same irreps target `m_\ell`.
- **Compression:** A linear mixer reduces multiplicity by $4\times$ (`num_hidden_channels // 4`), mapping to an intermediate target field $\mathbf{t}_i$ with scalar and vector blocks.
- **Final projection:** Another mixer aligns $\mathbf{t}_i$ with the concatenated target irreps
  $$
  \mathbf{y}_i = \left[\Delta\mathbf{x}_i \,\|\, \Delta\mathbf{v}_i\right],
  $$
  where each block inherits the canonical $(1^-)$ irrep for vectors (`FIELD_IRREPS`).

---

## 5. Normalisation and physical constraints

### 5.1 Target normalisation

`NormalizationLayer` standardises displacement and velocity targets using dataset RMS/mean statistics cached as buffers:

$$
\tilde{\mathbf{y}} = \frac{\mathbf{y}-\boldsymbol{\mu}}{\boldsymbol{\sigma}}.
$$

This layer executes at the very start of the sequential model so all downstream losses operate on dimensionless quantities. Inference can optionally call the inverse transform.

### 5.2 Conservation layer

The terminal `ConservationLayer` enforces linear (and optionally angular) momentum while respecting normalisation constants (`disp_norm_const`, `vel_norm_const`).

1. **Linear momentum:** After unnormalising velocities, compute
   $$
   \mathbf{P}_\text{out} = \sum_i m_i \mathbf{v}_i^\text{pred},\qquad
   \Delta \mathbf{v} = \frac{\mathbf{P}_\text{in} - \mathbf{P}_\text{out}}{\sum_i m_i},
   $$
   and shift every velocity by $\Delta\mathbf{v}$.
2. **Angular momentum (optional):** For each graph/batch, evaluate inertia tensor $\mathbf{I}$ and solve
   $$
   \boldsymbol{\omega} = \mathbf{I}^{-1}\left(\mathbf{L}_\text{in}-\mathbf{L}_\text{out}\right),\qquad
   \Delta\mathbf{v}_i = \boldsymbol{\omega}\times(\mathbf{x}_i-\mathbf{x}_\text{COM}),
   $$
   adding it to the velocities. If the prescribed net momentum is zero, displacements are recentred to keep the centre of mass fixed.
3. **Renormalise:** Velocities and displacements are re-scaled by the stored RMS to remain in training units.

The layer also stores `net_lin_mom` and `net_ang_mom` for constrained rollouts.

---

## 6. Autoregressive rollout and thermostats

`models/trajcast/model/forecast.py` implements the self-feeding rollout (Algorithm 1 in [Thiemann et al., 2025]citeturn8search9):

1. **State update:** At each step, apply predicted $\Delta\mathbf{x}_i$ and $\Delta\mathbf{v}_i$ to positions and velocities, recompute edges, and normalise targets for the next pass.
2. **Momentum management:** Optional `ZeroMomentum` utilities zero linear/angular momentum after each step to mitigate drift.
3. **Thermostatting:** If requested, a canonical stochastic velocity rescaling (CSVR) thermostat (`CSVRThermostat`) rescales velocities by a random factor $\alpha$ drawn from the exact distribution in Bussi et al. (2007):
   $$
   \alpha^2 = c_1 + c_2\left(R_1^2 + \sum R_2^2\right) + 2R_1\sqrt{c_1 c_2},
   $$
   where $c_1=\exp(-\Delta t/\tau)$ and $c_2$ targets the desired kinetic energy. Degrees of freedom are adjusted for imposed momentum constraints.
4. **Velocity initialisation:** When starting from positions only, `init_velocity` samples velocities from uniform or Gaussian distributions, optionally removes net momenta, then rescales them to match the desired temperature.

---

## 7. Training-time tricks

- **Loss weighting:** `MultiobjectiveLoss` can combine MSE/MAE with cross-angle penalties and per-atom cosine similarity, all with learnable weights (`models.py` and `losses.py`).
- **Gradient clipping:** Configurable `max_grad_norm`.
- **Normalised residuals:** Every residual addition divides by $\sqrt{2}$ to stabilise signal magnitudes across depth.
- **Average degree cache:** The buffer `avg_num_neighbors` prevents noisy scaling when batch sizes vary and can be overridden per dataset.
- **Precision management:** Double precision is supported by toggling `precision` in the config, propagating to datasets (`torch.set_default_dtype`).
- **CuEquivariance integration:** Setting `o3_backend="cueq"` swaps e3nn tensor products/linears with their CUDA-accelerated counterparts via wrapper ops.

---

## 8. Efficient TrajCast variant

`EfficientTrajCastModel` shares the same encoding but:

- uses `edge_cutoff` from the dataset preprocessor,
- replaces deep residual blocks with a lighter stack (no explicit `layers` attribute),
- compresses directly to a pure vector irrep (`target \sim m\times 1^-$) before the final projection.

This matches the lightweight configuration discussed in Appendix B of the paper for large-batch molecular liquidsciteturn8search9.

---

## 9. Summary of key tensors

| Symbol | Description | Irrep block |
| --- | --- | --- |
| $\mathbf{e}_i$ | Species one-hot | $C_Z \times 0^+$ |
| $\boldsymbol{\phi}_r(r_{ij})$ | Radial Bessel basis | $K_r \times 0^+$ |
| $\boldsymbol{\phi}_v(\|\mathbf{v}_i\|, Z_i)$ | Velocity norm + species | $K_v \times 0^+$ |
| $\mathbf{Y}_{ij}^{(\ell)}$ | Edge spherical harmonics | $\bigoplus_{\ell\leq \ell_{\max}} \ell^{ (-1)^\ell }$ |
| $\mathbf{H}_i^{(k)}$ | Hidden node features layer $k$ | $\bigoplus_{\ell} m_\ell \times \ell^{\pm}$ |
| $\mathbf{y}_i$ | Normalised displacement & velocity target | $1^- \oplus 1^-$ |

---

## 10. References

- F. Thiemann *et al.*, *Force-Free Molecular Dynamics Through Autoregressive Equivariant Networks*, 2025. [arXiv:2503.23794](https://arxiv.org/abs/2503.23794).
- Implementation sources: `models/trajcast/model/models.py`, `models/trajcast/nn/_message_passing.py`, `models/trajcast/nn/_encoding.py`, `models/trajcast/nn/modules.py`, `models/trajcast/model/forecast.py`, `models/trajcast/model/forecast_tools`.

---

This document should equip SEGNN practitioners with the exact tensor algebra and engineering heuristics employed by TrajCast, bridging the formalism of the paper and the practical PyTorch implementation.
