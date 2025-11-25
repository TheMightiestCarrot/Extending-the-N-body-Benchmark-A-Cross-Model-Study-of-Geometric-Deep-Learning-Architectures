# Ponita / PΘNITA — Mathematical Blueprint
Blueprint that matches the ICLR‑2024 paper *Fast, Expressive SE(n) Equivariant Networks Through Weight Sharing in Position‑Orientation Space* (openreview: dPHLbUqGbr) **and** the code in `models/ponita`. Everything is written so the network can be rebuilt from math alone.

## 0. Notation & Objects
- Ambient Euclidean dimension: $n\in\{2,3\}$ (repo uses $n=3$).  
- Positions $p_i\in\mathbb R^n$.  
- Optional node scalars $s_i\in\mathbb R^{F_s}$; vectors $v_i\in\mathbb R^{F_v\times n}$.  
- Discrete orientation grid $O=\{o_m\}_{m=1}^N\subset S^{n-1}$ shared across all nodes. `PositionOrientationGraph` builds it with a uniform repulsion grid (`uniform_grid_s2/s1`).  
- Graph edges $E\subset V\times V$ (k‑NN or radius; supplied by dataloader). For each edge $(j\!\to\!i)$, relative position $r_{ij}=p_j-p_i$.
- Hidden width $C$ (called `hidden_dim`); number of layers $L$.

## 1. Invariant Edge Attributes (weight sharing key)
For a homogeneous space $X$ of $SE(n)$, the attribute $a_{ij}$ must bijectively label the equivalence class of point pairs $[x_i,x_j]$. Implemented cases:

### 1.1 Pure position space $\mathbb R^n$
$$
a_{ij}^{\text{pos}}=\lVert r_{ij}\rVert\in\mathbb R.
$$

### 1.2 Position–orientation fiber bundle $\mathbb R^3\times S^2$ (separable form used in code)
For each orientation sample $o\in O$:

$$
\alpha_{ij}(o)=o^\top r_{ij}\qquad\text{(parallel shift)}
$$
$$
\beta_{ij}(o)=\big\|\,r_{ij}-\alpha_{ij}(o)\,o\big\|\qquad\text{(orthogonal shift)}
$$
$$
\gamma(o,o')=o^\top o' \qquad\text{(relative angle, independent of }i,j\text{).}
$$
`SEnInvariantAttributes(separable=True)` returns:
- $attr = [\alpha,\beta]\in\mathbb R^{E\times N\times 2}$
- $fiber\_attr = [\gamma]\in\mathbb R^{N\times N\times 1}$
- scalar distances $dists=\|r_{ij}\|$.

### 1.3 Position‑orientation point cloud (when `lift_graph=True`)
Nodes already live in $(p,o)$ with $p\in\mathbb R^3, o\in S^2$. For edge $j\!\to\!i$:
$$
(\alpha,\beta,\gamma)_{ij}=\big(o_i^\top r_{ij},\ \lVert r_{ij}-o_i(o_i^\top r_{ij})\rVert,\ o_j^\top o_i\big).
$$
Shape `attr∈R^{E×3}`.

## 2. Lifting Inputs to Spherical Signals
Given grid $O$:
- **Scalars $\to$ sphere**: $f_i(o)=s_i$ (broadcast).  
- **Vectors $\to$ sphere**: $f_i(o)=v_i^\top o$ (dot with orientation).  
This yields feature tensor $f\in\mathbb R^{|V|\times N\times F_{\text{in}}}$. Implemented by `scalar_to_sphere`, `vec_to_sphere` inside `PositionOrientationGraph`.

## 3. Kernel Parameterisation
Edge attributes are converted to continuous kernels via:
1. **PolynomialFeatures(degree = d)**:  
   $ \text{poly}(x)= [x,\ x\otimes x,\ldots,x^{\otimes d}] $ flattened.  
2. **MLP**: `Linear(d0→C)` → GELU → `Linear(C→B)` → GELU. (`B=basis_dim`, defaults to `hidden_dim`.)  
   - Separate MLPs for spatial (`basis_fn`) and fiber/orientation (`fiber_basis_fn`) channels.
3. **Cut‑off window** (optional radius $r_{\max}$): envelope  
$$
w(r)=1-\tfrac{(p+1)(p+2)}{2}(r/r_{\max})^{p}+p(p+2)(r/r_{\max})^{p+1}-\tfrac{p(p+1)}{2}(r/r_{\max})^{p+2},
$$
zeroed for $r\!>\!r_{\max}$ (`PolynomialCutoff`, $p=6$).  
Final sampled bases:  
`kernel_basis = basis_fn(attr) * w(dists)` and `fiber_kernel_basis = fiber_basis_fn(fiber_attr)`.

## 4. Convolutions
All message passing uses PyG `MessagePassing(aggr="add", node_dim=0)`.

### 4.1 Spatial Conv on lifted fiber bundle (`FiberBundleConv`, separable depth‑wise)
Input $x\in\mathbb R^{|V|\times N\times C}$.
1) **Spatial step** (per orientation, per channel):
$$
m_{ij}(o,c) = k^{(1)}_{ij}(o,c)\, x_j(o,c),
\quad k^{(1)} = \text{Linear}( \text{kernel\_basis}_{ij}(o) ) .
$$
Aggregate over neighbors: $x^{(1)}_i(o,c)=\sum_{j\in\mathcal N(i)} m_{ij}(o,c)$.
2) **Orientation (fiber) mixing**:
$$
x^{(2)}_i(p,c)=\frac{1}{N}\sum_{o} x^{(1)}_i(o,c)\,k^{(2)}(o,p,c),
\quad k^{(2)}=\text{Linear}( \text{fiber\_kernel\_basis}(o,p) ).
$$
Depth‑wise means $c$ is unchanged; bias added afterwards.

### 4.2 Plain spatial Conv for point clouds (`Conv`)
Input $x\in\mathbb R^{|V|\times C}$.
$$
x_i^{\text{out}}(c)=\sum_{j\in\mathcal N(i)}\sum_{c'} k_{ij}(c,c')\,x_j(c'),
\quad k_{ij}=\text{Linear}(\text{kernel\_basis}_{ij}).
$$
With `groups=C` this reduces to per‑channel scaling $k_{ij}(c)\,x_j(c)$ (depth‑wise).

### 4.3 Initialization calibration
After the first forward pass in training, kernels are rescaled so that $\operatorname{std}_{\text{in}}$ matches $\operatorname{std}_{\text{out}}$ (see `callibrate` methods).

## 5. ConvNeXt Block (used in every layer)
Given channels $C$:
1. `conv` (one of 4.1 / 4.2)  
2. LayerNorm over channel dim  
3. MLP: `Linear(C→wC)` → GELU → `Linear(wC→C)` where $w=$`widening_factor`  
4. Optional layer scale $\lambda$ (default $10^{-6}$): $x \leftarrow \lambda x$  
5. Residual add if shape matches.

## 6. Network Architecture
```
Input f (scalars+vectors)
↓ x_embedder: Linear(F_in → C)
for ℓ=1…L:
    x ← ConvNeXtℓ(x, edge_index,
                   edge_attr=kernel_basis,
                   fiber_attr=fiber_kernel_basis,
                   batch)
    optional readoutℓ = Linear(C → F_out^scalar + F_out^vec)
Average all available readouts → r
Split r = (r_scalar, r_vec)
↓ Readout heads (below)
Outputs: scalar field / vector field
```

### 6.1 Scalar & Vector readout (fiber‑bundle mode)
- Scalars: `sphere_to_scalar` → $ \hat{s}_i = \frac{1}{N}\sum_{o} r_{\text{scalar},i}(o) $.  
- Vectors: `sphere_to_vec` → $ \hat{v}_i = \frac{1}{N}\sum_{o} r_{\text{vec},i}(o)\,o $.  
- Optional graph pooling: `global_add_pool` over batches when `task_level="graph"`.

### 6.2 Readout for lifted point‑cloud mode (`lift_graph=True`)
- Aggregate all oriented copies back to their base node using `scatter_mean` with `scatter_projection_index` (maps lifted node → base node).  
- Scalars: mean of $r_{\text{scalar}}$ over lifted nodes of each base; optional global sum pool.  
- Vectors: each lifted orientation contributes $r_{\text{vec}}(o)\,o$; mean over lifted nodes; optional global sum pool.

### 6.3 Multiple readouts
If `multiple_readouts=True`, every layer that defines a readout contributes; final prediction is the arithmetic mean of all layer readouts (stabilizes training).

## 7. Optional Data Transform: `RandomRotate`
During training/testing (configurable), a random SO(3) (or SO(2)) rotation is sampled; applied to selected tensors (`pos`, `vec`, targets) to enforce equivariance empirically.

## 8. Discrete Implementation vs. Paper Equations
- Continuous separable group convolution (paper Eq. 11–15):  
  $$
  k([(p,o),(p',o')]) = K^{(3)}\, k^{(2)}(o^\top o')\, k^{(1)}(o^\top\!(p'-p), \|o_\perp(p'-p)\|).
  $$
  Discretization replaces integrals by sums over neighbors $j$ and orientations $o,o'$ on grid $O$ (paper Eq. 5).  
- Code exactly mirrors this: spatial step (k¹) over neighbors, then spherical step (k²) over grid, then channel MLP (K³ is absorbed into the post‑ConvNeXt MLP + optional readout Linear).
- Attributes in §1 correspond to the bijective maps proven in Theorem 1 of the paper, ensuring universal equivariance (paper Corollary 1.1); separable form keeps efficiency.

## 9. Shapes Summary (fiber‑bundle default)
- `graph.x` after lifting: $[|V|, N, F_{\text{in}}]$.  
- `edge_attr` (`kernel_basis`): $[|E|, N, B]$.  
- `fiber_attr` basis: $[N, N, B]$.  
- Hidden state: $[|V|, N, C]$.  
- Readout logits: $[|V|, N, F_{\text{out}}^{\text{tot}}]$.

## 10. Minimal Recreation Recipe
1. Build edge_index (k‑NN or radius) and `rel_pos`.  
2. Lift to position‑orientation space (shared grid $O$); compute invariants (§1).  
3. Sample spatial kernels with polynomial MLP + cutoff; sample fiber kernels analogously.  
4. Run L ConvNeXt blocks (§5) using FiberBundleConv (§4.1).  
5. Linear readout(s) → split scalar/vector → sphere readout (§6.1) → (optional) graph sum.  
6. If using point‑cloud mode, insert lifting graph construction and point‑cloud invariants; use `Conv` instead of `FiberBundleConv`; aggregate back (§6.2).

This matches both the theoretical formulation and every implementation detail present in `models/ponita`.
