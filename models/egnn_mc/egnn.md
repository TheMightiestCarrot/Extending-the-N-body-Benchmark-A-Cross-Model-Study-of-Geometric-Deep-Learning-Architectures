# N-Body EGNN\_vel Configuration

This note documents how the charged $N$-body experiment in `main_nbody.py` constructs its graph inputs and how the multi-channel EGNN\_vel model processes them.

## Data Snapshot

Each sample in `NBodyDataset` is a charged-particle system with $N = 5$ bodies simulated in 3D. For the default `nbody_small` split the dataset returns:

- Current positions $x_i \in \mathbb{R}^3$ at frame $t_0 = 30$.
- Current velocities $v_i \in \mathbb{R}^3$ at frame $t_0$.
- Target positions $x_i^{\text{target}} \in \mathbb{R}^3$ at frame $t_1 = 40$ ($\Delta t = 10$ simulation steps later).
- Static scalar charges $q_i \in \{-1,0,1\}$. The dataset also precomputes $q_i q_j$ for every ordered edge to use as an edge attribute.

For batching, the code builds a complete directed graph:
$$
V = \{1,\dots,N\}, \qquad
E = \{(i,j)\mid i \neq j\},
$$
and tiles it for every graph in the mini-batch.

## Feature Construction

### Node scalar feature ($h$ input)

Before entering the network each node receives a single scalar feature derived from its speed:
$$
h_i^{(0)} = \|v_i\|_2.
$$
This scalar is passed through a learnable linear embedding layer to produce the hidden node state for layer $0$.

### Node coordinate tensor ($X$ input)

Both the coordinate tensor and its multi-channel extensions start from the raw position vector:
$$
X_i^{(0)} = x_i \in \mathbb{R}^{3 \times 1}.
$$
When `--num_vectors = K > 1`, the first layer expands this to $\mathbb{R}^{3 \times K}$; subsequent layers maintain $K$ coordinate channels unless explicitly reduced in the last layer.

### Node vector feature (velocity injection)

The full velocity vector is supplied separately to every EGNN\_vel layer:
$$
v_i \in \mathbb{R}^3,
$$
and is used to learn an additive coordinate update (see below). It is *not* appended as an extra coordinate channel.

### Edge attributes

The dataset stores one static scalar per ordered pair $(i,j)$:
$$
a^{\text{charge}}_{ij} = q_i q_j.
$$
During training the script augments this with the squared Euclidean distance computed on the fly:
$$
a^{\text{dist}}_{ij} = \|x_i - x_j\|_2^2.
$$
The final edge-feature vector passed to the model is
$$
a_{ij} = \big[a^{\text{charge}}_{ij},\; a^{\text{dist}}_{ij}\big] \in \mathbb{R}^2.
$$

## EGNN\_vel Layer Mathematics

Each layer $\ell$ keeps three learnable modules: an edge MLP $\phi_e^{(\ell)}$, a node MLP $\phi_h^{(\ell)}$, and a coordinate MLP $\phi_x^{(\ell)}$. The number of coordinate channels entering layer $\ell$ is denoted $K_\ell$ and exiting layer $\ell$ is $K_{\ell+1}$ (equal to `num_vectors` except for the first and last layers).

### 1. Radial term

For every edge $(i,j)$ the layer recomputes the (possibly multi-channel) squared distance:
$$
r_{ij}^{(\ell)} = \sum_{c=1}^{K_\ell} \left\|X_{i,c}^{(\ell)} - X_{j,c}^{(\ell)}\right\|_2^2 \in \mathbb{R}^{K_\ell},
$$
where $X_{i,c}^{(\ell)} \in \mathbb{R}^3$ is the $c$-th coordinate channel.

### 2. Edge message

Concatenate source/target node states, the radial features, and the edge attributes, then pass them through the edge network:
$$
m_{ij}^{(\ell)} = \phi_e^{(\ell)}\left(\big[h_i^{(\ell)},\; h_j^{(\ell)},\; r_{ij}^{(\ell)},\; a_{ij}\big]\right) \in \mathbb{R}^{H_e},
$$
where $H_e$ is the hidden edge width. Optional attention gates (disabled in the default config) would scale this output.

### 3. Node update

Aggregate incoming messages by summation, concatenate with the current node embedding, process via the node MLP, and apply a residual connection:
$$
\tilde{h}_i^{(\ell)} = \sum_{j:(i,j)\in E} m_{ij}^{(\ell)},\qquad
h_i^{(\ell+1)} = h_i^{(\ell)} + \phi_h^{(\ell)}\left(\big[h_i^{(\ell)},\; \tilde{h}_i^{(\ell)}\big]\right).
$$

### 4. Coordinate update (multi-channel)

The edge features also parameterize a learnable linear map that mixes coordinate differences across channels:
$$
\begin{aligned}
W_{ij}^{(\ell)} &= \mathrm{reshape}\left(\phi_x^{(\ell)}\big(m_{ij}^{(\ell)}\big),\, K_\ell \times K_{\ell+1}\right), \\
\Delta X_{ij}^{(\ell)} &= W_{ij}^{(\ell)\top}\left(X_i^{(\ell)} - X_j^{(\ell)}\right) \in \mathbb{R}^{3 \times K_{\ell+1}}.
\end{aligned}
$$
These deltas are averaged per receiver node using the mean aggregator implemented in `unsorted_segment_mean`:
$$
\Delta X_i^{(\ell)} = \frac{1}{|\mathcal{N}(i)|} \sum_{j:(i,j)\in E} \Delta X_{ij}^{(\ell)}.
$$
If $\ell$ is not the final layer the coordinates are updated additively:
$$
X_i^{(\ell+1)} = X_i^{(\ell)} + \Delta X_i^{(\ell)}.
$$
On the last layer the existing channels are first averaged to remove the channel dimension:
$$
\bar{X}_i^{(\ell)} = \frac{1}{K_\ell} \sum_{c=1}^{K_\ell} X_{i,c}^{(\ell)}.
$$
This ensures the output has a single 3D vector per node.

### 5. Velocity injection

Every layer also forms a velocity mixing matrix from the current node embedding:
$$
B_i^{(\ell)} = \mathrm{reshape}\left(\psi^{(\ell)}(h_i^{(\ell)}),\, K_\ell \times K_{\ell+1}\right),
$$
where $\psi^{(\ell)}$ is the `coord_mlp_vel` MLP. The layer adds a learned projection of the raw velocity to the coordinates:
$$
X_i^{(\ell+1)} = X_i^{(\ell+1)} + B_i^{(\ell)\top} v_i.
$$
If `--update_vel=True` the model would also propagate the modified velocity tensor returned by the layer, otherwise $v_i$ remains the original input at every depth.

## Output and Loss

After the final layer the predicted coordinates are flattened back to $\mathbb{R}^3$ per node:
$$
\hat{x}_i = X_i^{(L)} \in \mathbb{R}^3.
$$
Training minimizes the mean-squared error to the target positions:
$$
\mathcal{L} = \frac{1}{B N} \sum_{b=1}^B \sum_{i=1}^N \left\| \hat{x}_{b,i} - x_{b,i}^{\text{target}} \right\|_2^2.
$$

## Feature Summary

| Component | Symbol | Dimension | Construction |
|-----------|--------|-----------|--------------|
| Node scalar | $h_i^{(0)}$ | $\mathbb{R}$ | Speed $\|v_i\|_2$ |
| Node coordinates | $X_i^{(0)}$ | $\mathbb{R}^{3 \times 1}$ | Raw position $x_i$ |
| Node vector | $v_i$ | $\mathbb{R}^3$ | Raw velocity |
| Edge attr. (charge) | $a^{\text{charge}}_{ij}$ | $\mathbb{R}$ | $q_i q_j$ |
| Edge attr. (distance) | $a^{\text{dist}}_{ij}$ | $\mathbb{R}$ | $\|x_i - x_j\|_2^2$ |
| Target | $x_i^{\text{target}}$ | $\mathbb{R}^3$ | Position at frame $t_1$ |

This decomposition captures every signal the N-body experiment feeds into EGNN\_vel and the equations that govern its message passing and coordinate evolution. Adjusting `--num_vectors` simply changes $K_\ell$, letting the network carry multiple learned coordinate channels without altering the feature definitions above.

## EGNNMultiChannel Implementation (This Repository)

`models/egnn_mc/egnn_mc.py` adapts the baseline EGNN\_vel to serve multi-target predictions for the gravity and molecular dynamics pipelines in this repository. The core block structure stays close to the reference, but the inputs, message features, and output heads are tailored to the datasets used here.

### Inputs and Graph Construction

- **Node features**: The `EgnnMcNBodyDataLoader` supplies two scalars per node, `[ \|v_i\|_2, m_i ]`, combining speed with particle mass. The baseline model embeds only the speed. (`node_input_dim=2` by default.)
- **Edge attributes**: Instead of charge products and squared distance, the loader concatenates four signals: $m_i m_j$, the projections of $v_i$ and $v_j$ onto the edge direction, and $\|x_i - x_j\|_2^2`. (`edge_attr_dim=4`.)
- **Connectivity**: Edges are created with a batched k-NN builder. With the default `num_neighbors = N-1`, this collapses to the same complete directed graph as the baseline, but it allows sparse neighborhoods when desired.

### Message Block Differences

- **Single coordinate channel**: Every `_EGNNMessageBlock` is instantiated with `num_vectors_in = num_vectors_out = 1`. The multi-channel expansion/aggregation described for EGNN\_vel is disabled, so coordinates stay in $\mathbb{R}^{3 \times 1}$ throughout.
- **Edge feature composition**: The edge MLP receives the concatenation `[h_i, h_j, r_{ij}, a_{ij}]` with the new four-dimensional attribute vector. A small-gain Xavier initialization and optional `tanh` limiter bound the learned coordinate mixing weights, and the update is scaled by `coords_weight`.
- **Clamped coordinate deltas**: The coordinate updates are clamped to $[-100, 100]$ before aggregation to avoid numerical blow-ups, a safeguard absent from the baseline description.
- **Optional normalisation**: The implementation can normalise coordinate differences (`norm_diff=True`) before message passing, another deviation from the base configuration.

### Velocity Handling

- Each layer still forms a velocity mixing matrix from the node state, but velocities are never rewritten—`vel_state` remains the original `graph.vel`. The baseline optionally propagated updated velocities when `--update_vel=True`.
- Because there is only a single coordinate channel, the velocity projection reduces to a $3 \times 3$ map instead of a channel-mixing tensor.

### Output Heads and Targets

- After the final layer the model computes the displacement $x_i^{(L)} - x_i^{(0)}$ (`pos_dt`) and concatenates it with the preserved velocities.
- A dedicated `_VectorHead` is created for each requested target name (e.g., `"pos_dt"`, `"vel"`). Each head ingests `[h_i^{(L)}, \text{pos\_dt}_i, v_i]` and outputs a 3D vector, so the network can jointly predict multiple vector quantities.
- The two back-to-back linear layers before the first activation in `_VectorHead` effectively implement a learned affine projection distinct from the baseline’s direct coordinate output.

### Summary of Differences vs. Basic EGNN\_vel

1. **Feature encoding**: adds particle mass to node features and augments edge attributes with velocity projections and mass products instead of charge-derived terms.
2. **Coordinate channels**: fixes the model to a single coordinate channel, removing the baseline’s configurable multi-channel expansion and final averaging.
3. **Stability tweaks**: introduces coordinate update clamping, optional difference normalisation, and tunable `coords_weight`.
4. **Velocity path**: keeps the original velocities untouched while still injecting them into coordinate updates, whereas the baseline can optionally update the velocity tensor.
5. **Prediction head**: replaces direct position outputs with per-target vector heads operating on `[h, \text{pos\_dt}, v]`, enabling simultaneous prediction of displacements, velocities, or other 3D targets.

These adaptations make `EGNNMultiChannel` better suited to the repository’s multi-target N-body and MD tasks while preserving the equivariant message-passing backbone of the original EGNN\_vel.

