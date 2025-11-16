import torch


def _build_fully_connected_edge_index(batch_size, num_nodes, device):
    """Return the directed fully connected edge index for the whole batch."""

    total_nodes = batch_size * num_nodes

    # Precompute a single-graph pattern (all nodes except self-loops).
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=device)
    row_single, col_single = torch.nonzero(mask, as_tuple=True)
    edges_per_graph = row_single.numel()

    # Broadcast the single-graph pattern across the batch with offsets.
    offsets = torch.arange(0, total_nodes, num_nodes, device=device)
    offsets = offsets.repeat_interleave(edges_per_graph)
    row = row_single.repeat(batch_size) + offsets
    col = col_single.repeat(batch_size) + offsets

    return torch.stack([row, col], dim=0)


def build_graph_with_knn(loc, batch_size, num_nodes, device, num_neighbors):
    num_nodes = int(num_nodes)
    if num_neighbors is not None:
        num_neighbors = int(num_neighbors)
    else:
        num_neighbors = num_nodes - 1

    if num_neighbors >= num_nodes:
        raise ValueError(
            "Graph cannot have more neighbors than there are nodes in simulation - 1"
        )

    # When every node connects to all other nodes, skip the KNN computation and
    # return the fully connected pattern directly. This preserves the exact
    # topology (all ordered pairs except self) while avoiding the expensive
    # pairwise distance + top-k path that yielded the same multiset of edges.
    if num_neighbors == num_nodes - 1:
        return _build_fully_connected_edge_index(batch_size, num_nodes, device)

    # Precompute total number of nodes in all batches
    total_nodes = batch_size * num_nodes

    # Compute pairwise distances for all batches at once
    loc_reshaped = loc.view(
        batch_size, num_nodes, -1
    )  # Shape: (batch_size, num_nodes, num_features)

    # Efficiently compute pairwise distances for all batches
    dist_matrix = torch.cdist(
        loc_reshaped, loc_reshaped
    )  # Shape: (batch_size, num_nodes, num_nodes)

    # Get the k-nearest neighbors for each node (excluding self-loops)
    # Ensure k doesn't exceed the number of nodes
    k = min(num_neighbors + 1, num_nodes)
    knn_indices = torch.topk(dist_matrix, k=k, largest=False).indices[
        :, :, 1:
    ]  # Shape: (batch_size, num_nodes, num_neighbors)

    # Create an edge index matrix for all batches
    row_indices = (
        torch.arange(num_nodes, device=device)
        .view(1, -1, 1)
        .expand(batch_size, num_nodes, num_neighbors)
    )

    # Add batch offsets to row_indices and knn_indices
    batch_offsets = torch.arange(0, total_nodes, num_nodes, device=device).view(
        batch_size, 1, 1
    )
    row_indices = row_indices + batch_offsets
    knn_indices = knn_indices + batch_offsets

    # Stack row and column indices into a single tensor for the edge index
    edge_index = torch.stack(
        [row_indices.flatten(), knn_indices.flatten()], dim=0
    )  # Shape: (2, batch_size * num_nodes * num_neighbors)
    return edge_index.to(device)
