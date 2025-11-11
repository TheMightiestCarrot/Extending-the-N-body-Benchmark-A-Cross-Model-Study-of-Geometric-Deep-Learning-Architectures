import torch
from types import SimpleNamespace

from models.PaiNN.PaiNN import PaiNN


def make_complete_graph(n: int):
    i = torch.arange(n)
    row = i.repeat_interleave(n - 1)
    col = torch.cat([torch.cat([torch.arange(j), torch.arange(j + 1, n)]) for j in range(n)])
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def main():
    torch.manual_seed(0)
    N = 5
    pos = torch.randn(N, 3)
    vel = torch.randn(N, 3)
    mass = torch.rand(N, 1)
    edge_index = make_complete_graph(N)

    model = PaiNN(
        hidden_features=32,
        num_layers=2,
        num_rbf=16,
        cutoff=10.0,
        targets=("pos_dt", "vel"),
        use_velocity_input=True,
        include_velocity_norm=True,
    )

    graph = SimpleNamespace(pos=pos, vel=vel, mass=mass, edge_index=edge_index)
    out = model(graph)
    print("OK -- out shape:", tuple(out.shape))


if __name__ == "__main__":
    main()

