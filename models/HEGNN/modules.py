# HEGNN with dual equivariant vector heads (pos & vel)
# ----------------------------------------------------
# Requirements:
#   torch, torch_scatter, e3nn
#
# Notes:
# - Uses only invariant scalars to gate vector differences, so updates are O(3)-equivariant
#   and translation-invariant.
# - Replaces the old velocity "nudge" with a proper velocity head.
# - Forward returns {"pos": node_pos, "vel": node_vel} for rollouts.

import math
from functools import partial
from typing import Dict, Optional

import torch
from torch import nn
from torch_scatter import scatter

import e3nn


class BaseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: nn.Module = nn.SiLU(),
        residual: bool = False,
        last_act: bool = False,
    ):
        super().__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim, "Residual MLP must keep dimensionality."
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation if last_act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x) if self.residual else self.mlp(x)


class SH_Msg(nn.Module):
    """
    Compute per-degree inner products (scalarization) between steerable features
    on edge endpoints. For each degree l, sum across multiplicities/channels.
    Output shape: (E, lmax+1) of invariant scalars.
    """

    def __init__(self, sh_irreps: e3nn.o3.Irreps):
        super().__init__()
        self.sh_irreps = sh_irreps

    def forward(self, edge_index: torch.Tensor, node_sh: torch.Tensor) -> torch.Tensor:
        # node_sh: (N, sum_l mul_l * dim_l) must match irreps.dim
        assert node_sh.size(1) == self.sh_irreps.dim
        row, col = edge_index  # messages j->i with i=row, j=col
        temp = node_sh[row] * node_sh[col]  # elementwise product per channel

        # For each degree l, sum channels of that degree to get one scalar per l
        idx = 0
        ip = torch.zeros(
            (temp.size(0), self.sh_irreps.lmax + 1),
            device=node_sh.device,
            dtype=node_sh.dtype,
        )
        for mul, ir in self.sh_irreps:
            ip[:, ir.l] = torch.sum(temp[:, idx : idx + ir.dim], dim=-1).to(
                node_sh.dtype
            )
            idx += ir.dim

        # Optional normalization could be added here if you see NaNs during training.
        return ip  # (E, lmax+1)


class SmoothBesselBasis(nn.Module):
    """Finite-support radial basis with sine/Bessel functions and smooth polynomial cutoff."""

    def __init__(self, num_basis: int, cutoff: float, envelope_power: int = 5):
        super().__init__()
        if num_basis <= 0:
            raise ValueError("num_basis must be positive.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if envelope_power < 3:
            raise ValueError("envelope_power must be >= 3 for smooth cutoff.")
        self.cutoff = cutoff
        self.register_buffer(
            "bands", torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype())
        )
        self.envelope_power = envelope_power

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return radial basis evaluated at distances (E,).

        Uses sin(k * pi * r / R_c) / r multiplied by a C^2 polynomial envelope that
        smoothly decays to zero at r = R_c.
        """
        r = distances.unsqueeze(-1)  # (E, 1)
        rc = self.cutoff
        x = (r / rc).clamp(min=0.0)

        # Smooth polynomial envelope of order 5 (C2 continuous) by default.
        # c(x) = 1 - 6x^5 + 15x^4 - 10x^3 for x in [0, 1], else 0.
        # Generalises to arbitrary envelope_power by matching coefficients.
        power = float(self.envelope_power)
        x_clip = torch.clamp(x, max=1.0)
        if power == 5:
            envelope = 1 - 6 * x_clip**5 + 15 * x_clip**4 - 10 * x_clip**3
        else:
            # Construct coefficients ensuring c(0)=1 and c(1)=0 with zero first/second derivatives.
            # Using general form (1 - x)^p * (1 + px + p(p-1)/2 x^2)
            envelope = (1 - x_clip) ** power * (
                1 + power * x_clip + 0.5 * power * (power - 1) * x_clip**2
            )
        mask = (distances <= rc).unsqueeze(-1)
        envelope = envelope * mask

        # Avoid division by zero at r = 0
        denom = torch.where(r > 0, r, torch.ones_like(r))
        bands = self.bands.to(r)
        radial = torch.sin(bands * math.pi * r / rc) / denom
        radial = radial * envelope
        return radial


class SH_INIT(nn.Module):
    """
    Initialize high-degree steerable node features by aggregating spherical harmonics
    of normalized relative positions with learned per-edge weights.
    """

    def __init__(
        self,
        radial_dim: int,
        hidden_dim: int,
        max_ell: int,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            self.sh_irreps, normalize=True, normalization="norm"
        )

        # CG: (L) x (scalar) -> (L)
        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, "1x0e", self.sh_irreps, shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        # msg uses [node_feat[row], node_feat[col], radial]
        self.mlp_sh = MLP(
            input_dim=2 * hidden_dim + radial_dim,
            output_dim=self.sh_coff.weight_numel,
            last_act=True,
        )

    def forward(
        self,
        node_feat: torch.Tensor,  # (N, H)
        diff_pos: torch.Tensor,  # (E, 3)
        edge_index: torch.Tensor,  # (2, E)
        radial: torch.Tensor,  # (E, R)
        precomputed_sh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        row, col = edge_index

        msg = torch.cat([node_feat[row], node_feat[col], radial], dim=-1)
        msg = self.mlp_sh(msg)  # (E, W)

        if precomputed_sh is None:
            # Spherical harmonics on directions only (no gradients through SH basis)
            diff_sh = self.spherical_harmonics(diff_pos).to(diff_pos.dtype).detach()
        else:
            diff_sh = precomputed_sh.to(msg.dtype).detach()

        one = torch.ones(
            (diff_sh.size(0), 1), device=diff_sh.device, dtype=diff_sh.dtype
        ).detach()
        diff_sh = self.sh_coff(diff_sh, one, msg)  # (E, irreps.dim)

        # Mean aggregate to nodes
        node_sh = scatter(
            diff_sh, index=row, dim=0, dim_size=node_feat.size(0), reduce="mean"
        )  # (N, irreps.dim)
        return node_sh
