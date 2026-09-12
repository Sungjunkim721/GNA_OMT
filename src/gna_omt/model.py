"""OMT-mixer forward surrogate.

The module/parameter names intentionally match the original research
implementation so that existing ``OMT_Mixer.pth`` state dictionaries can be
loaded without key remapping.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class LinearEmbedding(nn.Module):
    def __init__(self, d_i: int, d_embed: int, d_channel: int = 2) -> None:
        super().__init__()
        self.d_i = d_i
        self.d_embed = d_embed
        self.d_channel = d_channel
        self.projection = nn.Linear(d_channel, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def material_normalization(material: torch.Tensor, minimum: float = 0.0, maximum: float = 10.0) -> torch.Tensor:
    """Min-max scale material indices and append the feature dimension."""
    return ((material - minimum) / (maximum - minimum)).unsqueeze(2)


class OMTMixer(nn.Module):
    """OMT-mixer architecture used as the differentiable forward surrogate."""

    def __init__(
        self,
        d_i: int,
        d_e: int,
        d_o: int,
        n_blocks: int,
        embedding: nn.Module,
        mat_embedding=material_normalization,
    ) -> None:
        super().__init__()
        # Keep original attribute names for checkpoint compatibility.
        self.omt_mixers = nn.ModuleList()
        self.embedding_mixers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()

        omt_mixer = nn.Sequential(
            Transpose(1, 2),
            nn.Linear(d_i, 2 * d_i, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_i, d_i, bias=False),
            Transpose(1, 2),
        )
        embedding_mixer = nn.Sequential(
            nn.Linear(d_e, 2 * d_e, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_e, d_e, bias=False),
        )
        norm = nn.LayerNorm(d_e, bias=False)

        for _ in range(n_blocks):
            self.omt_mixers.append(copy.deepcopy(omt_mixer))
            self.embedding_mixers.append(copy.deepcopy(embedding_mixer))
            self.norms1.append(copy.deepcopy(norm))
            self.norms2.append(copy.deepcopy(norm))

        self.norm = nn.LayerNorm(d_e, bias=False)
        self.regressor = nn.Linear(d_e, d_o)
        self.embedding = embedding
        self.mat_embedding = mat_embedding

    @staticmethod
    def masking(out: torch.Tensor, pad_idx: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        out[pad_idx[0], pad_idx[1], :] = 0
        return out

    def forward(self, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        pad_idx = torch.where(t < 0)
        m = self.mat_embedding(m)
        out = torch.dstack([t.unsqueeze(2), m])
        out = self.embedding(out)
        out = self.masking(out, pad_idx)

        for omt_mixer, embedding_mixer, norm1, norm2 in zip(
            self.omt_mixers,
            self.embedding_mixers,
            self.norms1,
            self.norms2,
        ):
            out = omt_mixer(norm1(out)) + out
            out = self.masking(out, pad_idx)
            out = embedding_mixer(norm2(out)) + out

        out = self.norm(out)
        return self.regressor(torch.mean(out, dim=1))


# Original class name retained for existing scripts/checkpoints.
OMT_Mixer = OMTMixer
