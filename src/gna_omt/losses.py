"""Loss functions used by the Global Neural Adjoint (GNA) method.

The functions in this module mirror the equations in the revised manuscript.
Padding positions are identified by a non-positive range and are excluded where
appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GNALossTerms:
    design: torch.Tensor
    thickness_boundary: torch.Tensor
    material_boundary: torch.Tensor
    material_redundancy: torch.Tensor
    constraint: torch.Tensor
    total: torch.Tensor


def design_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and target spectra."""
    return torch.mean((prediction - target) ** 2)


def thickness_boundary_loss(
    thickness: torch.Tensor,
    mean: torch.Tensor,
    value_range: torch.Tensor,
) -> torch.Tensor:
    """Thickness boundary loss (Eq. 5 in the revised manuscript)."""
    return torch.mean(F.relu(torch.abs(thickness - mean) - 0.5 * value_range))


def material_boundary_loss(
    material: torch.Tensor,
    mean: torch.Tensor,
    value_range: torch.Tensor,
) -> torch.Tensor:
    """Material-index boundary loss averaged over candidates and valid layers.

    ``value_range == 0`` marks padding positions.  Averaging only over active
    entries makes the implementation explicit and independent of padding.
    """
    penalty = F.relu(torch.abs(material - mean) - 0.5 * value_range)
    valid = value_range > 0
    if not torch.any(valid):
        return penalty.new_zeros(())
    return penalty[valid].mean()


def material_redundancy_loss(
    material: torch.Tensor,
    min_index: float = 0.0,
    max_index: float = 10.0,
) -> torch.Tensor:
    """Adjacent-material redundancy regularization (Eq. 8).

    The rounded preceding-layer index is detached from the computational graph,
    so the gradient for each pair flows only to the subsequent layer.
    """
    if material.shape[1] < 2:
        return material.new_zeros(())

    previous = torch.round(material[:, :-1].detach())
    current = material[:, 1:]
    valid = (
        (current >= min_index)
        & (current <= max_index)
        & (material[:, :-1] >= min_index)
        & (material[:, :-1] <= max_index)
    )
    pair_penalty = F.relu(1.0 - torch.abs(current - previous))
    pair_penalty = pair_penalty * valid.to(pair_penalty.dtype)

    # Paper implementation: batch mean for each pair, followed by averaging
    # over all adjacent-layer positions.
    return pair_penalty.mean(dim=0).sum() / (material.shape[1] - 1)


def constraint_loss(
    thickness: torch.Tensor,
    material: torch.Tensor,
    desired_thickness: Optional[torch.Tensor] = None,
    desired_material: Optional[torch.Tensor] = None,
    thickness_mask: Optional[torch.Tensor] = None,
    material_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """User-defined structural constraint loss with independent layer masks.

    Masks may be one-dimensional ``[max_layers]`` or broadcastable to the
    design tensors.  A value of 1 selects a constrained component; 0 excludes
    it from the loss.
    """
    loss = thickness.new_zeros(())

    if desired_thickness is not None and thickness_mask is not None:
        t_mask = thickness_mask.to(device=thickness.device, dtype=thickness.dtype)
        t_target = desired_thickness.to(device=thickness.device, dtype=thickness.dtype)
        loss = loss + torch.mean(torch.sum(t_mask * (thickness - t_target) ** 2, dim=1))

    if desired_material is not None and material_mask is not None:
        m_mask = material_mask.to(device=material.device, dtype=material.dtype)
        m_target = desired_material.to(device=material.device, dtype=material.dtype)
        loss = loss + torch.mean(torch.sum(m_mask * (material - m_target) ** 2, dim=1))

    return loss


def gna_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thickness: torch.Tensor,
    material: torch.Tensor,
    thickness_mean: torch.Tensor,
    thickness_range: torch.Tensor,
    material_mean: torch.Tensor,
    material_range: torch.Tensor,
    *,
    weight_thickness: float = 1.0,
    weight_material: float = 1.0,
    weight_constraint: float = 0.0,
    desired_thickness: Optional[torch.Tensor] = None,
    desired_material: Optional[torch.Tensor] = None,
    thickness_mask: Optional[torch.Tensor] = None,
    material_mask: Optional[torch.Tensor] = None,
    material_min: float = 0.0,
    material_max: float = 10.0,
) -> GNALossTerms:
    """Compute all GNA loss terms and their weighted sum."""
    ld = design_loss(prediction, target)
    lt = thickness_boundary_loss(thickness, thickness_mean, thickness_range)
    lmb = material_boundary_loss(material, material_mean, material_range)
    lmr = material_redundancy_loss(material, material_min, material_max)
    lc = constraint_loss(
        thickness,
        material,
        desired_thickness,
        desired_material,
        thickness_mask,
        material_mask,
    )
    total = ld + weight_thickness * lt + weight_material * (lmb + lmr) + weight_constraint * lc
    return GNALossTerms(ld, lt, lmb, lmr, lc, total)
