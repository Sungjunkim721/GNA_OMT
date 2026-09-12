"""Clean implementation of the Global Neural Adjoint optimization loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import r2_score

from .initialization import Population
from .losses import gna_loss


@dataclass
class GNAResult:
    thickness: torch.Tensor
    material: torch.Tensor
    thickness_sensitivity: np.ndarray
    material_sensitivity: np.ndarray
    redundancy_sensitivity: np.ndarray
    loss_history: list[float]
    rmse: np.ndarray
    r2: np.ndarray
    kept_indices: np.ndarray


def _grad_or_zero(loss: torch.Tensor, variable: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(loss, variable, allow_unused=True, retain_graph=True)[0]
    return torch.zeros_like(variable) if grad is None else grad


def _filter_candidates(
    thickness: torch.Tensor,
    rounded_material: torch.Tensor,
    layer_numbers: torch.Tensor,
    material_min: int,
    material_max: int,
) -> torch.Tensor:
    keep = torch.ones(thickness.shape[0], dtype=torch.bool, device=thickness.device)
    for row, n_layers in enumerate(layer_numbers.tolist()):
        t = thickness[row, :n_layers]
        m = rounded_material[row, :n_layers]
        if torch.any(t < 0) or torch.any(t > 1) or torch.any(m < material_min) or torch.any(m > material_max):
            keep[row] = False
            continue
        if n_layers > 1 and torch.any(m[:-1] == m[1:]):
            keep[row] = False
    return keep


def run_gna(
    forward_model: torch.nn.Module,
    population: Population,
    target_spectrum: np.ndarray | torch.Tensor,
    *,
    num_iterations: int = 1000,
    learning_rate: float = 1e-2,
    weight_thickness: float = 1.0,
    weight_material: float = 1.0,
    weight_constraint: float = 0.0,
    desired_thickness: Optional[torch.Tensor] = None,
    desired_material: Optional[torch.Tensor] = None,
    thickness_mask: Optional[torch.Tensor] = None,
    material_mask: Optional[torch.Tensor] = None,
    material_min: int = 0,
    material_max: int = 10,
    print_every: int = 200,
) -> GNAResult:
    """Optimize thickness and material indices for a fixed candidate population.

    ``num_iterations`` is the number of optimizer updates. If an initial
    (iteration-0) history point is needed for plotting, record it separately
    rather than setting this value to 1001.
    """
    thickness = population.thickness
    material = population.material
    device = thickness.device

    target = torch.as_tensor(target_spectrum, dtype=torch.float32, device=device)
    target = target.reshape(1, -1).expand(thickness.shape[0], -1)

    variables = []
    if thickness.requires_grad:
        variables.append(thickness)
    if material.requires_grad:
        variables.append(material)
    if not variables:
        raise ValueError("At least one of thickness or material must require gradients")

    optimizer = torch.optim.Adam(variables, lr=learning_rate)
    forward_model.eval()

    s_t = torch.zeros_like(thickness)
    s_m = torch.zeros_like(material)
    s_mr = torch.zeros_like(material)
    history: list[float] = []

    for step in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_model(thickness, material)
        terms = gna_loss(
            prediction,
            target,
            thickness,
            material,
            population.thickness_mean,
            population.thickness_range,
            population.material_mean,
            population.material_range,
            weight_thickness=weight_thickness,
            weight_material=weight_material,
            weight_constraint=weight_constraint,
            desired_thickness=desired_thickness,
            desired_material=desired_material,
            thickness_mask=thickness_mask,
            material_mask=material_mask,
            material_min=material_min,
            material_max=material_max,
        )

        # Sensitivity profiles are defined with respect to L_d only.
        if thickness.requires_grad:
            g_design_t = _grad_or_zero(terms.design, thickness)
            s_t += torch.abs(g_design_t.detach())
        if material.requires_grad:
            g_design_m = _grad_or_zero(terms.design, material)
            s_m += torch.abs(g_design_m.detach())
            g_red_m = _grad_or_zero(weight_material * terms.material_redundancy, material)
            s_mr += torch.abs(g_red_m.detach())

        terms.total.backward()
        optimizer.step()
        history.append(float(terms.design.detach().cpu()))

        if print_every and step % print_every == 0:
            print(
                f"step={step:4d} Ld={terms.design.item():.6f} "
                f"Lt={terms.thickness_boundary.item():.6f} "
                f"Lm={(terms.material_boundary + terms.material_redundancy).item():.6f} "
                f"Lc={terms.constraint.item():.6f}"
            )

    final_t = thickness.detach()
    final_m = material.detach().round()
    keep = _filter_candidates(final_t, final_m, population.layer_numbers, material_min, material_max)
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    final_t = final_t[keep]
    final_m = final_m[keep]

    with torch.no_grad():
        pred = forward_model(final_t, final_m).cpu().numpy()
    target_np = np.asarray(target_spectrum, dtype=np.float32).reshape(-1)
    rmse = np.sqrt(np.mean((pred - target_np[None, :]) ** 2, axis=1))
    r2 = np.asarray([r2_score(target_np, row) for row in pred])

    return GNAResult(
        thickness=final_t,
        material=final_m,
        thickness_sensitivity=s_t[keep].cpu().numpy(),
        material_sensitivity=s_m[keep].cpu().numpy(),
        redundancy_sensitivity=s_mr[keep].cpu().numpy(),
        loss_history=history,
        rmse=rmse,
        r2=r2,
        kept_indices=kept_indices.cpu().numpy(),
    )
