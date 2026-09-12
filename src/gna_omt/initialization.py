"""Population initialization for GNA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class Population:
    thickness: torch.Tensor
    material: torch.Tensor
    thickness_mean: torch.Tensor
    thickness_range: torch.Tensor
    material_mean: torch.Tensor
    material_range: torch.Tensor
    layer_numbers: torch.Tensor


def _sample_material_sequence(rng: np.random.Generator, length: int, low: int, high: int) -> np.ndarray:
    values = [int(rng.integers(low, high + 1))]
    while len(values) < length:
        candidates = np.arange(low, high + 1)
        candidates = candidates[candidates != values[-1]]
        values.append(int(rng.choice(candidates)))
    return np.asarray(values, dtype=np.float32)


def initialize_population(
    population_size: int,
    layer_numbers: Sequence[int],
    max_layers: int,
    *,
    thickness_min_nm: float = 20.0,
    thickness_max_nm: float = 100.0,
    material_min: int = 0,
    material_max: int = 10,
    seed: int = 0,
    device: torch.device | str = "cpu",
    optimize_thickness: bool = True,
    optimize_material: bool = True,
) -> Population:
    """Uniformly distribute candidates over the admissible layer-number set.

    Thickness is represented in the normalized [0, 1] space used to train the
    OMT-mixer. Padding thicknesses and materials are represented by -0.25 and
    -1, respectively, matching the original research implementation for the
    20--100 nm range.
    """
    layer_numbers = tuple(int(n) for n in layer_numbers)
    if not layer_numbers:
        raise ValueError("layer_numbers must contain at least one value")
    if max(layer_numbers) > max_layers:
        raise ValueError("max_layers must be >= every admissible layer number")
    if population_size % len(layer_numbers) != 0:
        raise ValueError("population_size must be divisible by the number of layer-number choices")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    per_layer_count = population_size // len(layer_numbers)
    thickness_padding = (0.0 - thickness_min_nm) / (thickness_max_nm - thickness_min_nm)

    t_rows, m_rows, tm_rows, tr_rows, mm_rows, mr_rows, n_rows = [], [], [], [], [], [], []
    for n_layers in layer_numbers:
        for _ in range(per_layer_count):
            t = torch.rand(max_layers, dtype=torch.float32)
            m = torch.full((max_layers,), -1.0, dtype=torch.float32)
            tm = torch.full((max_layers,), thickness_padding, dtype=torch.float32)
            tr = torch.zeros(max_layers, dtype=torch.float32)
            mm = torch.full((max_layers,), -1.0, dtype=torch.float32)
            mr = torch.zeros(max_layers, dtype=torch.float32)

            t[:n_layers] = torch.rand(n_layers)
            t[n_layers:] = thickness_padding
            m[:n_layers] = torch.from_numpy(_sample_material_sequence(rng, n_layers, material_min, material_max))
            tm[:n_layers] = 0.5
            tr[:n_layers] = 1.0
            mm[:n_layers] = 0.5 * (material_min + material_max)
            mr[:n_layers] = float(material_max - material_min)

            t_rows.append(t); m_rows.append(m)
            tm_rows.append(tm); tr_rows.append(tr)
            mm_rows.append(mm); mr_rows.append(mr)
            n_rows.append(n_layers)

    thickness = torch.stack(t_rows).to(device)
    material = torch.stack(m_rows).to(device)
    if optimize_thickness:
        thickness.requires_grad_(True)
    if optimize_material:
        material.requires_grad_(True)

    return Population(
        thickness=thickness,
        material=material,
        thickness_mean=torch.stack(tm_rows).to(device),
        thickness_range=torch.stack(tr_rows).to(device),
        material_mean=torch.stack(mm_rows).to(device),
        material_range=torch.stack(mr_rows).to(device),
        layer_numbers=torch.tensor(n_rows, dtype=torch.long, device=device),
    )
