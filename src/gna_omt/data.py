"""Dataset helpers for OMT surrogate training."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ThinFilmDataset(Dataset):
    def __init__(self, thickness: np.ndarray, material: np.ndarray, spectrum: np.ndarray) -> None:
        if not (len(thickness) == len(material) == len(spectrum)):
            raise ValueError("Thickness, material, and spectrum arrays must have the same length.")
        self.thickness = thickness
        self.material = material
        self.spectrum = spectrum

    def __len__(self) -> int:
        return len(self.spectrum)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(self.thickness[index], dtype=torch.float32),
            torch.as_tensor(self.material[index], dtype=torch.int32),
            torch.as_tensor(self.spectrum[index], dtype=torch.float32),
        )


# Backward-compatible alias.
TFDataSet = ThinFilmDataset
