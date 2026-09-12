"""Training entry point for the OMT-mixer.

Expected .npz files contain arrays named ``thickness``, ``material``, and
``spectrum``. Thickness values should already be normalized to [0, 1], with
padding values < 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gna_omt import LinearEmbedding, OMTMixer, ThinFilmDataset, material_normalization
from gna_omt.training import train_forward_model


def load_dataset(path: Path) -> ThinFilmDataset:
    data = np.load(path)
    return ThinFilmDataset(data["thickness"], data["material"], data["spectrum"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "OMT_Mixer.pth")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(load_dataset(args.train), batch_size=500, shuffle=True)
    val_loader = DataLoader(load_dataset(args.val), batch_size=125, shuffle=False)

    embedding = LinearEmbedding(24, 400, 2).to(device)
    model = OMTMixer(24, 400, 301, 8, embedding, material_normalization).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result = train_forward_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        criterion=torch.nn.MSELoss(),
        device=device,
        epochs=args.epochs,
        patience=1000,
        checkpoint_path=args.checkpoint,
    )
    print(f"training time: {result.elapsed_seconds:.3f} s; best val: {result.best_validation_loss:.6g}")


if __name__ == "__main__":
    main()
