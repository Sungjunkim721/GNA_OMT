"""Example entry point for GNA inverse design.

This script intentionally does not assume a private local dataset/checkpoint
path. Supply a trained OMT-mixer checkpoint and a target spectrum before use.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gna_omt import LinearEmbedding, OMTMixer, initialize_population, material_normalization, run_gna


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True, help="CSV/TXT file containing a 301-point target spectrum")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "gna_result.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = LinearEmbedding(24, 400, 2).to(device)
    model = OMTMixer(24, 400, 301, 8, embedding, material_normalization).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    for p in model.parameters():
        p.requires_grad_(False)

    pop = initialize_population(3000, [4, 8, 12, 16, 20, 24], 24, seed=args.seed, device=device)
    target = np.loadtxt(args.target, delimiter=",").reshape(-1)
    result = run_gna(model, pop, target, num_iterations=1000, learning_rate=1e-2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        thickness=result.thickness.cpu().numpy(),
        material=result.material.cpu().numpy(),
        rmse=result.rmse,
        r2=result.r2,
        thickness_sensitivity=result.thickness_sensitivity,
        material_sensitivity=result.material_sensitivity,
        redundancy_sensitivity=result.redundancy_sensitivity,
        loss_history=np.asarray(result.loss_history),
    )


if __name__ == "__main__":
    main()
