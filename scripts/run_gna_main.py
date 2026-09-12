"""Small GNA smoke test for Spyder / interactive use.

1. Put OMT_Mixer.pth in ``checkpoints/``.
2. Run this file directly.
3. For a quick test, keep POPULATION_SIZE and NUM_ITERATIONS small.
4. After confirming operation, use the paper settings if desired.
"""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gna_omt import (
    LinearEmbedding,
    OMTMixer,
    initialize_population,
    material_normalization,
    run_gna,
)

# =========================
# User settings
# =========================
CHECKPOINT_PATH = ROOT / "checkpoints" / "OMT_Mixer.pth"
TARGET_FILE = ROOT / "example_data" / "transmittance_test.csv"
TARGET_INDEX = 0  # 0=Test1, ..., 5=Test6

# Small values for a quick smoke test.
POPULATION_SIZE = 300
NUM_ITERATIONS = 100
LEARNING_RATE = 1e-2
SEED = 0

# Set to 3000 and 1000, respectively, for the paper-scale GNA run.
# POPULATION_SIZE = 3000
# NUM_ITERATIONS = 1000

LAYER_NUMBERS = [4, 8, 12, 16, 20, 24]
OUTPUT_PATH = ROOT / "results" / f"gna_Test{TARGET_INDEX + 1}_smoke_test.npz"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = OMTMixer(
        24, 400, 301, 8,
        LinearEmbedding(24, 400, 2),
        material_normalization,
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print("Checkpoint loaded successfully.")

    targets = np.loadtxt(TARGET_FILE, delimiter=",")
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    if not 0 <= TARGET_INDEX < len(targets):
        raise IndexError(f"TARGET_INDEX must be between 0 and {len(targets)-1}.")
    target = targets[TARGET_INDEX]
    print(f"Target: Test{TARGET_INDEX + 1}, shape={target.shape}")

    population = initialize_population(
        POPULATION_SIZE,
        LAYER_NUMBERS,
        24,
        seed=SEED,
        device=device,
    )

    result = run_gna(
        model,
        population,
        target,
        num_iterations=NUM_ITERATIONS,
        learning_rate=LEARNING_RATE,
        print_every=1,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        thickness=result.thickness.cpu().numpy(),
        material=result.material.cpu().numpy(),
        rmse=result.rmse,
        r2=result.r2,
        thickness_sensitivity=result.thickness_sensitivity,
        material_sensitivity=result.material_sensitivity,
        redundancy_sensitivity=result.redundancy_sensitivity,
        loss_history=np.asarray(result.loss_history),
    )

    print("\nGNA smoke test finished.")
    print(f"Retained candidates: {len(result.rmse)} / {POPULATION_SIZE}")
    if len(result.rmse):
        best = int(np.argmin(result.rmse))
        print(f"Best RMSE: {result.rmse[best]:.6f}")
        print(f"Best R2:   {result.r2[best]:.6f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
