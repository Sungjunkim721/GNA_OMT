"""Quick checkpoint sanity check.

Designed to be run directly from Spyder or a terminal.
Edit CHECKPOINT_PATH only if the checkpoint is stored elsewhere.
"""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gna_omt import LinearEmbedding, OMTMixer, material_normalization

# =========================
# User settings
# =========================
CHECKPOINT_PATH = ROOT / "checkpoints" / "OMT_Mixer.pth"
TEST_THICKNESS_PATH = ROOT / "example_data" / "thicknesses_test.csv"
TEST_MATERIAL_PATH = ROOT / "example_data" / "material_test.csv"
TEST_TRANSMITTANCE_PATH = ROOT / "example_data" / "transmittance_test.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = OMTMixer(
        24, 400, 301, 8,
        LinearEmbedding(24, 400, 2),
        material_normalization,
    ).to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Checkpoint loaded successfully.")

    thickness = np.loadtxt(TEST_THICKNESS_PATH, delimiter=",")
    material = np.loadtxt(TEST_MATERIAL_PATH, delimiter=",")
    target = np.loadtxt(TEST_TRANSMITTANCE_PATH, delimiter=",")

    with torch.no_grad():
        prediction = model(
            torch.tensor(thickness, dtype=torch.float32, device=device),
            torch.tensor(material, dtype=torch.float32, device=device),
        ).cpu().numpy()

    rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=1))
    print(f"Input shape: thickness={thickness.shape}, material={material.shape}")
    print(f"Output shape: {prediction.shape}")
    print("RMSE for the six supplied test structures:")
    for i, value in enumerate(rmse, start=1):
        print(f"  Test{i}: {value:.6f}")
    print(f"Mean RMSE: {rmse.mean():.6f}")


if __name__ == "__main__":
    main()
