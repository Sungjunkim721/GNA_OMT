"""Training utilities for the OMT-mixer forward surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch


@dataclass
class TrainingResult:
    elapsed_seconds: float
    history: np.ndarray
    best_validation_loss: float | None
    best_epoch: int | None


def evaluate_mse(loader, model, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for thickness, material, spectrum in loader:
            thickness = thickness.to(device)
            material = material.to(device)
            spectrum = spectrum.to(device)
            total += float(criterion(model(thickness, material), spectrum))
    model.train()
    return total / len(loader)


def train_forward_model(
    model,
    train_loader,
    val_loader,
    *,
    optimizer,
    criterion,
    device,
    epochs: int,
    checkpoint_path: str | Path,
    scheduler=None,
    patience: int | None = None,
    print_every: int = 25,
) -> TrainingResult:
    """Train the forward surrogate and save the best validation checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and str(device).startswith("cuda"))

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = None
    stale = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for thickness, material, spectrum in train_loader:
            thickness = thickness.to(device)
            material = material.to(device)
            spectrum = spectrum.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(thickness, material), spectrum)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach())

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_loader)
        val_loss = evaluate_mse(val_loader, model, criterion, device)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if print_every and epoch % print_every == 0:
            print(f"epoch={epoch:4d} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

        if val_loss <= best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if patience is not None and stale > patience:
                break

    elapsed = time.time() - start
    history = np.column_stack([np.asarray(train_hist), np.asarray(val_hist)])
    return TrainingResult(elapsed, history, best_val, best_epoch)
