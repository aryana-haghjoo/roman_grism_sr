#!/usr/bin/env python3
"""
finetune_sr1_roman.py

Fine-tune SR1 to accept Roman grism inputs. SR1 was originally trained on
full 1–5 µm JWST prism spectra; here we adapt it to the Roman wavelength
window (0.99–1.95 µm, 24% coverage), where it must reconstruct the JWST
grating truth despite seeing a partial, differently-normalised input.

Loss (two terms):
  1. Supervised NLL inside the Roman window — match JWST grating truth
     (heteroscedastic, same formulation as original SR1 training)
  2. Distillation MSE outside the Roman window — match frozen SR1(JWST prism)
     This prevents catastrophic forgetting of the full-spectrum behaviour
     used by SR2 and the z-head.

Inputs:
  data/roman_train_spectra.npz   — forward-modeled Roman training set
                                   (contains roman_flux, flux_low, flux_high,
                                    roman_wave_mask, z)

Outputs:
  train/roman_sr1/best_sr1_roman.pth

Usage:
    source ../../super_resolution/sup_res/bin/activate
    python scripts/finetune_sr1_roman.py
    python scripts/finetune_sr1_roman.py --wandb_mode online
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import copy

# ── paths ─────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
REPO    = HERE.parent
SR_REPO = Path(os.path.dirname(REPO)) / "super_resolution"

SR1_DIR   = SR_REPO / "train" / "sr1_best"
TRAIN_NPZ = REPO / "data" / "roman_train_spectra.npz"
VAL_NPZ   = REPO / "data" / "roman_mock_spectra.npz"
OUT_DIR   = REPO / "train" / "roman_sr1"

os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, str(SR1_DIR))
from train_sr1 import SuperRes1D


# ── normalization ─────────────────────────────────────────────────────────────
def normalize_roman_window(flux, roman_mask, eps=1e-25):
    out = np.zeros(len(flux), dtype=np.float32)
    region = flux[roman_mask].astype(np.float64)
    mean, std = np.nanmean(region), np.nanstd(region)
    if std < eps:
        std = eps
    out[roman_mask] = ((region - mean) / std).astype(np.float32)
    return out


def normalize_full(flux, eps=1e-25):
    mean, std = np.nanmean(flux), np.nanstd(flux)
    if std < eps:
        std = eps
    return ((flux - mean) / std).astype(np.float32), mean, std


# ── dataset ───────────────────────────────────────────────────────────────────
class SR1RomanDataset(Dataset):
    """
    Returns per-spectrum:
      roman_norm   : (2500,) Roman input, normalised on Roman-window pixels
      prism_norm   : (2500,) JWST prism input, normalised on full spectrum
      high_norm    : (2500,) JWST grating target, normalised on full spectrum
      high_err_norm: (2500,) JWST grating uncertainty, scaled by same std
    """
    def __init__(self, npz_path):
        data       = np.load(str(npz_path), allow_pickle=True)
        roman_flux = data["roman_flux"]
        flux_low   = data["flux_low"]
        flux_high  = data["flux_high"]
        flux_hi_err= data["flux_high_err"]
        self.roman_mask = data["roman_wave_mask"].astype(bool)
        self.z     = data["z"].astype(np.float32)
        N          = len(roman_flux)

        print(f"Pre-processing {N} spectra from {Path(npz_path).name}...")
        self.roman_norm    = np.zeros((N, 2500), dtype=np.float32)
        self.prism_norm    = np.zeros((N, 2500), dtype=np.float32)
        self.high_norm     = np.zeros((N, 2500), dtype=np.float32)
        self.high_err_norm = np.zeros((N, 2500), dtype=np.float32)

        for i in range(N):
            self.roman_norm[i] = normalize_roman_window(roman_flux[i], self.roman_mask)
            self.prism_norm[i], _, _ = normalize_full(flux_low[i])
            self.high_norm[i], _, std_hi = normalize_full(flux_high[i])
            self.high_err_norm[i] = (flux_hi_err[i] / (std_hi if std_hi > 1e-9 else 1.0)).astype(np.float32)
        print("Done.")

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.roman_norm[idx]),
            torch.tensor(self.prism_norm[idx]),
            torch.tensor(self.high_norm[idx]),
            torch.tensor(self.high_err_norm[idx]),
        )


# ── SR1 loader ────────────────────────────────────────────────────────────────
def load_sr1(device, freeze=False):
    import yaml
    with open(str(SR1_DIR / "best_config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = {k: v["value"] if isinstance(v, dict) and "value" in v else v
           for k, v in cfg.items() if not k.startswith("_")}
    model = SuperRes1D(
        in_channels=1,
        hidden_dim=int(cfg.get("hidden_dim", 96)),
        num_res_blocks=int(cfg.get("num_res_blocks", 12)),
        dropout=float(cfg.get("dropout", 0.02)),
    ).to(device)
    ckpt = torch.load(str(SR1_DIR / "best_superres_model.pth"), map_location="cpu")
    model.load_state_dict(ckpt)
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


# ── loss ─────────────────────────────────────────────────────────────────────
def nll_loss(mean, log_var, target, err, mask, eps=1e-8):
    """Heteroscedastic NLL on masked pixels only."""
    model_var = torch.exp(log_var)
    total_var = (model_var + err ** 2).clamp_min(eps)
    nll = 0.5 * (torch.log(total_var) + (mean - target) ** 2 / total_var)
    return nll[:, :, mask].mean()


def distill_loss(mean_ft, mean_frozen, mask):
    """MSE between fine-tuned and frozen SR1 outputs on non-Roman pixels."""
    outside = ~mask
    return F.mse_loss(mean_ft[:, :, outside], mean_frozen[:, :, outside])


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(sr1_ft, sr1_frozen, loader, device, roman_mask,
             lam_roman, lam_distill):
    sr1_ft.eval()
    total, n = 0.0, 0
    for roman, prism, high, high_err in loader:
        roman    = roman.to(device).unsqueeze(1)
        prism    = prism.to(device).unsqueeze(1)
        high     = high.to(device).unsqueeze(1)
        high_err = high_err.to(device).unsqueeze(1)

        mean_ft, lv_ft   = sr1_ft(roman)
        mean_frz, _      = sr1_frozen(prism)

        loss = (lam_roman   * nll_loss(mean_ft, lv_ft, high, high_err, roman_mask)
              + lam_distill * distill_loss(mean_ft, mean_frz, roman_mask))
        total += loss.item()
        n += 1
    sr1_ft.train()
    return total / max(n, 1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=5e-5,
                    help="Low LR to avoid catastrophic forgetting (default: 5e-5)")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--lam_roman",    type=float, default=1.0,
                    help="Weight on supervised NLL inside Roman window")
    ap.add_argument("--lam_distill",  type=float, default=0.5,
                    help="Weight on distillation MSE outside Roman window")
    ap.add_argument("--wandb_project",type=str,   default="roman_grism_sr")
    ap.add_argument("--wandb_name",   type=str,   default="sr1_roman_finetune")
    ap.add_argument("--wandb_mode",   type=str,   default="disabled",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not TRAIN_NPZ.exists():
        raise FileNotFoundError(
            f"{TRAIN_NPZ} not found.\n"
            "Run: python scripts/make_roman_spectra.py --split train"
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = SR1RomanDataset(TRAIN_NPZ)
    val_ds   = SR1RomanDataset(VAL_NPZ)
    roman_mask = torch.tensor(train_ds.roman_mask, device=device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Models ────────────────────────────────────────────────────────────────
    sr1_ft     = load_sr1(device, freeze=False)   # will be fine-tuned
    sr1_frozen = load_sr1(device, freeze=True)    # teacher for distillation

    n_params = sum(p.numel() for p in sr1_ft.parameters() if p.requires_grad)
    print(f"SR1 fine-tune: {n_params:,} trainable params  "
          f"(lr={args.lr}, lam_roman={args.lam_roman}, lam_distill={args.lam_distill})")

    opt = torch.optim.AdamW(sr1_ft.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    # ── W&B ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name,
               config=vars(args), mode=args.wandb_mode)

    # ── Training loop ─────────────────────────────────────────────────────────
    out_path = str(OUT_DIR / "best_sr1_roman.pth")
    best_val = 1e30

    for epoch in range(args.epochs):
        sr1_ft.train()
        tr_loss = 0.0

        for roman, prism, high, high_err in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            roman    = roman.to(device).unsqueeze(1)
            prism    = prism.to(device).unsqueeze(1)
            high     = high.to(device).unsqueeze(1)
            high_err = high_err.to(device).unsqueeze(1)

            mean_ft, lv_ft = sr1_ft(roman)

            with torch.no_grad():
                mean_frz, _ = sr1_frozen(prism)

            loss = (args.lam_roman   * nll_loss(mean_ft, lv_ft, high, high_err, roman_mask)
                  + args.lam_distill * distill_loss(mean_ft, mean_frz, roman_mask))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sr1_ft.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))
        val_loss = evaluate(sr1_ft, sr1_frozen, val_loader, device, roman_mask,
                            args.lam_roman, args.lam_distill)

        print(f"Epoch {epoch+1:3d}  tr={tr_loss:.4f}  val={val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": tr_loss, "val_loss": val_loss},
                  step=epoch+1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "sr1_state_dict": sr1_ft.state_dict(),
                "config": vars(args),
            }, out_path)
            print(f"  -> best_sr1_roman.pth  (val={best_val:.4f})")

    print(f"\nDone. Best checkpoint: {out_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
