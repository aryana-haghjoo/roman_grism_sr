#!/usr/bin/env python3
"""
train_roman_zhead.py

Retrain the z-head on JWST prism spectra masked to the Roman grism wavelength
window (0.99–1.95 µm). SR1 stays frozen; only the z-head weights are updated.

This directly targets the domain gap found in the zero-shot experiment: the
JWST z-head was trained on full 1–5 µm spectra and fails on the 24% wavelength
coverage that Roman provides. Re-training on the same partial-window input
teaches the z-head to estimate redshift from Roman-like inputs.

Input to SR1: flux_low (JWST prism) zeroed outside the Roman window, normalized
on in-window pixels only — matching the normalization used in zeroshot_inference.py.

Usage:
    source ../../super_resolution/sup_res/bin/activate
    python train_roman_zhead.py
    python train_roman_zhead.py --finetune        # warm-start from JWST z-head
    python train_roman_zhead.py --wandb_mode online

Output:
    train/roman_zhead/best_roman_zhead.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
REPO    = HERE.parent
SR_REPO = Path(os.path.dirname(REPO)) / "super_resolution"

SR1_DIR   = SR_REPO / "train" / "sr1_best"
ZHEAD_DIR = SR_REPO / "train" / "redshift_head"
DATA_NPZ  = SR_REPO / "data" / "spectra_dataset_2500.npz"
ROMAN_NPZ = REPO / "data" / "roman_mock_spectra.npz"
OUT_DIR   = REPO / "train" / "roman_zhead"

os.makedirs(OUT_DIR, exist_ok=True)

for p in [str(SR1_DIR), str(ZHEAD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_sr1 import SuperRes1D, get_or_make_split
from model_z_head import ZHead1D, heteroscedastic_nll


# ── normalization ─────────────────────────────────────────────────────────────
def normalize_roman_window(flux, roman_mask, eps=1e-25):
    """
    Zero outside Roman window, normalize using only in-window pixels.
    Matches normalize_roman_input() in zeroshot_inference.py.
    """
    out = np.zeros(len(flux), dtype=np.float32)
    region = flux[roman_mask].astype(np.float64)
    mean = np.nanmean(region)
    std  = np.nanstd(region)
    if std < eps:
        std = eps
    out[roman_mask] = ((region - mean) / std).astype(np.float32)
    return out


# ── dataset ───────────────────────────────────────────────────────────────────
class RomanWindowDataset(Dataset):
    """
    JWST prism spectra (flux_low) masked and renormalized to the Roman grism
    wavelength window. Pre-processed at init to avoid per-batch overhead.
    """
    def __init__(self, dataset_npz_path, roman_mask):
        data = np.load(dataset_npz_path, allow_pickle=True)
        flux_low = data["flux_low"]
        self.z   = data["z"].astype(np.float32)

        print(f"Pre-processing {len(flux_low)} spectra to Roman window "
              f"({roman_mask.sum()}/2500 pixels)...")
        N = len(flux_low)
        self.flux_masked = np.zeros((N, 2500), dtype=np.float32)
        for i in range(N):
            f = flux_low[i].copy()
            f[~roman_mask] = 0.0
            self.flux_masked[i] = normalize_roman_window(f, roman_mask)
        print("Done.")

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        x = torch.tensor(self.flux_masked[idx], dtype=torch.float32)
        z = torch.tensor(self.z[idx],           dtype=torch.float32)
        return x, z


# ── SR1 loader ────────────────────────────────────────────────────────────────
def load_sr1(device):
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
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.parameters())
    print(f"SR1 loaded and frozen ({n:,} params)")
    return model


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(zhead, sr1, loader, device, z_mean, z_std, z_min_n, z_max_n):
    zhead.eval()
    all_pred, all_true = [], []
    total_loss = 0.0

    for x, z in loader:
        x = x.to(device).unsqueeze(1)
        z = z.to(device)

        sr_mean, sr_logvar = sr1(x)
        z_in = torch.cat([sr_mean, 0.5 * sr_logvar], dim=1)

        z_n = (z - z_mean) / z_std
        mu_raw, logvar_n = zhead(z_in)
        logvar_n = logvar_n.clamp(-12, 12)
        mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)

        total_loss += heteroscedastic_nll(mu_n, logvar_n, z_n).item()

        z_pred = (mu_n.squeeze(-1) * z_std + z_mean).reshape(-1)
        all_pred.append(z_pred.cpu())
        all_true.append(z.cpu())

    z_pred = torch.cat(all_pred).numpy()
    z_true = torch.cat(all_true).numpy()
    dz = np.abs(z_pred - z_true) / (1.0 + np.abs(z_true))
    return (
        total_loss / max(1, len(loader)),
        float(np.median(dz)),
        float((dz > 0.15).mean()),
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",        type=int,   default=200)
    ap.add_argument("--batch_size",    type=int,   default=64)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-5)
    ap.add_argument("--hidden_dim",    type=int,   default=64)
    ap.add_argument("--num_blocks",    type=int,   default=4)
    ap.add_argument("--dropout",       type=float, default=0.1)
    ap.add_argument("--finetune",      action="store_true",
                    help="Warm-start from the JWST z-head weights")
    ap.add_argument("--wandb_project", type=str,   default="roman_grism_sr")
    ap.add_argument("--wandb_name",    type=str,   default="roman_zhead")
    ap.add_argument("--wandb_mode",    type=str,   default="disabled",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Roman window mask ─────────────────────────────────────────────────────
    roman_data = np.load(str(ROMAN_NPZ), allow_pickle=True)
    roman_mask = roman_data["roman_wave_mask"].astype(bool)

    # ── Dataset & split ───────────────────────────────────────────────────────
    full_ds = RomanWindowDataset(str(DATA_NPZ), roman_mask)
    N = len(full_ds)

    # Reuse the exact same split as SR1 / JWST z-head training
    train_idx, val_idx, _ = get_or_make_split(
        str(DATA_NPZ), N, train_frac=0.8, seed=42,
        split_dir=str(SR_REPO / "splits"),
    )
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")

    train_loader = DataLoader(
        Subset(full_ds, train_idx), batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx), batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── z normalisation ───────────────────────────────────────────────────────
    z_train = full_ds.z[train_idx]
    z_mean  = float(z_train.mean())
    z_std   = float(z_train.std())
    z_min_n = float((z_train.min() - z_mean) / z_std)
    z_max_n = float((z_train.max() - z_mean) / z_std)
    print(f"z: mean={z_mean:.4f}  std={z_std:.4f}  "
          f"norm_range=[{z_min_n:.3f}, {z_max_n:.3f}]")

    # ── Models ────────────────────────────────────────────────────────────────
    sr1   = load_sr1(device)
    zhead = ZHead1D(in_channels=2, hidden_dim=args.hidden_dim,
                    num_blocks=args.num_blocks, dropout=args.dropout).to(device)

    if args.finetune:
        ck = torch.load(str(ZHEAD_DIR / "best_zhead.pth"), map_location="cpu")
        state = ck.get("zhead_state_dict", ck)
        missing, unexpected = zhead.load_state_dict(state, strict=False)
        print(f"Warm-start from JWST z-head "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in zhead.parameters() if p.requires_grad)
    print(f"Z-head: {n_params:,} trainable params")

    opt = torch.optim.AdamW(
        zhead.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name,
               config=vars(args), mode=args.wandb_mode)

    # ── Training loop ─────────────────────────────────────────────────────────
    out_path   = str(OUT_DIR / "best_roman_zhead.pth")
    best_val   = 1e30

    for epoch in range(args.epochs):
        zhead.train()
        tr_loss = 0.0

        for x, z in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                          leave=False):
            x = x.to(device).unsqueeze(1)
            z = z.to(device)

            with torch.no_grad():
                sr_mean, sr_logvar = sr1(x)

            z_in = torch.cat([sr_mean, 0.5 * sr_logvar], dim=1)
            z_n  = (z - z_mean) / z_std
            mu_raw, logvar_n = zhead(z_in)
            logvar_n = logvar_n.clamp(-12, 12)
            mu_n = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)

            loss = heteroscedastic_nll(mu_n, logvar_n, z_n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(zhead.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()

        tr_loss /= max(1, len(train_loader))
        val_loss, med_dz, outlier = evaluate(
            zhead, sr1, val_loader, device,
            z_mean, z_std, z_min_n, z_max_n,
        )

        print(f"Epoch {epoch+1:3d}  "
              f"tr={tr_loss:.4f}  val={val_loss:.4f}  "
              f"med|Δz|/(1+z)={med_dz:.4f}  outlier={outlier:.1%}")

        wandb.log({
            "epoch":                epoch + 1,
            "train_loss":           tr_loss,
            "val_loss":             val_loss,
            "val_med_dz_over_1pz":  med_dz,
            "val_outlier_rate":     outlier,
        }, step=epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "zhead_state_dict": zhead.state_dict(),
                "z_mean":           z_mean,
                "z_std":            z_std,
                "z_min_n":          z_min_n,
                "z_max_n":          z_max_n,
                "use_sigma":        True,
                "roman_mask_npixels": int(roman_mask.sum()),
                "config":           vars(args),
            }, out_path)
            print(f"  -> best_roman_zhead.pth  (val={best_val:.4f})")

    print(f"\nDone. Best checkpoint: {out_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
