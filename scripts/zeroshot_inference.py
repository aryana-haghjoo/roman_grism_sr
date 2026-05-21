#!/usr/bin/env python3
"""
zeroshot_inference.py

Run the JWST-trained SR1 + z-head + SR2 pipeline on mock Roman grism spectra
without any fine-tuning.  This characterises the domain gap.

Input:  data/roman_mock_spectra.npz  (from make_roman_spectra.py)
Output: results/zeroshot_results.npz
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
REPO     = HERE.parent
SR_REPO  = Path(os.path.dirname(REPO)) / "super_resolution"

SR1_DIR      = SR_REPO / "train" / "sr1_best"
ZHEAD_DIR    = SR_REPO / "train" / "redshift_head"
SR2_DIR      = SR_REPO / "train" / "sr2_best"
DATA_NPZ     = SR_REPO / "data" / "spectra_dataset_2500.npz"
ROMAN_NPZ    = REPO / "data" / "roman_mock_spectra.npz"
OUT_NPZ      = REPO / "results" / "zeroshot_results.npz"

os.makedirs(REPO / "results", exist_ok=True)

for p in [str(SR1_DIR), str(ZHEAD_DIR), str(SR2_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from train_sr1 import SuperRes1D, get_or_make_split
from model_z_head import ZHead1D
from line_list_rest import LINE_LIST_REST_AA
from train_sr2_attention import (
    SR2Attention, load_sr1, load_zhead,
    build_line_mask, constrain_delta, angstrom_to_micron,
)


# ── normalization (mirrors SR1 training) ─────────────────────────────────────
def normalize_spectrum(x, eps=1e-25):
    mean = np.nanmean(x)
    std  = np.nanstd(x)
    if std < eps:
        std = eps
    return (x - mean) / std, mean, std


def normalize_roman_input(roman_flux, roman_wave_mask):
    """
    Normalize the Roman grism spectrum using only the pixels inside the
    Roman wavelength coverage (non-zero region), then embed in zeros.
    This avoids the mean/std being swamped by the zero-padded region.
    """
    out = np.zeros_like(roman_flux)
    region = roman_flux[roman_wave_mask]
    region_norm, mean, std = normalize_spectrum(region)
    out[roman_wave_mask] = region_norm
    return out, mean, std


# ── load models ───────────────────────────────────────────────────────────────
def load_pipeline(device):
    sr1, sr1_cfg = load_sr1(
        str(SR1_DIR / "best_config.yaml"),
        str(SR1_DIR / "best_superres_model.pth"),
        device,
    )

    zhead, z_mean, z_std, use_sigma, _ = load_zhead(
        str(ZHEAD_DIR / "best_zhead.pth"), device
    )

    # ── SR2 (attention) ───────────────────────────────────────────────────────
    sr2_ckpt_path = SR2_DIR / "best_sr2_attn.pth"
    ck = torch.load(str(sr2_ckpt_path), map_location="cpu")
    sr2_cfg = ck["config"]

    data_npz    = np.load(str(DATA_NPZ), allow_pickle=True)
    wave_hi_um  = np.asarray(data_npz["wavelength_high"], dtype=np.float32)
    line_rest_um = angstrom_to_micron([w for _, w in LINE_LIST_REST_AA])

    in_ch = 2
    if sr2_cfg.get("use_sr1_sigma", True):    in_ch += 1
    if sr2_cfg.get("use_line_mask", True):    in_ch += 1
    if sr2_cfg.get("use_zhat_channel", True): in_ch += 1

    sr2 = SR2Attention(
        in_channels=in_ch,
        line_rest_um=line_rest_um,
        wave_hi_um=wave_hi_um,
        line_dim=int(sr2_cfg.get("line_dim", 128)),
        num_attn_heads=int(sr2_cfg.get("num_attn_heads", 4)),
        num_attn_layers=int(sr2_cfg.get("num_attn_layers", 4)),
        window_half=int(sr2_cfg.get("window_half", 25)),
        cnn_dim=int(sr2_cfg.get("cnn_dim", 96)),
        num_cnn_blocks=int(sr2_cfg.get("num_cnn_blocks", 6)),
        dropout=float(sr2_cfg.get("dropout", 0.02)),
    ).to(device)
    # Remap old checkpoint key names to current class attribute names
    key_map = {
        "line_rest_um_buf": "line_rest_um",
        "wave_hi_um_buf":   "wave_hi_um",
        "cnn_initial.0.weight": "cnn_in.0.weight",
        "cnn_initial.0.bias":   "cnn_in.0.bias",
        "cnn_delta.weight":     "cnn_out.weight",
        "cnn_delta.bias":       "cnn_out.bias",
    }
    sd = {key_map.get(k, k): v for k, v in ck["sr2_state_dict"].items()}
    sr2.load_state_dict(sd)
    sr2.eval()
    for p in sr2.parameters():
        p.requires_grad = False

    # z normalisation bounds (reproduce training split)
    N = len(data_npz["flux_low"])
    train_idx, _, _ = get_or_make_split(str(DATA_NPZ), N, train_frac=0.8, seed=42)
    z_all = data_npz["z"]
    z_train = z_all[train_idx].astype(np.float32)
    z_min_n = float((z_train.min() - z_mean) / z_std)
    z_max_n = float((z_train.max() - z_mean) / z_std)

    wave_t = torch.tensor(wave_hi_um, device=device)

    print(f"SR1, ZHead, SR2 loaded on {device}")
    return dict(
        sr1=sr1, zhead=zhead, sr2=sr2, sr2_cfg=sr2_cfg,
        z_mean=z_mean, z_std=z_std, z_min_n=z_min_n, z_max_n=z_max_n,
        use_sigma=use_sigma, wave_hi_um=wave_hi_um,
        wave_t=wave_t, line_rest_um=line_rest_um,
    )


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_zeroshot(pipeline, roman_data, batch_size=64):
    device   = next(pipeline["sr1"].parameters()).device
    sr1      = pipeline["sr1"]
    zhead    = pipeline["zhead"]
    sr2      = pipeline["sr2"]
    cfg      = pipeline["sr2_cfg"]
    z_mean   = pipeline["z_mean"]
    z_std    = pipeline["z_std"]
    z_min_n  = pipeline["z_min_n"]
    z_max_n  = pipeline["z_max_n"]
    use_sig  = pipeline["use_sigma"]
    wave_t   = pipeline["wave_t"]
    line_rest = pipeline["line_rest_um"]

    delta_cap     = float(cfg.get("delta_cap", 10.0))
    logvar_min    = float(cfg.get("logvar_min", np.log(0.05)))
    logvar_max    = float(cfg.get("logvar_max", np.log(10.0)))
    sigma_base_um = float(cfg.get("sigma_base_um", 0.005))
    use_sr1_sigma = bool(cfg.get("use_sr1_sigma", True))
    use_line_mask = bool(cfg.get("use_line_mask", True))
    use_zhat_ch   = bool(cfg.get("use_zhat_channel", True))

    roman_flux_raw  = roman_data["roman_flux"]        # (N, 2500) unnormalised
    roman_wave_mask = roman_data["roman_wave_mask"]   # (2500,) bool
    flux_high       = roman_data["flux_high"]         # (N, 2500) JWST grating truth
    flux_high_err   = roman_data["flux_high_err"]
    flux_low        = roman_data["flux_low"]          # (N, 2500) JWST prism
    z_true          = roman_data["z"]
    N               = len(roman_flux_raw)

    # Normalise inputs
    print("Normalising Roman spectra...")
    roman_norm = np.zeros_like(roman_flux_raw)
    for i in range(N):
        roman_norm[i], _, _ = normalize_roman_input(roman_flux_raw[i], roman_wave_mask)

    # Also normalise the JWST prism for comparison baseline
    jwst_prism_norm = np.zeros_like(flux_low)
    for i in range(N):
        jwst_prism_norm[i], _, _ = normalize_spectrum(flux_low[i])

    # Also normalise the JWST grating truth for evaluation in normalised space
    flux_high_norm = np.zeros_like(flux_high)
    high_means, high_stds = np.zeros(N), np.zeros(N)
    for i in range(N):
        flux_high_norm[i], high_means[i], high_stds[i] = normalize_spectrum(flux_high[i])

    # Output arrays
    sr1_out  = np.zeros((N, 2500), dtype=np.float32)
    sr2_out  = np.zeros((N, 2500), dtype=np.float32)
    sr2_sig  = np.zeros((N, 2500), dtype=np.float32)
    zhat_all = np.zeros(N, dtype=np.float32)
    sigz_all = np.zeros(N, dtype=np.float32)

    # Also run the JWST prism baseline through the same pipeline
    sr1_jwst  = np.zeros((N, 2500), dtype=np.float32)
    sr2_jwst  = np.zeros((N, 2500), dtype=np.float32)
    zhat_jwst = np.zeros(N, dtype=np.float32)

    print(f"Running zero-shot inference on {N} Roman spectra (batch={batch_size})...")
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if start % 500 == 0:
            print(f"  {start}/{N}")

        # ── Roman grism input ─────────────────────────────────────────────────
        x_roman = torch.tensor(roman_norm[start:end], dtype=torch.float32,
                               device=device).unsqueeze(1)   # (B,1,2500)
        x_roman = torch.nan_to_num(x_roman)

        s1_mean, s1_logvar = sr1(x_roman)
        s1_log_sigma = 0.5 * s1_logvar
        s1_sigma     = torch.exp(s1_log_sigma).clamp_min(1e-6)

        z_in   = torch.cat([s1_mean, s1_log_sigma], dim=1) if use_sig else s1_mean
        mu_raw, logvar_z = zhead(z_in)
        mu_n   = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_raw)
        zhat   = (mu_n.squeeze(-1) * z_std + z_mean).reshape(-1)
        zhat   = torch.nan_to_num(zhat, nan=float(z_mean))
        sigz   = (torch.exp(0.5 * logvar_z).squeeze(-1) * z_std).reshape(-1)

        lmask = build_line_mask(wave_t, zhat, line_rest, sigma_base_um=sigma_base_um)

        chans = [x_roman, s1_mean]
        if use_sr1_sigma: chans.append(s1_sigma)
        if use_line_mask: chans.append(lmask)
        if use_zhat_ch:   chans.append(zhat[:, None, None].expand(-1, 1, s1_mean.shape[-1]))
        x_in = torch.cat(chans, dim=1)

        delta_raw, s2_logvar = sr2(x_in, zhat)
        delta  = constrain_delta(delta_raw, delta_cap)
        s2_mean = s1_mean + delta
        s2_logvar = s2_logvar.clamp(logvar_min, logvar_max)

        sr1_out[start:end]  = s1_mean[:, 0].cpu().numpy()
        sr2_out[start:end]  = s2_mean[:, 0].cpu().numpy()
        sr2_sig[start:end]  = torch.exp(0.5 * s2_logvar)[:, 0].cpu().numpy()
        zhat_all[start:end] = zhat.cpu().numpy()
        sigz_all[start:end] = sigz.cpu().numpy()

        # ── JWST prism baseline (same pipeline, original input) ───────────────
        x_prism = torch.tensor(jwst_prism_norm[start:end], dtype=torch.float32,
                               device=device).unsqueeze(1)
        x_prism = torch.nan_to_num(x_prism)

        s1p, s1p_lv = sr1(x_prism)
        s1p_ls = 0.5 * s1p_lv
        zp_in  = torch.cat([s1p, s1p_ls], dim=1) if use_sig else s1p
        mu_rp, lv_zp = zhead(zp_in)
        mu_np   = z_min_n + (z_max_n - z_min_n) * torch.sigmoid(mu_rp)
        zhat_p  = (mu_np.squeeze(-1) * z_std + z_mean).reshape(-1)
        zhat_p  = torch.nan_to_num(zhat_p, nan=float(z_mean))

        lmask_p = build_line_mask(wave_t, zhat_p, line_rest, sigma_base_um=sigma_base_um)
        chans_p = [x_prism, s1p]
        if use_sr1_sigma: chans_p.append(torch.exp(0.5*s1p_lv).clamp_min(1e-6))
        if use_line_mask: chans_p.append(lmask_p)
        if use_zhat_ch:   chans_p.append(zhat_p[:, None, None].expand(-1, 1, s1p.shape[-1]))
        dp_raw, _ = sr2(torch.cat(chans_p, dim=1), zhat_p)
        s2p = s1p + constrain_delta(dp_raw, delta_cap)

        sr1_jwst[start:end]  = s1p[:, 0].cpu().numpy()
        sr2_jwst[start:end]  = s2p[:, 0].cpu().numpy()
        zhat_jwst[start:end] = zhat_p.cpu().numpy()

    print(f"Saving → {OUT_NPZ}")
    np.savez(
        str(OUT_NPZ),
        # Roman pipeline outputs
        roman_sr1=sr1_out,
        roman_sr2=sr2_out,
        roman_sr2_sigma=sr2_sig,
        roman_zhat=zhat_all,
        roman_sigz=sigz_all,
        # JWST prism baseline outputs
        jwst_sr1=sr1_jwst,
        jwst_sr2=sr2_jwst,
        jwst_zhat=zhat_jwst,
        # Inputs and truth
        roman_input_norm=roman_norm,
        jwst_prism_norm=jwst_prism_norm,
        flux_high_norm=flux_high_norm,
        flux_high=roman_data["flux_high"],
        flux_high_err=roman_data["flux_high_err"],
        flux_low=flux_low,
        wavelength=roman_data["wavelength_high"],
        roman_wave_mask=roman_wave_mask,
        z_true=z_true,
        high_means=high_means,
        high_stds=high_stds,
    )
    print("Done.")
    return OUT_NPZ


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Roman mock spectra...")
    roman_data = dict(np.load(str(ROMAN_NPZ), allow_pickle=True))

    pipeline = load_pipeline(device)
    run_zeroshot(pipeline, roman_data)


if __name__ == "__main__":
    main()
