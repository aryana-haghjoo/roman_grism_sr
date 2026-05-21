#!/usr/bin/env python3
"""
plot_zeroshot.py

Diagnostic plots for the zero-shot Roman grism transfer experiment.

Panels produced:
  1. Example spectra  — LR prism / Roman grism input / SR2(Roman) / HR truth
  2. Residual maps    — (SR2-HR) as function of redshift, Roman vs JWST baseline
  3. Redshift scatter — z_true vs z_hat for Roman input vs JWST prism baseline
  4. SNR comparison   — per-galaxy SNR gain in the Roman wavelength window
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE    = Path(__file__).resolve().parent
REPO    = HERE.parent
RES_NPZ = REPO / "results" / "zeroshot_results.npz"
OUT_DIR = REPO / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load():
    d = dict(np.load(str(RES_NPZ), allow_pickle=True))
    return d


def pick_example_galaxies(d, n=4):
    """
    Pick n well-behaved test galaxies:
      - redshift z in [0.5, 2.5]  (Roman science window)
      - Roman input has reasonable SNR
      - SR2 is finite everywhere in the Roman window
    """
    z  = d["z_true"]
    mask_z = (z >= 0.5) & (z <= 2.5)
    rmask  = d["roman_wave_mask"].astype(bool)

    roman_in = d["roman_input_norm"]
    snr_roman = np.abs(roman_in[:, rmask]).mean(axis=1) / \
                (np.abs(roman_in[:, rmask]).std(axis=1) + 1e-8)

    sr2_ok = np.isfinite(d["roman_sr2"][:, rmask]).all(axis=1)
    cand   = np.where(mask_z & sr2_ok & (snr_roman > 0.5))[0]

    # Pick evenly spaced across the z range
    if len(cand) < n:
        return cand[:n]
    z_cand = z[cand]
    idx    = np.argsort(z_cand)
    picks  = idx[np.linspace(0, len(idx)-1, n, dtype=int)]
    return cand[picks]


def panel_example_spectra(d, ax_list, picks):
    wave     = d["wavelength"].astype(float)
    rmask    = d["roman_wave_mask"].astype(bool)
    roman_min = wave[rmask].min()
    roman_max = wave[rmask].max()

    for ax, i in zip(ax_list, picks):
        hr    = d["flux_high_norm"][i]
        roman = d["roman_input_norm"][i]
        sr2_r = d["roman_sr2"][i]
        sr2_j = d["jwst_sr2"][i]
        z     = float(d["z_true"][i])
        zhat  = float(d["roman_zhat"][i])

        # clip to 1-2.5 µm for readability
        win = (wave >= 0.95) & (wave <= 2.6)

        ax.plot(wave[win], hr[win],    color="#6a0dad", lw=1.3, alpha=0.85, label="JWST grating (truth)")
        ax.plot(wave[win], sr2_j[win], color="#1f77b4", lw=1.0, alpha=0.75, ls="--", label="SR2 (JWST prism in)")
        ax.plot(wave[win], sr2_r[win], color="#d62728", lw=1.2, alpha=0.90, label="SR2 (Roman in, zero-shot)")
        ax.plot(wave[win], roman[win], color="#ff7f0e", lw=0.8, alpha=0.6,  ls=":",  label="Roman grism input")

        # Roman window shading
        ax.axvspan(roman_min, roman_max, alpha=0.06, color="gray", label="_nolegend_")

        vals = np.concatenate([hr[win], sr2_r[win]])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = np.percentile(vals, [1, 99])
            pad = 0.12 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlim(0.95, 2.6)
        ax.set_ylabel("Norm. flux")
        ax.set_title(f"z = {z:.3f}  |  ẑ(Roman) = {zhat:.3f}", fontsize=9)

    ax_list[-1].set_xlabel("Observed wavelength (µm)")
    ax_list[0].legend(loc="upper right", fontsize=8, framealpha=0.7)


def panel_residual_maps(d, ax_roman, ax_jwst):
    wave  = d["wavelength"].astype(float)
    rmask = d["roman_wave_mask"].astype(bool)
    z     = d["z_true"]

    valid = (z > 0.0) & (z < 8.0)
    z_v   = z[valid]
    hr_v  = d["flux_high_norm"][valid]
    sr2_r = d["roman_sr2"][valid]
    sr2_j = d["jwst_sr2"][valid]

    # Residual only in Roman wavelength window
    resid_r = (sr2_r - hr_v)[:, rmask]
    resid_j = (sr2_j - hr_v)[:, rmask]

    # Bin by redshift
    z_bins  = np.linspace(0, 7, 35)
    z_cents = 0.5 * (z_bins[:-1] + z_bins[1:])
    rms_r, rms_j = [], []
    for zlo, zhi in zip(z_bins[:-1], z_bins[1:]):
        sel = (z_v >= zlo) & (z_v < zhi)
        if sel.sum() < 3:
            rms_r.append(np.nan)
            rms_j.append(np.nan)
        else:
            rms_r.append(np.sqrt(np.nanmean(resid_r[sel]**2)))
            rms_j.append(np.sqrt(np.nanmean(resid_j[sel]**2)))

    rms_r = np.array(rms_r)
    rms_j = np.array(rms_j)

    ax_roman.plot(z_cents, rms_r, color="#d62728", lw=1.5, label="SR2 (Roman, zero-shot)")
    ax_roman.plot(z_cents, rms_j, color="#1f77b4", lw=1.5, ls="--", label="SR2 (JWST prism)")
    ax_roman.set_xlabel("Redshift")
    ax_roman.set_ylabel("RMS residual (Roman window)")
    ax_roman.legend()
    ax_roman.set_xlim(0, 7)

    # Also show pixel-level residual map for Roman SR2
    # Sort by redshift, show as 2D image
    sort_idx = np.argsort(z_v)
    img = resid_r[sort_idx]
    # Clip and show median-binned version (downsample to 100 rows)
    n_rows = min(150, img.shape[0])
    step   = max(1, img.shape[0] // n_rows)
    img_ds = img[::step]
    z_ds   = z_v[sort_idx][::step]

    vmax = np.nanpercentile(np.abs(img_ds), 95)
    im = ax_jwst.imshow(
        img_ds, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[0, rmask.sum(), z_ds.min(), z_ds.max()],
        origin="lower",
    )
    ax_jwst.set_xlabel("Wavelength pixel (Roman window)")
    ax_jwst.set_ylabel("Redshift")
    ax_jwst.set_title("SR2(Roman) − HR  residual map", fontsize=9)
    plt.colorbar(im, ax=ax_jwst, label="Residual")


def panel_redshift(d, ax):
    z_true   = d["z_true"]
    zhat_r   = d["roman_zhat"]
    zhat_j   = d["jwst_zhat"]

    valid = (z_true > 0.0) & (z_true < 8.0) & np.isfinite(zhat_r) & np.isfinite(zhat_j)
    z_v, zr_v, zj_v = z_true[valid], zhat_r[valid], zhat_j[valid]

    dz_r = np.abs(zr_v - z_v) / (1 + z_v)
    dz_j = np.abs(zj_v - z_v) / (1 + z_v)
    out_r = (dz_r > 0.15).mean()
    out_j = (dz_j > 0.15).mean()
    med_r = np.median(dz_r)
    med_j = np.median(dz_j)

    ax.scatter(z_v, zr_v, s=1, alpha=0.3, color="#d62728",
               label=f"Roman zero-shot  med|Δz|/(1+z)={med_r:.3f}  outlier={out_r:.1%}")
    ax.scatter(z_v, zj_v, s=1, alpha=0.3, color="#1f77b4",
               label=f"JWST prism baseline  med={med_j:.3f}  outlier={out_j:.1%}")
    ax.plot([0, 8], [0, 8], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True redshift")
    ax.set_ylabel("Predicted redshift")
    ax.set_xlim(0, 8); ax.set_ylim(0, 8)
    ax.legend(fontsize=8, markerscale=5)


def panel_snr(d, ax):
    rmask  = d["roman_wave_mask"].astype(bool)
    hr     = d["flux_high_norm"]
    sr2_r  = d["roman_sr2"]
    sr2_j  = d["jwst_sr2"]
    roman_in = d["roman_input_norm"]

    def snr_gain(sr, lr, hr, mask):
        # SNR = signal / noise, where noise estimated from residuals
        sig = np.abs(hr[:, mask]).mean(axis=1)
        noise_lr = np.sqrt(np.mean((lr[:, mask] - hr[:, mask])**2, axis=1))
        noise_sr = np.sqrt(np.mean((sr[:, mask] - hr[:, mask])**2, axis=1))
        # Avoid division by zero
        gain = noise_lr / (noise_sr + 1e-8)
        return gain

    gain_roman = snr_gain(sr2_r, roman_in, hr, rmask)
    gain_jwst  = snr_gain(sr2_j, d["jwst_prism_norm"], hr, rmask)

    bins = np.logspace(-1, 1.5, 40)
    ax.hist(gain_roman, bins=bins, histtype="step", color="#d62728", lw=1.5,
            label=f"Roman zero-shot  median={np.median(gain_roman):.2f}×")
    ax.hist(gain_jwst,  bins=bins, histtype="step", color="#1f77b4", lw=1.5, ls="--",
            label=f"JWST prism  median={np.median(gain_jwst):.2f}×")
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("SR reconstruction gain vs input (Roman window)")
    ax.set_ylabel("Number of galaxies")
    ax.legend(fontsize=8)


def main():
    d = load()
    picks = pick_example_galaxies(d, n=4)
    print(f"Example galaxies: indices {picks}, z = {d['z_true'][picks]}")

    # ── Figure 1: example spectra ─────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)
    panel_example_spectra(d, axes, picks)
    plt.tight_layout()
    out = OUT_DIR / "zeroshot_examples.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ── Figure 2: residuals + redshift + SNR ─────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax_rms  = fig.add_subplot(gs[0, :2])
    ax_map  = fig.add_subplot(gs[0, 2])
    ax_z    = fig.add_subplot(gs[1, :2])
    ax_snr  = fig.add_subplot(gs[1, 2])

    panel_residual_maps(d, ax_rms, ax_map)
    panel_redshift(d, ax_z)
    panel_snr(d, ax_snr)

    fig.suptitle("Zero-shot Roman grism transfer: JWST-trained SR1+SR2 on mock Roman spectra",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out2 = OUT_DIR / "zeroshot_summary.png"
    plt.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")

    # ── Print summary stats ───────────────────────────────────────────────────
    rmask  = d["roman_wave_mask"].astype(bool)
    z      = d["z_true"]
    valid  = (z > 0) & (z < 8)
    dz_r   = np.abs(d["roman_zhat"][valid] - z[valid]) / (1 + z[valid])
    dz_j   = np.abs(d["jwst_zhat"][valid]  - z[valid]) / (1 + z[valid])

    print("\n── Zero-shot summary (Roman window only) ─────────────────────────")
    print(f"  Galaxies evaluated : {valid.sum()}")
    print(f"  Median |Δz|/(1+z)  : Roman={np.median(dz_r):.4f}  JWST={np.median(dz_j):.4f}")
    print(f"  Outlier rate (>0.15): Roman={( dz_r>0.15).mean():.1%}  JWST={(dz_j>0.15).mean():.1%}")

    hr    = d["flux_high_norm"][:, rmask]
    sr2_r = d["roman_sr2"][:, rmask]
    sr2_j = d["jwst_sr2"][:, rmask]
    rms_r = np.sqrt(np.nanmean((sr2_r - hr)**2))
    rms_j = np.sqrt(np.nanmean((sr2_j - hr)**2))
    print(f"  RMS residual (Roman window): Roman={rms_r:.4f}  JWST={rms_j:.4f}")
    print(f"  Domain gap factor  : {rms_r/rms_j:.2f}×  (1.0 = no gap)")


if __name__ == "__main__":
    main()
