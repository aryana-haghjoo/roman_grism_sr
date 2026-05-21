#!/usr/bin/env python3
"""
make_roman_spectra.py

Forward-model JWST grating spectra through the Roman WFI grism response
to produce simulated Roman grism 1D spectra.

Inputs:
  - JWST prism+grating dataset from super_resolution repo
  - Roman grism 1st-order throughput from romanisim

Outputs (depending on --split):
  - data/roman_mock_spectra.npz   (--split test, default)
  - data/roman_train_spectra.npz  (--split train)

Each file contains:
    roman_flux       : (N, 2500) Roman grism spectrum on JWST grid, zeroed outside Roman range
    roman_flux_err   : (N, 2500) noise estimate on same grid
    roman_wave_mask  : (2500,)   bool mask of Roman wavelength coverage
    flux_high        : (N, 2500) JWST grating truth (HR target)
    flux_high_err    : (N, 2500)
    flux_low         : (N, 2500) JWST prism (for comparison)
    wavelength_high  : (2500,)   wavelength grid in microns
    z                : (N,)      redshifts
    split_indices    : (N,)      indices into original dataset

Usage:
    python make_roman_spectra.py             # test split (default)
    python make_roman_spectra.py --split train
"""

import os
import sys
import argparse
import hashlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ---- paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SR_REPO = os.path.join(os.path.dirname(REPO), "super_resolution")
DATASET = os.path.join(SR_REPO, "data", "spectra_dataset_2500.npz")
SPLIT_DIR = os.path.join(SR_REPO, "splits")

os.makedirs(os.path.join(REPO, "data"), exist_ok=True)

# ---- Roman grism throughput from romanisim ----
def load_roman_grism_throughput():
    from romanisim.models import bandpass as bp_mod
    bp = bp_mod.getBandpass("Grism_1stOrder", sca=1)
    wave_nm = np.array(bp.wave_list)           # nanometers
    wave_um = wave_nm * 1e-3                   # microns
    throughput = np.array([bp(w) for w in wave_nm])
    return wave_um, throughput


def resolve_roman_lsf_sigma_pixels(wave_um_roman, R_func=None):
    """
    Gaussian sigma in pixels of the Roman grism LSF at each wavelength.
    R varies from ~461 (blue) to ~887 (red); we use a linear interpolation.
    Returns sigma in pixels of the Roman wavelength grid.
    """
    # R(lambda) linear fit from Gong et al. 2020 Table 1 (approx)
    wave_min, wave_max = wave_um_roman.min(), wave_um_roman.max()
    R_min, R_max = 461.0, 887.0
    R_wave = R_min + (R_max - R_min) * (wave_um_roman - wave_min) / (wave_max - wave_min)

    # pixel spacing in microns
    dwave = np.median(np.diff(wave_um_roman))

    # FWHM in microns: delta_lambda = lambda / R
    fwhm_um = wave_um_roman / R_wave
    sigma_um = fwhm_um / (2.355)
    sigma_px = sigma_um / dwave
    return sigma_px


def forward_model_roman(flux_hi, wave_jwst_um, wave_roman_um, throughput_roman,
                        exptime_s=626.0, area_cm2=4.5e4, read_noise_e=5.0,
                        seed=None):
    """
    Forward-model a JWST grating spectrum through the Roman grism.

    Parameters
    ----------
    flux_hi        : (L,) JWST grating flux (arbitrary normalised units)
    wave_jwst_um   : (L,) JWST wavelength grid in microns
    wave_roman_um  : (M,) Roman grism wavelength grid in microns
    throughput_roman: (M,) Roman grism throughput
    exptime_s      : Roman grism exposure time per dither (seconds)
    area_cm2       : Roman WFI collecting area (cm^2)
    read_noise_e   : read noise in electrons
    seed           : random seed for reproducibility

    Returns
    -------
    roman_flux_norm  : (M,) Roman grism spectrum (normalised, same units as input)
    roman_err_norm   : (M,) uncertainty on same scale
    """
    rng = np.random.default_rng(seed)

    # 1. Interpolate JWST grating onto Roman wavelength grid
    #    Only interpolate within the JWST wavelength coverage
    in_range = (wave_roman_um >= wave_jwst_um.min()) & (wave_roman_um <= wave_jwst_um.max())
    f_interp = interp1d(wave_jwst_um, flux_hi, kind="linear",
                        bounds_error=False, fill_value=0.0)
    roman_flux = f_interp(wave_roman_um)
    roman_flux[~in_range] = 0.0

    # 2. Convolve with Roman grism LSF
    #    JWST grating is already at R~1000; Roman grism is R~461-887.
    #    Effective broadening sigma: sqrt(sigma_roman^2 - sigma_jwst^2)
    sigma_roman_px = resolve_roman_lsf_sigma_pixels(wave_roman_um)
    dwave_roman = np.median(np.diff(wave_roman_um))
    dwave_jwst  = np.median(np.diff(wave_jwst_um))
    # JWST grating LSF sigma in Roman pixels (after interpolation)
    R_jwst = 1000.0
    fwhm_jwst_um = wave_roman_um / R_jwst
    sigma_jwst_px = (fwhm_jwst_um / 2.355) / dwave_roman

    # Effective broadening needed (add in quadrature where Roman is broader)
    sigma_eff_px = np.where(
        sigma_roman_px > sigma_jwst_px,
        np.sqrt(np.clip(sigma_roman_px**2 - sigma_jwst_px**2, 0, None)),
        0.0,
    )
    # Use median effective sigma for a single Gaussian pass
    sigma_median = float(np.median(sigma_eff_px[in_range])) if in_range.any() else 0.0
    if sigma_median > 0.1:
        roman_flux = gaussian_filter1d(roman_flux, sigma=sigma_median)

    # 3. Apply throughput (relative weighting; keeps flux in input units)
    roman_flux_through = roman_flux * throughput_roman

    # 4. Noise model: Poisson + read noise
    #    Scale flux to approximate electron counts for noise estimation only
    flux_positive = np.clip(roman_flux_through, 0, None)
    # Use a reference SNR: assume median flux gives ~20 e/s at the peak
    peak = flux_positive.max()
    if peak > 0:
        scale = 20.0 * exptime_s / peak
    else:
        scale = 1.0
    signal_e = flux_positive * scale
    noise_e = np.sqrt(signal_e + read_noise_e**2)
    noise_norm = noise_e / scale

    # Add noise
    roman_flux_noisy = roman_flux_through + rng.normal(0, noise_norm)

    # 5. Renormalize to match the input flux scale (undo throughput scaling)
    tp_med = float(np.median(throughput_roman[in_range])) if in_range.any() else 1.0
    if tp_med > 0:
        roman_flux_noisy /= tp_med
        noise_norm /= tp_med

    return roman_flux_noisy.astype(np.float32), noise_norm.astype(np.float32)


def embed_in_jwst_grid(roman_flux, roman_err, wave_roman_um, wave_jwst_um):
    """
    Interpolate the Roman spectrum (M points) onto the full JWST 2500-point
    grid, zeroing out wavelengths outside the Roman coverage.

    Returns
    -------
    out_flux  : (2500,)
    out_err   : (2500,)
    mask      : (2500,) bool — True where Roman data exists
    """
    roman_min = wave_roman_um.min()
    roman_max = wave_roman_um.max()
    mask = (wave_jwst_um >= roman_min) & (wave_jwst_um <= roman_max)

    f_interp = interp1d(wave_roman_um, roman_flux, kind="linear",
                        bounds_error=False, fill_value=0.0)
    e_interp = interp1d(wave_roman_um, roman_err, kind="linear",
                        bounds_error=False, fill_value=0.0)

    out_flux = np.zeros(len(wave_jwst_um), dtype=np.float32)
    out_err  = np.zeros(len(wave_jwst_um), dtype=np.float32)
    out_flux[mask] = f_interp(wave_jwst_um[mask])
    out_err[mask]  = e_interp(wave_jwst_um[mask])

    return out_flux, out_err, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="Which dataset split to forward-model (default: test)")
    args = ap.parse_args()

    out_path = os.path.join(
        REPO, "data",
        "roman_mock_spectra.npz" if args.split == "test" else "roman_train_spectra.npz"
    )

    print("Loading JWST dataset...")
    data = np.load(DATASET, allow_pickle=True)
    flux_low      = data["flux_low"]
    flux_high     = data["flux_high"]
    flux_high_err = data["flux_high_err"]
    wave_jwst     = data["wavelength_high"].astype(np.float32)
    z_all         = data["z"]

    # Load the same train/test split used in SR1 training
    with open(DATASET, "rb") as f:
        ds_hash = hashlib.md5(f.read()).hexdigest()
    split_path = os.path.join(SPLIT_DIR, f"split_{ds_hash}.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Run the SR1 training script first to generate it."
        )
    split     = np.load(split_path)
    split_idx = split["train_idx"] if args.split == "train" else split["test_idx"]
    print(f"{args.split.capitalize()} split: {len(split_idx)} galaxies")

    print("Loading Roman grism throughput from romanisim...")
    wave_roman_um, throughput_roman = load_roman_grism_throughput()
    print(f"Roman grism: {wave_roman_um.min():.3f} – {wave_roman_um.max():.3f} µm "
          f"({len(wave_roman_um)} pixels)")

    print(f"Forward-modelling {args.split} set through Roman grism...")
    N = len(split_idx)
    roman_on_jwst_grid = np.zeros((N, 2500), dtype=np.float32)
    roman_err_on_grid  = np.zeros((N, 2500), dtype=np.float32)

    for i, idx in enumerate(split_idx):
        if i % 500 == 0:
            print(f"  {i}/{N}")
        roman_flux, roman_err = forward_model_roman(
            flux_high[idx].astype(np.float64), wave_jwst, wave_roman_um, throughput_roman,
            seed=int(idx)
        )
        out_flux, out_err, _ = embed_in_jwst_grid(
            roman_flux, roman_err, wave_roman_um, wave_jwst
        )
        roman_on_jwst_grid[i] = out_flux
        roman_err_on_grid[i]  = out_err

    # Roman wavelength mask (same for all galaxies)
    _, _, roman_wave_mask = embed_in_jwst_grid(
        np.ones(len(wave_roman_um)), np.ones(len(wave_roman_um)),
        wave_roman_um, wave_jwst
    )

    print(f"\nRoman wavelength coverage on JWST grid: "
          f"{roman_wave_mask.sum()} / 2500 pixels "
          f"({100*roman_wave_mask.mean():.1f}%)")

    print(f"Saving to {out_path}")
    np.savez(
        out_path,
        roman_flux=roman_on_jwst_grid,
        roman_flux_err=roman_err_on_grid,
        roman_wave_mask=roman_wave_mask,
        flux_high=flux_high[split_idx],
        flux_high_err=flux_high_err[split_idx],
        flux_low=flux_low[split_idx],
        wavelength_high=wave_jwst,
        wave_roman_um=wave_roman_um,
        throughput_roman=throughput_roman,
        z=z_all[split_idx],
        split_indices=split_idx,
    )
    print("Done.")


if __name__ == "__main__":
    main()
