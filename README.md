# Roman Grism Super-Resolution

Neural spectral super-resolution for the Nancy Grace Roman Space Telescope grism,
transferred from a working JWST NIRSpec prism–grating prototype.

---

## Background and motivation

Roman's High Latitude Wide Area Survey (HLWAS) will deliver grism spectroscopy at
R~460–900 for ~10 million emission-line galaxies across z=0.5–2.5. At this resolution
the Hα+[NII] complex is unresolved, introducing redshift-dependent velocity offsets
that bias Roman's BAO scale measurements and degrade the spectroscopic calibration
sets for weak-lensing photo-z validation.

This project transfers a physics-informed spectral super-resolution framework —
already trained and validated on JWST NIRSpec prism–grating pairs — to Roman grism
inputs. A Roman Cycle 1 proposal for this work has been submitted (Haghjoo et al. 2026).
A conference abstract has been submitted to *The Roman Space Telescope: Towards First
Light and Beyond* (July 2026).

---

## Parent project (JWST prototype)

The upstream codebase is `../super_resolution`. That repo contains:

- **SR1** (`train/sr1_best/train_sr1.py`): a 1D ResNet that maps JWST NIRSpec
  prism spectra (R~100, 1–5 µm, 2500 pixels) to grating-resolution spectra
  (R~1000). Trained on ~25,000 JADES + NIRSpec Wide GTO spectral pairs.
  Uses a heteroscedastic NLL loss with a gated line-sharpness penalty.
  Best weights: `train/sr1_best/best_superres_model.pth`.

- **Z-head** (`train/redshift_head/`): a small 1D CNN that takes the SR1 output
  and estimates redshift. Best weights: `train/redshift_head/best_zhead.pth`.

- **SR2** (`train/sr2_best/train_sr2_attention.py`): a residual refiner
  conditioned on the SR1 output, estimated redshift, and a Gaussian line mask
  built from a rest-frame emission-line catalog. The line mask restricts spectral
  features to physically permitted wavelengths. Uses attention layers.
  Best weights: `train/sr2_best/best_sr2_attn.pth`.

- **Dataset**: `data/spectra_dataset_2500.npz` — 24,927 JADES galaxies,
  wavelength grid 1–5 µm, 2500 pixels. Keys: `flux_low`, `flux_high`,
  `flux_high_err`, `wavelength_low`, `wavelength_high`, `z`.
  Train/test split (80/20) saved in `splits/`.

**JWST prototype results** (held-out test set):
- Noise-limited residuals (SR−HR structureless)
- ~4× median S/N improvement on Hα
- ~42% reduction in redshift error
- ~22% reduction in catastrophic outlier fraction

**Architecture note**: `best_sr2_attn.pth` was saved with older attribute names.
When loading, remap: `line_rest_um_buf→line_rest_um`, `wave_hi_um_buf→wave_hi_um`,
`cnn_initial→cnn_in`, `cnn_delta→cnn_out`.

---

## This repo: Roman transfer

### Strategy

Three-phase transfer (mirrors the proposal):

1. **Zero-shot** (done): apply the JWST-trained model to mock Roman grism spectra
   without any fine-tuning, to quantify the domain gap.
2. **Simulation-based domain adaptation** (next): fine-tune on matched Roman grism /
   JWST grating pairs from published Roman grism simulations (Wang et al. 2022 at
   IRSA, DOI: 10.26131/IRSA548). Roman grism simulations are generated using
   **romanisim** (the official STScI Roman image simulator, pip-installable).
3. **Fine-tuning on real data**: once HLWAS Deep Tier data (COSMOS, GOODS-S) arrive,
   cross-match with JWST observations to build real Roman–JWST pairs.

### Photometric conditioning decision

The multimodal (spectra + photometry) approach was tested in `../multimodal_superresolution`
using JADES data. Result: adding Roman WFI broadband photometry (F106/F129/F158)
**did not improve spectral reconstruction quality**, only reduced the outlier rate on
redshift prediction. Decision: **use spectra only** for the SR model. Photometric
redshift information may still be valuable as an external prior for the z-head,
but not as a second encoder branch.

---

## Zero-shot experiment (completed)

### How the mock Roman spectra are made

Script: `scripts/make_roman_spectra.py`

1. Load the JWST test set galaxies from `../super_resolution/data/spectra_dataset_2500.npz`
   (same 80/20 split used in SR1 training).
2. Load the Roman WFI `Grism_1stOrder` throughput curve from **romanisim**
   (wavelength range 0.992–1.954 µm, 226 pixels).
3. For each galaxy: interpolate the JWST grating spectrum onto the Roman wavelength
   grid, convolve with the Roman LSF (Gaussian, R~461–887 from Gong et al. 2020
   linearly interpolated), apply throughput, add Poisson + read noise.
4. Embed the Roman spectrum back onto the 2500-pixel JWST grid (zeros outside
   Roman coverage). Roman coverage: 597/2500 pixels (24% of the JWST grid).

Output: `data/roman_mock_spectra.npz`

### Inference

Script: `scripts/zeroshot_inference.py`

Runs SR1 → z-head → SR2 on the mock Roman spectra using the JWST-trained weights
(no fine-tuning). Also runs the JWST prism baseline (same pipeline, original prism
input) for direct comparison.

**Normalization**: the Roman input is normalized using only the pixels within the
Roman wavelength window, then embedded in zeros — avoids the mean/std being
dominated by the zero-padded region.

Uses `../super_resolution/sup_res` virtualenv (Python 3.11).

### Key findings

Script: `scripts/plot_zeroshot.py` → `results/zeroshot_examples.png`, `results/zeroshot_summary.png`

| Metric | JWST prism (in-domain) | Roman grism (zero-shot) |
|--------|------------------------|------------------------|
| Median \|Δz\|/(1+z) | 0.10 | 0.46 |
| Outlier rate (>0.15) | 37% | 81% |
| RMS residual (Roman window) | 1.59 | 1.59 |
| SR gain vs input | 1.22× | 0.88× |

**Interpretation**: The domain gap is almost entirely in the **z-head**, not in the
reconstruction network. The SR2 produces smooth but physically reasonable spectra
even with Roman input (RMS gap ≈ 1.0×). It fails to recover line structure because
the z-head — trained on full 1–5 µm spectra — cannot correctly place emission lines
from 24% wavelength coverage. The wrong redshift estimate misdirects the SR2 line
mask, degrading reconstruction in the Roman window.

**What this means for domain adaptation**: the primary target is the z-head.
Options: (a) train a Roman-specific z-head on the 1–1.95 µm window only, or
(b) condition on an external photometric redshift prior. Option (b) aligns with
the proposal's photo-z conditioning idea, but now motivated by the zero-shot
result rather than by the multimodal SED experiment.

---

## Repo structure

```
abstract/
  roman2026_abstract.md   Conference abstract (submitted May 2026)

scripts/
  make_roman_spectra.py   Forward-model JADES galaxies through Roman grism response
  zeroshot_inference.py   Run JWST-trained SR1+z-head+SR2 on mock Roman spectra
  plot_zeroshot.py        Diagnostic plots for zero-shot experiment

data/                     (not committed — generated locally)
  roman_mock_spectra.npz  Mock Roman grism spectra for JADES test set

results/                  (not committed — generated locally)
  zeroshot_examples.png   Example spectra: Roman input / SR2 / JWST truth
  zeroshot_summary.png    Residual maps, redshift scatter, SNR gain

models/                   (not committed — symlink or copy weights here)
```

---

## Environment setup

```bash
# New venv for this project
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" "astropy>=7" scipy matplotlib torch romanisim

# For running inference, use the parent project's venv instead:
source ../super_resolution/sup_res/bin/activate
```

The inference scripts use `../super_resolution/sup_res` because that venv has the
trained model code on its path and matching PyTorch.

---

## What's next

1. **Roman-specific z-head**: retrain the z-head on a 1–1.95 µm truncated version
   of the JADES spectra to close the redshift domain gap.
2. **Simulation-based domain adaptation**: use Wang et al. 2022 Roman grism
   simulations (2D images at IRSA) + grizli extraction to get real Roman grism
   1D spectra, then fine-tune SR1+SR2.
3. **Conference talk (July 2026)**: present zero-shot results as the domain gap
   characterization, show the Roman-specific z-head fix, outline the transfer plan.

---

## Related

- JWST prototype: [../super_resolution](../super_resolution)
- Multimodal experiment (photometry + spectra, JADES only): [../multimodal_superresolution](../multimodal_superresolution)
- Roman proposal: Haghjoo et al., Roman Cycle 1 (2026)
- Roman grism throughput: `romanisim.models.bandpass.getBandpass("Grism_1stOrder")`
- Wang et al. 2022 Roman grism simulations: IRSA DOI 10.26131/IRSA548
