# Spectral Super-Resolution for the Roman Grism: Zero-Shot Transfer from a JWST NIRSpec Prototype

**Aryana Haghjoo** et al.

---

Roman's High Latitude Wide Area Survey will deliver grism spectroscopy at R~460–900 for
roughly ten million emission-line galaxies. At this resolution, the Hα+[NII] complex is
unresolved across z = 0.5–1.9, introducing redshift-dependent velocity offsets that propagate
into Roman's BAO and RSD measurements and degrade the spectroscopic calibration sets used
for weak-lensing photo-z validation. A survey-scale solution to this systematic does not yet
exist.

We present a physics-informed, two-stage neural framework for spectral super-resolution of
Roman grism spectra. The model is first trained on archival JWST NIRSpec prism–grating pairs
(R~100 → R~1000) from JADES and the NIRSpec Wide GTO Survey. A coarse super-resolution
stage encodes the low-resolution spectrum and produces a redshift estimate; a second stage
refines the reconstruction conditioned on that redshift and a rest-frame emission-line catalog,
ensuring spectral features are placed only at physically permitted wavelengths. Trained on ~25,000
JWST spectral pairs, the prototype achieves noise-limited residuals, a ~4× median signal-to-noise
improvement on Hα, a ~42% reduction in redshift error, and a ~22% reduction in catastrophic
outlier fraction on held-out JADES test spectra.

We present the first zero-shot transfer of this framework to Roman grism spectroscopy, applying
the JWST-trained model directly to simulated Roman grism spectra without fine-tuning.
Comparing reconstructions against JWST grating ground truth for the same galaxies, we
characterize the domain gap: where the JWST-trained model generalizes to Roman inputs, and
where the resolution mismatch and instrument differences require domain adaptation.
These results motivate a transfer strategy using published Roman grism simulations to adapt
the model before HLWAS data arrive, followed by progressive fine-tuning on real Roman–JWST
overlap fields in COSMOS and GOODS-S.

Applied survey-wide, the framework reconstructs Hα+[NII] for millions of galaxies, reducing
the dominant spectroscopic systematic on Roman's BAO scale while simultaneously enabling
resolved strong-line diagnostics at survey scale for galaxy evolution science.

---
*Submitted to: The Roman Space Telescope: Towards First Light and Beyond — Conference 2026*
