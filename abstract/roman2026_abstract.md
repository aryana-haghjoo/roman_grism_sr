# Full-Fidelity Grism Spectroscopy for Roman: A Physics-Informed Neural Super-Resolution Framework

**Aryana Haghjoo** et al.

---

Roman's High Latitude Wide Area Survey will deliver grism spectroscopy at R~460–900 for
roughly ten million emission-line galaxies. At this resolution, the Hα+[NII] complex is
unresolved across z = 0.5–1.9, introducing redshift-dependent velocity offsets that bias the
BAO scale and degrade spectroscopic calibration sets for weak-lensing photo-z validation.
A survey-scale solution does not yet exist.

We present a physics-informed, two-stage neural framework for spectral super-resolution of
Roman grism spectra. The model is trained on ~25,000 archival JWST NIRSpec prism–grating
pairs (R~100 → R~1000) from JADES and the NIRSpec Wide GTO Survey. A coarse
super-resolution stage jointly encodes the grism spectrum and broadband photometric SED to
produce a redshift estimate; a second stage refines the reconstruction conditioned on that
redshift and a rest-frame emission-line catalog, restricting spectral features to physically
permitted wavelengths. On held-out JWST test spectra, the prototype achieves noise-limited
residuals, a ~4× median S/N improvement on Hα, a ~42% reduction in redshift error, and a
~22% reduction in catastrophic outlier fraction.

We characterize the domain gap via zero-shot transfer to simulated Roman grism spectra,
identifying the redshift estimation stage as the primary failure mode under Roman's partial
wavelength coverage (0.99–1.95 µm). We then present a three-phase transfer strategy:
(1) simulation-based domain adaptation using published Roman grism simulations (Wang et al.
2022) with 1D spectra extracted via grizli, paired with synthetic JWST grating spectra of the
same Galacticus sources; (2) progressive fine-tuning on real Roman–JWST spectral pairs from
HLWAS Deep Tier overlap fields in COSMOS and GOODS-S as data arrive; and (3) survey-scale
application across the full ~2,400 deg² HLWAS footprint. Applied survey-wide, the framework
reconstructs Hα+[NII] for millions of galaxies, directly reducing the dominant spectroscopic
systematic on Roman's BAO scale and enabling resolved strong-line diagnostics at survey scale.

---
*Submitted to: The Roman Space Telescope: Towards First Light and Beyond — Conference 2026*
