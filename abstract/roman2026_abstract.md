# Full-Fidelity Grism Spectroscopy for Roman: A Physics-Informed Neural Super-Resolution Framework

**Aryana Haghjoo** et al.

---

The Roman Space Telescope's High Latitude Wide Area Survey will deliver grism spectroscopy at R~460–900 for roughly ten million emission-line galaxies, but at this resolution key diagnostic features such as the Hα+[N II] complex remain unresolved across a wide redshift range, introducing systematic biases in redshift measurements and derived physical quantities. Assembling the survey-scale high-resolution spectroscopic data needed to overcome this limitation is observationally prohibitive. We introduce a three-stage physics-informed deep-learning framework for spectral super-resolution, trained on 1,187 paired JWST/NIRSpec prism–grating observations from JADES, that enhances low-resolution galaxy spectra by a factor of 10 in resolving power (R~100 to R~1000). The model infers a redshift from a coarse super-resolved intermediate, then applies a residual refinement stage that uses multi-head self-attention across emission-line tokens to learn inter-line relationships and predict physically interpretable line profiles. We then benchmark this pipeline against seven classical deconvolution methods, demonstrating a 30% reduction in global reconstruction error, superior line detectability and width recovery. Here we characterize the domain gap between JWST NIRSpec and Roman grism spectroscopy and present a transfer strategy. Applied to simulated Roman grism observations, the adapted framework reconstructs Hα+[N II] for millions of galaxies, enabling resolved strong-line diagnostics and reducing the dominant spectroscopic systematic on Roman's BAO scale.

---
*Submitted to: The Roman Space Telescope: Towards First Light and Beyond — Conference 2026*
