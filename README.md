# Roman Grism Super-Resolution

Neural spectral super-resolution for the Nancy Grace Roman Space Telescope grism, transferred from a JWST NIRSpec prism–grating prototype.

## Overview

Roman's High Latitude Wide Area Survey (HLWAS) will deliver grism spectra at R~460–900 for ~10 million emission-line galaxies. At this resolution, Hα+[NII] blends are unresolved, introducing redshift-dependent velocity offsets that bias BAO scale measurements. This project applies a physics-informed, two-stage super-resolution framework — trained on JWST NIRSpec prism–grating pairs — to Roman grism inputs.

## Structure

```
abstract/       Conference abstract (Roman 2026)
data/           Wavelength grids and mock Roman spectra (generated; not committed)
models/         Model weights (symlinked or downloaded; not committed)
scripts/        Python scripts for mock data generation, inference, and plotting
results/        Output plots and summary statistics
```

## Zero-shot transfer

The first experiment runs the JWST-trained model on mock Roman spectra constructed by degrading
JWST grating spectra to Roman's resolution (R~600) and wavelength coverage (1.0–1.93 µm),
without any fine-tuning. This characterizes the domain gap before domain adaptation.

## Dependencies

```
torch, numpy, scipy, matplotlib, astropy
```

## Related

- JWST prototype: [super_resolution](../super_resolution)
- Roman proposal: Haghjoo et al., Roman Cycle 1 (2026)
