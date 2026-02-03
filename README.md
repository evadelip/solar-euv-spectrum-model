# Solar EUV Spectrum Synthesis Model

A Python-based spectral synthesis tool for modeling solar extreme ultraviolet (EUV) spectra using CHIANTI atomic database and f30 radio flux diagnostics.

## Features

- **CHIANTI-based**: Uses baseline spectral lines from CHIANTI atomic database
- **f30-dependent corrections**: Applies transition and ion-level corrections based on solar f30 radio flux (40-180)
- **Density diagnostics**: Calculates electron density from Fe XII/XIII emission ratios
- **Gaussian line profiles**: Convolves spectral lines with configurable FWHM (default 2.5 Å)
- **Multi-band plotting**: Generates 11 wavelength-range plots (60-1040 Å)
- **Continua modeling**: Includes Lyman 1/2 and He continuum components

## Installation

### Requirements
- Python 3.9+
- pandas
- numpy
- scipy
- matplotlib

### Setup

```bash
pip install pandas numpy scipy matplotlib
```

## Usage

Run the spectrum synthesis:

```bash
python final_model.py
```

When prompted, enter an f30 value:
- **40-50**: Quiet Sun (QS)
- **60-100**: Active Region (AR)
- **140-180**: Active Sun (AS)

The script will:
1. Load baseline spectral lines (CHIANTI format)
2. Apply solar photospheric abundances (Asplund 2021)
3. Apply f30-dependent corrections
4. Calculate electron density from pressure diagnostics
5. Apply density-dependent emissivity corrections
6. Generate and display 11 spectral plots

## Input Files Required

- `baseline_lines.csv` - CHIANTI spectral lines with intensities, wavelengths, and level information
- `output_2nd_list.txt` - Transition-level f30 correction coefficients
- `output_list_newdem.txt` - Ion-level f30 correction coefficients
- `output_3rd_list_1107.txt` - Density-dependent emissivity ratios
- `sun_photospheric_2021_asplund.abund` - Solar photospheric abundances (optional - has built-in defaults)

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| FWHM | 2.5 Å | Gaussian line profile width |
| Wavelength range | 60-1040 Å | EUV spectral region |
| Temperature | 10^6.15 K | Constant temperature assumption |
| f30 input | 40-180 | Solar f30 radio flux index |

## Output

The script generates 11 sequential matplotlib plots displaying model spectra across different wavelength ranges. Plots show only the model spectrum (red) in 10^8 photons cm^-2 s^-1 Å^-1 units.

## Physical Model

### Density Calculation
Electron density is derived from Fe XII/XIII emission line ratios:

$$\log_{10}(n_e) = 10.811510 - 2.793 \log_{10}(f30) + 0.837 \log_{10}^2(f30)$$

### Line Corrections
Spectral line intensities are corrected based on:
- **Transition-level**: f30-dependent corrections for specific level transitions
- **Ion-level**: f30-dependent corrections applied to remaining ion transitions
- **Density-level**: Emissivity ratio corrections interpolated from density grid

### Continua
Three continuum components with f30-dependent polynomial coefficients:
- Lyman 1 (635-775 Å)
- Lyman 2 (750-912 Å)
- He continuum (450-504 Å)

## Author

Eva Delia

## References

- Asplund, M., et al. (2021) - Solar photospheric abundances
- CHIANTI - Atomic database for astrophysics

## License

MIT
