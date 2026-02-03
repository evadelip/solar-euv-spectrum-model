import os
import pandas as pd
import numpy as np
import scipy.io
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

# Physical constants for unit conversion
SOLAR_SOLID_ANGLE = 6.8e-5  # sr, solar disk solid angle at 1 AU
HC_ERG_AA = 1.98644586e-8   # h*c in erg*Angstrom

#--- Functions to read data files ---

def data_1(file):
    df = pd.read_fwf(file, widths=[8, 10, 10, 10, 10, 10, 10, 10], skiprows=1,
                     names=['ion', 'wavelength', 'Int', 'Pmin', 'Imin', 'a', 'b', 'r'])

    df['ion'] = df['ion'].str.strip()
    return df

def data2(file):
    df = pd.read_fwf(file, widths=[8, 10, 10, 10, 10, 10, 10, 10, 3, 3], skiprows=1,
                     names=['ion', 'wavelength', 'Int', 'Pmin', 'Imin', 'a', 'b', 'r', 'lvl1', 'lvl2'],
                     dtype={'ion': str, 'wavelength': float, 'Int': float, 'Pmin': float, 'Imin': float,
                            'a': float, 'b': float, 'r': float, 'lvl1': int, 'lvl2': int})
    df['ion'] = df['ion'].str.strip()
    return df

def data3(file):
    widths = [8, 1, 9, 1, 2, 1, 2, 1, 5, 1, 10, 10, 10, 10, 10, 10]
    names = ['ion', 's1', 'wavelength', 's2', 'lvl1', 's3', 'lvl2', 's4', 'logt', 's5',
             'ratio0', 'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']
    df = pd.read_fwf(file, widths=widths, skiprows=2, names=names)
    df = df.drop(columns=['s1', 's2', 's3', 's4', 's5']) # Drop spacer columns
    df['ion'] = df['ion'].str.strip()
    df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce') # Convert types
    df['lvl1'] = pd.to_numeric(df['lvl1'], errors='coerce').astype('Int64')
    df['lvl2'] = pd.to_numeric(df['lvl2'], errors='coerce').astype('Int64')
    df['logt'] = pd.to_numeric(df['logt'], errors='coerce')
    for col in ['ratio0', 'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

#--- Function to calculate density and pressure from f30 ---

def calculate_density_and_pressure(f30):
    coeffs_fe13 = [2.0420288, 6.1597835, -1.4312914]
    coeffs_fe12 = [10.811510, -2.7931326, 0.83675505]

    log_ne_fe12 = np.polyval(coeffs_fe12[::-1], np.log10(f30))
    dens = 10 ** log_ne_fe12
    t_fe = 10 ** 6.15

    pressure = dens * t_fe
    print('Density: ', dens)
    print('Assumed constant pressure = ', pressure)
    return dens, pressure, t_fe

#--- Function to read abundance file and create abundance dictionary ---

def read_abundances(abund_file='python/sun_photospheric_2021_asplund.abund'):
    """
    Read CHIANTI abundance file and return a dictionary of abundances by element.
    Format: Element symbol, Z, log(N)+12
    """
    abundances = {}
    try:
        with open(abund_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    element = parts[0]  # First column is element symbol
                    z_num = int(parts[1])  # Second column is atomic number
                    abund_log = float(parts[2])  # Third column is log abundance
                    # Convert from log scale (relative to hydrogen = 12) to linear
                    abundances[element] = 10 ** (abund_log - 12.0)
                except (ValueError, IndexError):
                    continue
        print(f"Loaded {len(abundances)} element abundances from {abund_file}")
        return abundances
    except FileNotFoundError:
        print(f"Warning: Abundance file {abund_file} not found. Using solar defaults.")
        # Default Asplund 2021 abundances (relative to hydrogen)
        default_abundances = {
            'H': 1.0,
            'He': 0.0851,
            'Li': 1.58e-11,
            'Be': 1.61e-11,
            'B': 4.35e-10,
            'C': 2.69e-4,
            'N': 6.76e-5,
            'O': 4.90e-4,
            'F': 3.63e-8,
            'Ne': 3.44e-5,
            'Na': 1.73e-6,
            'Mg': 3.92e-5,
            'Al': 2.82e-6,
            'Si': 3.24e-5,
            'P': 2.57e-7,
            'S': 1.32e-5,
            'Cl': 1.04e-7,
            'Ar': 2.51e-6,
            'K': 9.77e-8,
            'Ca': 1.94e-6,
            'Sc': 8.12e-9,
            'Ti': 7.47e-8,
            'V': 7.51e-9,
            'Cr': 1.26e-7,
            'Mn': 4.00e-8,
            'Fe': 2.82e-5,
            'Co': 7.84e-9,
            'Ni': 1.62e-6,
            'Cu': 1.04e-8,
            'Zn': 3.63e-9
        }
        return default_abundances

#--- Function to extract element from ion name ---

def get_element_from_ion(ion_name):
    """
    Extract element symbol from ion name like 'Fe XII', 'He II', 'H I', etc.
    Returns the element symbol or None if not found.
    """
    ion_name = ion_name.strip()
    parts = ion_name.split()
    if len(parts) > 0:
        return parts[0]
    return None

#--- Read baseline lines CSV file ---

df = pd.read_csv("baseline_lines.csv", skipinitialspace=True)

#--- Load and apply abundances ---
# The baseline_lines.csv contains raw CHIANTI intensities (NO abundances)

abundances = read_abundances()

# Extract element from each ion and apply abundance correction (vectorized for speed)
df['element'] = df['ion'].str.extract(r'^(\w+)', expand=False)
abundance_factors = df['element'].map(abundances)

# Check for missing elements
missing_mask = abundance_factors.isna() & df['element'].notna()
if missing_mask.any():
    missing_ions = df.loc[missing_mask, 'ion'].unique()
    print(f"Warning: {len(missing_ions)} ions have elements not in abundance table - keeping original intensity:")
    for ion in missing_ions[:5]:
        print(f"  {ion}")
    if len(missing_ions) > 5:
        print(f"  ... and {len(missing_ions) - 5} more")

n_with_abund = abundance_factors.notna().sum()
df['intensity'] *= abundance_factors.fillna(1.0)
df = df.drop(columns=['element'])

print(f"\nApplied abundances to {n_with_abund} out of {len(df)} lines")

# Conversion from ergs to photons
# The baseline_lines.csv intensities are in erg cm^-2 s^-1 sr^-1
# To convert to photons: multiply by wavelength / (h*c)
# where h*c = 1.98644586e-8 erg*Angstrom
print("\nConverting from ergs to photons...")
df['intensity'] = df['intensity'] * df['wavelength_A'] / HC_ERG_AA
print(f"Total intensity sum after erg→photon conversion: {df['intensity'].sum():.3e}")

#--- Apply transition-specific corrections (2nd list) ---

f30_input = input("Enter f30 value (from 40 for QS to 180 for AS): ").strip()
f30 = float(f30_input) if f30_input else 46.0

data2_df = data2('output_2nd_list.txt')

handled = np.zeros(len(df), dtype=int)

for jj in range(len(data2_df)):
    condition = (df['ion'].str.strip() == data2_df.loc[jj, 'ion'].strip()) & \
                (df['lvl1'] == data2_df.loc[jj, 'lvl1']) & \
                (df['lvl2'] == data2_df.loc[jj, 'lvl2'])
    ind = df[condition].index
    nn = len(ind)
    if nn > 0:
        correction = float(data2_df.loc[jj, 'r']) * (1 + data2_df.loc[jj, 'b'] * (f30 / data2_df.loc[jj, 'Pmin'] - 1))
        df.loc[ind, 'intensity'] *= correction
        handled[ind] = 1

#--- Apply ion-level corrections to remaining transitions (1st list) ---

data1_df = data_1('output_list_newdem.txt')

# Now apply ion-level correction to all transitions from data1,
# but only if they were not already corrected above
for ii in range(len(data1_df)):
    ind = df[(df['ion'].str.strip() == data1_df.loc[ii, 'ion'].strip()) & (handled == 0)].index
    nn = len(ind)
    if nn == 0:
        raise ValueError('Error, no unhandled lines found for ion: ' + data1_df.loc[ii, 'ion'].strip())

    correction = float(data1_df.loc[ii, 'r']) * (1 + data1_df.loc[ii, 'b'] * (f30 / data1_df.loc[ii, 'Pmin'] - 1))
    df.loc[ind, 'intensity'] *= correction

#--- Apply density correction using emissivity ratios from data3 ---

# Calculate density and pressure
dens, pressure, temp = calculate_density_and_pressure(f30)

dens_grid = [1.00000e+08, 1.58489e+08, 2.51188e+08, 3.98108e+08, 6.30958e+08, 1.00000e+09]

data3_df = data3('output_3rd_list_1107.txt')

snote_trimmed = df['ion'].str.strip()

for kk in range(len(data3_df)):
    # Calculate T_formation for this ion
    t_ion = 10 ** data3_df.loc[kk, 'logt']  # logT from the file
    ne_ion = pressure / t_ion  # scaled density
    log_ne_ion = np.log10(ne_ion)

    ratio_array = data3_df.loc[kk, ['ratio0', 'ratio1', 'ratio2', 'ratio3', 'ratio4', 'ratio5']].values.astype(float)
    ratio_interp = np.interp(ne_ion, dens_grid, ratio_array)

    ind = df[(snote_trimmed == data3_df.loc[kk, 'ion'].strip()) & \
             (df['lvl1'] == data3_df.loc[kk, 'lvl1']) & \
             (df['lvl2'] == data3_df.loc[kk, 'lvl2'])].index
    nn = len(ind)

    if nn == 1:
        ind_ref = data3_df[(data3_df['ion'].str.strip() == data3_df.loc[kk, 'ion'].strip()) & \
                           (data3_df['ratio0'] == 1.00)].index
        ind_res = df[(snote_trimmed == data3_df.loc[kk, 'ion'].strip()) & \
                     (df['lvl1'] == data3_df.loc[ind_ref[0], 'lvl1']) & \
                     (df['lvl2'] == data3_df.loc[ind_ref[0], 'lvl2'])].index
        df.loc[ind[0], 'intensity'] = df.loc[ind_res[0], 'intensity'] * ratio_interp
    else:
        print('Problem for: ', data3_df.loc[kk, 'ion'].strip(), '-', data3_df.loc[kk, 'wavelength'])
        raise ValueError("Problem with matching lines")

#--- Define wavelength grid for model spectrum ---
wave = np.arange(6.0, 106.0, 0.1)  # nm
wave_ang = 10.0 * wave
print(f"\nWavelength grid: {wave_ang[0]:.1f} - {wave_ang[-1]:.1f} Å ({len(wave_ang)} points)")

#--- Make Spectrum for the Model ---

def create_spectrum_from_lines(df, wave_grid, fwhm):
    """
    Create a synthetic spectrum by placing Gaussians at each line position.

    Parameters:
    - df: DataFrame with columns 'wavelength_A' and 'intensity'
    - wave_grid: wavelength grid to calculate spectrum on
    - fwhm: Full Width Half Maximum for Gaussian line profiles (in Angstroms)

    Returns:
    - spectrum: array of intensities at each wavelength in wave_grid
    """
    spectrum = np.zeros(len(wave_grid))
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # Convert FWHM to sigma

    # Normalization factor for Gaussian: peak = integral / (sigma * sqrt(2*pi))
    norm_factor = sigma * np.sqrt(2.0 * np.pi)

    for idx, row in df.iterrows():
        line_wave = row['wavelength_A']
        line_int = row['intensity']

        # Create normalized Gaussian profile where the integral equals line_int
        # Peak amplitude = line_int / (sigma * sqrt(2*pi))
        gaussian = (line_int / norm_factor) * np.exp(-0.5 * ((wave_grid - line_wave) / sigma) ** 2)
        spectrum += gaussian

    return spectrum

print("\nCreating model spectrum from baseline lines...")

mask = (df['wavelength_A'] >= 60) & (df['wavelength_A'] <= 1040)
df_filtered = df[mask].copy()
print(f"Wavelength range (60-1040 Å): {len(df_filtered)} lines")
model_spectrum = create_spectrum_from_lines(df_filtered, wave_ang, fwhm=2.5)

model_spectrum = model_spectrum / 1e8

# Continua calculations
wave1 = np.arange(450, 504 + 0.2, 0.2)  # He continuum: 450-504 Å
wave2 = np.arange(750, 912 + 0.2, 0.2)  # Lyman 2: 750-912 Å
wave3 = np.arange(635, 775 + 0.2, 0.2)  # Lyman 1: 635-775 Å

ind1 = (wave_ang > 460) & (wave_ang <= 504)    # He continuum
ind2 = (wave_ang > 775) & (wave_ang <= 912)    # Lyman 2
ind3 = (wave_ang > 635) & (wave_ang <= 775)    # Lyman 1

# Lyman 2 continuum (750-912 Å)
c0 = 1.10e1 - 9.37e-3 * f30 + 1.33e-5 * f30**2
c1 = -2.756e-02 - 7.70e-7 * f30 + 3.25e-09 * f30**2
c2 = 1.73e-5 + 1.64e-8 * f30 - 2.50e-11 * f30**2
cont_lym2_grid = c0 + wave2 * c1 + wave2**2 * c2
cont_lym2_grid = cont_lym2_grid / 0.2
cont_lym2 = np.interp(wave_ang[ind2], wave2, cont_lym2_grid)

# Lyman 1 continuum (635-775 Å)
c0 = -3.4e-2 - 3.04e-4 * f30 - 3.36e-7 * f30**2
c1 = -4.5e-5 - 4.07e-20 * f30 + 1.06e-22 * f30**2
c2 = 1.57e-7 + 8.68e-10 * f30 + 3.20e-13 * f30**2
cont_lym1_grid = c0 + wave3 * c1 + wave3**2 * c2
cont_lym1_grid = cont_lym1_grid / 0.2
cont_lym1_grid = cont_lym1_grid - 0.03
cont_lym1 = np.interp(wave_ang[ind3], wave3, cont_lym1_grid)

# He continuum (450-504 Å)
c0 = 1.18e1 - 4.57e-3 * f30 + 2.19e-6 * f30**2
c1 = -4.95e-2 - 5.55e-17 * f30 + 1.09e-19 * f30**2
c2 = 5.19e-5 + 2.21e-8 * f30 - 1.24e-11 * f30**2
cont_he_grid = c0 + wave1 * c1 + wave1**2 * c2
cont_he_grid = cont_he_grid / 0.2
cont_he = np.interp(wave_ang[ind1], wave1, cont_he_grid)

model_spectrum[ind3] = model_spectrum[ind3] + cont_lym1
model_spectrum[ind2] = model_spectrum[ind2] + cont_lym2
model_spectrum[ind1] = model_spectrum[ind1] + cont_he

print(f"Model spectrum created successfully!")
print(f"  Wavelength range: {wave_ang[0]:.2f} - {wave_ang[-1]:.2f} Å")
print(f"  Number of spectral lines: {len(df_filtered)}")
print(f"  Model spectrum range: {np.min(model_spectrum):.3e} to {np.max(model_spectrum):.3e}")

#--- Plotting ---

ranges = np.array([
    [60, 110], [110, 170], [170, 215], [215, 300], [300, 310],
    [310, 380], [380, 515], [515, 635], [635, 782],
    [782, 912], [912, 1050]
])

plt.rcParams.update({'font.size': 14})

for i in range(ranges.shape[0]):
    xrange = [ranges[i, 0], ranges[i, 1]]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wave_ang, model_spectrum, drawstyle='steps-mid', linewidth=2, color='red', label='Model')
    ax.set_xlim(xrange)

    mask_range = (wave_ang >= xrange[0]) & (wave_ang <= xrange[1])
    if np.any(mask_range):
        y_model = model_spectrum[mask_range]
        y_max = np.max(y_model)
        y_min = np.min(y_model)
        y_padding = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_xlabel('Wavelength (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel('10$^8$ phot cm$^{-2}$ s$^{-1}$ Å$^{-1}$', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, frameon=True)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.tick_params(width=2, length=6, labelsize=12)

    ax.set_title(f'Model Spectrum: {xrange[0]}-{xrange[1]} Å (F30={int(f30)})', fontsize=12)
    plt.tight_layout()
    plt.show()
