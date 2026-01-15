import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

H_0 = cosmo.H0  # Hubble constant
c_kms = c.to('km/s').value  # speed of light in km/s
q0 = -0.534  # deceleration parameter (assumed)

# Opening FITS file and extracting data
file = fits.open('C:/Users/Welcome-Pc/OneDrive/Desktop/python/cluster.fits')
data = file[1].data

def to_native(arr):#function to check whether byteorder is correct
    if arr.dtype.byteorder not in ('=', '|'):
        return arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr

# Extracting columns with correct byte order
specz = to_native(data['specz'])
objid = to_native(data['objid'])
ra = to_native(data['ra'])
dec = to_native(data['dec'])
proj_sep = to_native(data['proj_sep'])
rmag = to_native(data['rmag'])
gmag = to_native(data['gmag'])

# Creating DataFrame
df = pd.DataFrame({
    'specz': specz,
    'objid': objid,
    'ra': ra,
    'dec': dec,
    'proj_sep': proj_sep,
    'gmag': gmag,
    'rmag': rmag
})

# Grouping and averaging values by objid
averaged_df = df.groupby('objid').agg({
    'specz': 'mean',
    'rmag': 'mean',
    'gmag': 'mean',
    'ra': 'first',
    'dec': 'first',
    'proj_sep': 'first'
}).reset_index()

# Histogram of original redshift distribution
plt.hist(averaged_df['specz'], bins=90)
plt.grid()
plt.title("Distribution of redshift for original data")
plt.xlabel("Redshift (specz)")
plt.ylabel("Number of galaxies")
plt.show()

# Initial filtering by redshift outliers like 0.15 so that standard deviation does not get affected
filtered_df = averaged_df[(averaged_df['specz'] <= 0.095)].copy()
mean_z = filtered_df['specz'].mean()
std_z = filtered_df['specz'].std()
lower_limit = mean_z - 3 * std_z
upper_limit = mean_z + 3 * std_z

# Histogram after first filtering with 3-sigma lines
plt.hist(filtered_df['specz'], bins=90)
plt.grid()
plt.axvline(lower_limit, color='r', linestyle='--', label='Lower 3-sigma limit')
plt.axvline(upper_limit, color='r', linestyle='--', label='Upper 3-sigma limit')
plt.title("Redshift distribution after first filtering\n(Red lines show ±3σ limits)")
plt.xlabel("Redshift (specz)")
plt.ylabel("Number of galaxies")
plt.legend()
plt.show()

# Final filtering with 3-sigma limits(not really needed because as seen in histogram, all values lie within the limits)
filtered_df = averaged_df[(averaged_df['specz'] >= lower_limit) & (averaged_df['specz'] <= upper_limit)].copy()

# Histogram after final filtering
plt.hist(filtered_df['specz'], bins=90)
plt.grid()
plt.title("Final redshift distribution after filtering")
plt.xlabel("Redshift (specz)")
plt.ylabel("Number of galaxies")
plt.show()
# Calculating expansion velocity and add to dataframe using .loc
filtered_df.loc[:, 'velocity'] = c_kms * ((1 + filtered_df['specz'])**2 - 1) / ((1 + filtered_df['specz'])**2 + 1)

# Histogram of expansion velocities
plt.hist(filtered_df['velocity'], bins=90)
plt.grid()
plt.title(f"expansion velocity distribution\nAverage velocity = {filtered_df['velocity'].mean():.2f} km/s")
plt.xlabel("Velocity (km/s)")
plt.ylabel("Number of galaxies")
plt.show()

# Calculating cluster redshift mean and velocity dispersion
cluster_redshift = filtered_df['specz'].mean()
filtered_df.loc[:, 'disp'] = c_kms * ((1 + filtered_df['specz'])**2 - (1 + cluster_redshift)**2) / ((1 + filtered_df['specz'])**2 + (1 + cluster_redshift)**2)
velocity_disp = filtered_df['disp'].std()

print(f"Cluster redshift = {cluster_redshift:.4f}")
print(f"Velocity dispersion (line-of-sight) = {velocity_disp:.2f} km/s")

# Histogram of projected separation
plt.hist(filtered_df['proj_sep'], bins=90)
plt.grid()
plt.title("Projected separation distribution (arcminutes)")
plt.xlabel("Projected separation (arcmin)")
plt.ylabel("Number of galaxies")
plt.show()

# Convert projected separation max to radians
theta_rad = (filtered_df['proj_sep'].max()) * (np.pi / 180) / 60  # arcmin to radians

# Comoving distance and angular diameter distance in Mpc
r = (c_kms * cluster_redshift / H_0.value) * (1 - cluster_redshift * (1 + q0) / 2)
DA = r / (1 + cluster_redshift)
diameter = DA * theta_rad

# Convert velocity dispersion to m/s and diameter to meters
velocity_disp_ms = (velocity_disp * u.km / u.s).to(u.m / u.s).value
diameter_m = diameter * 3.0e22  # 1 Mpc in meters
print(f'Diameter of the cluster is {diameter}')
# Calculate dynamical mass in kg, then solar masses
M_dyn_kg = (velocity_disp_ms**2 * diameter_m) / (2 * G.value)#dividing by 2 because we want radius not diameter
M_dyn_solar = M_dyn_kg / 2.0e30
print(f"Dynamical mass of the cluster = {M_dyn_solar:.2e} solar masses")

# Calculating luminosity distance in parsecs (float)
filtered_df.loc[:, 'ld'] = cosmo.luminosity_distance(filtered_df['specz']).to('pc').value

# Calculating absolute magnitude in r-band
filtered_df.loc[:, 'abs_rmag'] = filtered_df['rmag'] - 5 * np.log10(filtered_df['ld'] / 10)
M_r_sun = 4.67  # Sun's absolute magnitude in r-band

# Calculating luminosity in solar units
filtered_df.loc[:, 'L_Lsun'] = 10 ** (0.4 * (M_r_sun - filtered_df['abs_rmag']))

# Calculating stellar mass-to-light ratio (log scale)
filtered_df.loc[:, 'logratio'] = -0.306 + 1.097 * (filtered_df['gmag'] - filtered_df['rmag'])

# Converting log ratio to ratio
filtered_df.loc[:, 'ratio'] = 10 ** filtered_df['logratio']

# Calculating stellar mass per galaxy
filtered_df.loc[:, 'stell_mass'] = filtered_df['ratio'] * filtered_df['L_Lsun']

# Total stellar mass of cluster
total_stellar_mass = filtered_df['stell_mass'].sum()
print(f"Total stellar/luminous mass of the cluster = {total_stellar_mass:.2e} solar masses")
