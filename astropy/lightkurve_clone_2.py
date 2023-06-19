# Author: 
# This file should serve as a test bed for cloning the lightkurve library
# so that we can work on downloaded files
# %% Important Links
# https://zkbt.github.io/henrietta/docs/lightcurves.html

# %%

from matplotlib.style import library

from astropy.timeseries import TimeSeries
from astropy.utils.data import get_pkg_data_filename
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares


from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample
import os
import lightkurve as lk
from astropy.table import vstack, hstack


import os
import datetime
import logging
import warnings
import collections

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
from matplotlib import pyplot as plt
from copy import deepcopy

from astropy.table import Table, Column, MaskedColumn
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from astropy.timeseries import TimeSeries, aggregate_downsample
from astropy.table import vstack
from astropy.stats import calculate_bin_edges
from astropy.utils.decorators import deprecated, deprecated_renamed_argument
from astropy.utils.exceptions import AstropyUserWarning


# %% LOAD THE FILE WE WANT
main_file = '000757450/'
lc_files = os.listdir(main_file)

# %% Load the Light Curve file from our dataset
lc = TimeSeries.read(os.path.join(main_file, lc_files[0]), format='kepler.fits')

# %% convert the pdcsap values into a list so that we can work on it further
not_inc = ' electron/s'       # don't include this
pdcsap_new_vals = []
pdcsap_all_vals = []
pdcsap_lc_vals = []
pdcsap_lcs = []

for lc_file in lc_files[:]:
    lc_n = TimeSeries.read(os.path.join(main_file, lc_file), format='kepler.fits')

    for x, time in zip(lc_n['pdcsap_flux'], lc_n.time.bkjd):
        stri = ''
        for n in str(x):
            if n not in not_inc:
                stri += n
        pdcsap_new_vals.append([stri, time])
        if stri != 'a':
            pdcsap_lc_vals.append(float(stri))
        elif stri != 'NaN':
            stri = np.nan
            pdcsap_lc_vals.append(stri)
        else:
            stri = np.nan
            pdcsap_lc_vals.append(stri)
    
    pdcsap_all_vals.append(pdcsap_new_vals)
    pdcsap_lcs.append(pdcsap_lc_vals)
    pdcsap_new_vals = []
    pdcsap_lc_vals = []



# %% Normalize
normalized_lc = []
normalized_lcs = []


for light_c in pdcsap_lcs:
    temp_array = np.array(light_c)
    median_lc = np.nanmedian(temp_array)

    for lc_val in light_c:
        if lc_val != 'a':
            new_val = lc_val/median_lc
            normalized_lc.append(new_val)
        else:
            new_val = np.nan
            normalized_lc.append(new_val)

    temp_array_lc = np.array(normalized_lc)
    normalized_lc = savgol_filter(temp_array_lc, 21, 2)
    normalized_lcs.append(normalized_lc)
    normalized_lc = []
# %% Now let's stitch our data
pdcsap_flux_vals = []
pdcsap_time_vals = []
for light_c in normalized_lcs:
    for val in light_c:
            pdcsap_flux_vals.append(val)

pdcsap_time_vals = []
for light_c in pdcsap_all_vals:
    for val in light_c:
            pdcsap_time_vals.append(float(val[1]))

pdcsap_flux_vals = savgol_filter(pdcsap_flux_vals, 5, 2)

plt.plot(pdcsap_time_vals, pdcsap_flux_vals)
plt.scatter(pdcsap_time_vals, pdcsap_flux_vals)


# %% Cloning the actual light curve flatten

mask = np.ones(len(pdcsap_time_vals), dtype=bool)
pdcsap_flux_vals = np.array(pdcsap_flux_vals)
pdcsap_time_vals = np.array(pdcsap_time_vals)


extra_mask = np.isfinite(pdcsap_flux_vals)
sigma = 3
extra_mask &= np.nan_to_num(np.abs(pdcsap_flux_vals - np.nanmedian(pdcsap_flux_vals))) <= (
            np.nanstd(pdcsap_flux_vals) * sigma
        )

niters = 3
break_tolerance = 5
window_length = 21
polyorder = 2

cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1

trend_signal = np.zeros(len(pdcsap_time_vals))
trend_signal = pdcsap_flux_vals

mask1 = np.nan_to_num(np.abs(pdcsap_flux_vals - trend_signal))




# %%
