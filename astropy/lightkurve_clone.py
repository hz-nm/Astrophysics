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
stri = ''
for x, time in zip(lc['pdcsap_flux'], lc.time.bkjd):
    stri = ''
    for n in str(x):
        if n not in not_inc:
            stri += n
    print(stri)
    pdcsap_new_vals.append((stri, time))

# %% Do further stuff to it
# 1 - Stitching
pdcsap_all_vals = [pdcsap_new_vals]
pdcsap_new_vals = []


for lc_file in lc_files[1:]:
    lc_n = TimeSeries.read(os.path.join(main_file, lc_file), format='kepler.fits')

    for x, time in zip(lc_n['pdcsap_flux'], lc_n.time.bkjd):
        stri = ''
        for n in str(x):
            if n not in not_inc:
                stri += n
        pdcsap_new_vals.append((stri, time))
    

    pdcsap_all_vals.append(pdcsap_new_vals)
    pdcsap_new_vals = []

# Now the pdc_all_vals list should contain all of the values for all the light curves
# So pdc_all_vals should contain 17 members
# %% Converging into a single list But first we have to normalize!
pdcsap_flux_vals = []
pdcsap_time_vals = []
for light_c in pdcsap_all_vals:
    for val in light_c:
        if val[0] != 'a':
            pdcsap_flux_vals.append(float(val[0]))
            pdcsap_time_vals.append(float(val[1]))

# pdcsap_flux_vals = savgol_filter(pdcsap_flux_vals, 21, 2)

plt.plot(pdcsap_time_vals, pdcsap_flux_vals)

# %% Flatten from LightKurve
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


for iter in np.arange(0, niters):
    if break_tolerance is None:
        break_tolerance = np.nan
    if polyorder >= window_length:
        polyorder = window_length - 1
        # log.warning(
        #     "polyorder must be smaller than window_length, "
        #     "using polyorder={}.".format(polyorder)
        # )
    # Split the lightcurve into segments by finding large gaps in time
    dt = pdcsap_time_vals[mask][1:] - pdcsap_time_vals[mask][0:-1]
    with warnings.catch_warnings():  # Ignore warnings due to NaNs
        warnings.simplefilter("ignore", RuntimeWarning)
        cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
    low = np.append([0], cut)
    high = np.append(cut, len(pdcsap_time_vals[mask]))
    # Then, apply the savgol_filter to each segment separately
    trend_signal = Quantity(np.zeros(len(pdcsap_time_vals[mask])), unit=u.day)
    for l, h in zip(low, high):
        # Reduce `window_length` and `polyorder` for short segments;
        # this prevents `savgol_filter` from raising an exception
        # If the segment is too short, just take the median
        if np.any([window_length > (h - l), (h - l) < break_tolerance]):
            trend_signal[l:h] = np.nanmedian(pdcsap_flux_vals[mask][l:h])
        else:
            # Scipy outputs a warning here that is not useful, will be fixed in version 1.2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                trsig = savgol_filter(
                    x=pdcsap_flux_vals[mask][l:h],
                    window_length=window_length,
                    polyorder=polyorder,
                )
                trend_signal[l:h] = Quantity(trsig, u.day)

        # Ignore outliers; note we add `1e-14` below to avoid detecting
        # outliers which are merely caused by numerical noise.
        mask1 = np.nan_to_num(np.abs(pdcsap_flux_vals[mask] - trend_signal)) < (
            np.nanstd(pdcsap_flux_vals[mask] - trend_signal) * sigma
            + Quantity(1e-14, u.day)
        )
        f = interp1d(
            pdcsap_time_vals[mask][mask1],
            trend_signal[mask1],
            fill_value="extrapolate",
        )
        trend_signal = Quantity(f(pdcsap_time_vals), u.day)
        # In astropy>=5.0, mask1 is a masked array
        if hasattr(mask1, 'mask'):
            mask[mask] &= mask1.filled(False)
        else:  # support astropy<5.0
            mask[mask] &= mask1

    flatten_lc = pdcsap_flux_vals.copy()
    with warnings.catch_warnings():
        # ignore invalid division warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        flatten_lc.flux = flatten_lc.flux / trend_signal.value
        flatten_lc.flux_err = flatten_lc.flux_err / trend_signal.value
# %%
