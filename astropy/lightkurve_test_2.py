# Author: 
# This file should serve as a test bed for cloning the lightkurve library
# so that we can work on downloaded files
# %% Important Links
# https://zkbt.github.io/henrietta/docs/lightcurves.html


# Very Important LINK
# https://docs.lightkurve.org/tutorials/3-science-examples/exoplanets-identifying-transiting-planet-signals.html#1.-Downloading-a-Light-Curve-and-Removing-Long-Term-Trends
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
import plotly.express as px

# %% Get the fits file and append them into a list
all_lightcurves = []        # a list that will contain all lightcurves.

main_folder = '000757450/'
fit_files = os.listdir(main_folder) 

for file in fit_files:
    path_to_file = os.path.join(main_folder, file)

    lightcurve_ts = TimeSeries.read(path_to_file, format='kepler.fits')
    lightcurve_ts.time.format = 'bkjd'

    lightcurve_lk = lk.LightCurve(data=lightcurve_ts, flux=lightcurve_ts['pdcsap_flux'], flux_err=lightcurve_ts['pdcsap_flux_err'])

    all_lightcurves.append(lightcurve_lk)

# %%
# Now generating a collection of all lightcurves
print("[INFO] Generating a collection of {} Lightcurves...".format(len(all_lightcurves)))

lightkurve_collection = lk.LightCurveCollection(all_lightcurves)

print("[INFO] Collection generated..")

# %%
# Stitching all the lightcurves.
stitched_lightcurves = lightkurve_collection.stitch()
stitched_lightcurves.plot()    # For some reason this isn't working

# %%
# Actual Plot
# plt.plot(stitched_lightcurves.time.bkjd, stitched_lightcurves['flux'])
plt.plot(stitched_lightcurves['flux'])
plt.xlabel('Time (days)')
plt.ylabel('Normalized flux')

plt.savefig('test_plot.png')
# plt.show()
# %% Now comes the folding and everything
# lc_flattened = stitched_lightcurves.flatten(window_length=901).remove_outliers(4)
lc_flattened = stitched_lightcurves.flatten().remove_outliers()
lc_flattened.plot()

# %% Box LEAST SQuares?
period = np.linspace(1, 20, 10000)
bls = lc_flattened.to_periodogram(method='bls', period=period, frequency_factor=500)
bls.plot()

# %% FOLDING
p, t0 = 8.884920, 134.452       # TCE_PERIOD, TCE_TIME0bK
duration = bls.duration_at_max_power
period_t0 = bls.period_at_max_power
transit_t0 = bls.transit_time_at_max_power

folded_lc = lc_flattened.fold(period=p, epoch_time=t0).scatter()
# folded_lc.scatter()
folded_lc.set_xlim(-1, 1)
# folded_lc.set_ylim(0.98, 1.01)


# %% Fit the transit model
planet_model = bls.get_transit_model(period=p, transit_time=t0, duration=duration)

folded_lc = lc_flattened.fold(period=p, epoch_time=t0)
# folded_lc.scatter()

folded_planet = planet_model.fold(p, t0)
# folded_planet.plot(ax=folded_lc, c='r', lw=2)
# folded_planet.plot()
planet_model.fold(p, t0).plot()

# folded_lc.set_xlim(-1, 1)

# %% RIVER PLOT?
# folded_lc.plot_river()

# %% Changing the scale to sigma deviation
# folded_lc.plot_river(bin_points=1, method='sigma')

# %% Now for the other planet
# create a cadence mask using the BLS parameter
planet_b_mask = bls.get_transit_mask(period=p, transit_time=t0, duration=duration)
masked_lc = lc_flattened[~planet_b_mask]
ax = masked_lc.scatter()
lc_flattened[planet_b_mask].scatter(ax=ax, c='r', label='MASKED')

# %% Finding if there's any other planet in the given lightcurve
period = np.linspace(1, 300, 10000)     # general value -- increase the max (i.e. 300) if there's another planet you would like to find.
bls = masked_lc.to_periodogram('bls', period=period, frequency_factor=500)
bls.plot()

# %%
planet_c_period = bls.period_at_max_power
planet_c_t0 = bls.transit_time_at_max_power
planet_c_dur = bls.duration_at_max_power


# We again plot the phase folded lightcurve to examine the transit
ax = masked_lc.fold(planet_c_period, planet_c_t0).bin(0.1)
# ax.plot(ax=ax, c='r', lw=2, label='Binned Flux')
ax.plot()

# ax.set_xlim(-5, 5)

# %% Use BLS model to visualize the transit timing in the light curve
planet_c_model = bls.get_transit_model(period=planet_c_period,
                                        transit_time = planet_c_t0,
                                        duration = planet_c_dur)

ax = lc_flattened.scatter()
lc_flattened.interact_bls(notebook_url='localhost:8000', resolution=2000)
planet_model.plot(ax=ax, c='dodgerblue', label='Planet B Transit Model')
planet_c_model.plot(ax=ax, c='r', label='Planet C Transit Model')

# ax.set_xlim(1150, 1170)


# %% Using the Plotly PLOT
import pandas as pd
columns = ['TIME', 'FLATTENED_FLUX', 'PLANET_B_FLUX', 'PLANET_C_FLUX']
[lc_flattened.time.bkjd, lc_flattened.flux, planet_model.flux, planet_c_model.flux]

time = pd.Series(lc_flattened.time.bkjd)
# flattened_flux = pd.Series(lc)
# df = pd.DataFrame(, columns=columns, index=False)

fig = px.scatter(x=lc_flattened.time.bkjd, y=lc_flattened.flux)
fig.add_scatter(x=lc_flattened.time.bkjd, y=planet_model.flux)
fig.add_scatter(x=lc_flattened.time.bkjd, y=planet_c_model.flux)
fig.show()


# %%
