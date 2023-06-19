# Author: 
# This file should serve as a test bed for cloning the lightkurve library
# so that we can work on downloaded files
# %% Important Links
# https://zkbt.github.io/henrietta/docs/lightcurves.html
# https://docs.astropy.org/en/stable/timeseries/times.html
# https://docs.astropy.org/en/stable/api/astropy.timeseries.TimeSeries.html#astropy.timeseries.TimeSeries
# https://docs.lightkurve.org/reference/api/lightkurve.LightCurveCollection.stitch.html
# https://docs.lightkurve.org/_modules/lightkurve/lightcurve.html#LightCurve.flatten
# https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.html


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

import lightkurve as lk

lc_test = TimeSeries.read(os.path.join(main_file, lc_files[0]), format='kepler.fits')
lc_test_1 = TimeSeries.read(os.path.join(main_file, lc_files[1]), format='kepler.fits')
lc_test.time.format = 'bkjd'

lk_test = lk.LightCurve(data=lc_test, flux=lc_test['pdcsap_flux'], flux_err=lc_test['pdcsap_flux_err'], time=lc_test.time.bkjd)
lk_test_1 = lk.LightCurve(data=lc_test_1, flux=lc_test_1['pdcsap_flux'], flux_err=lc_test_1['pdcsap_flux_err'], time=lc_test_1.time.bkjd)



lk_collection_test = lk.LightCurveCollection([lk_test, lk_test_1])

lk_stitch = lk_collection_test.stitch()
lk_stitch.plot()

# %%
plt.plot(lk_stitch.time.bkjd, lk_stitch['flux'])
plt.xlabel('Time (days)')
plt.ylabel('Normalized Flux')

