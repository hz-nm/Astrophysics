# from unittest import result
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


main_file = '000757450/'
lc_files = os.listdir(main_file)

filename_path = '000757450/kplr000757450-2009166043257_llc.fits'
# ts = TimeSeries.read(filename_path, format='kepler.fits')

# lc_test = ts
# lc_test.plot()


search_result = lk.search_lightcurve('757450', author='Kepler', cadence='long')
# Download all available Kepler light curves
lc_collection = search_result.download_all().stitch()
lc_collection.plot()

clc = lc_collection.flatten(21)

p, t0 = 8.884920, 134.452       # TCE_PERIOD, TCE_TIME0bK

folded_lc = clc.fold(period=p, epoch_time=t0)
folded_lc.scatter()


folded_lc.plot_river()

folded_lc.plot_river(bin_points=5, method='median')
folded_lc.plot_river(bin_points=1, method='sigma')


