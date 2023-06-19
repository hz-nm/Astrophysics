# from --> https://docs.astropy.org/en/stable/timeseries/index.html

from unittest import result
from astropy.timeseries import TimeSeries
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt

import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares

from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample
import os


main_file = '000757450/'
lc_files = os.listdir(main_file)



filename_path = '000757450/kplr000757450-2009166043257_llc.fits'


ts = TimeSeries.read(filename_path, format='kepler.fits')
periodogram = BoxLeastSquares.from_timeseries(ts, 'pdcsap_flux')

results = periodogram.autopower(0.2 * u.day)
print(results)
# print(dir(results))
# print(help(results))



print("")
print("")
best = np.argmax(results.power) 
print(best)
print("")
period = results.period[best]
print(period)
print("")

transit_time = results.transit_time[best]
# index = results.transit_time[best]
# print(results.transit_time)
all_vals = []
# for res in results.transit_time:

all_vals = [res for res in results.transit_time]
print("The lenght is : {}".format(len(all_vals)))

    
print("")
print("Transit time: {}".format(transit_time))

ts_folded = ts.fold(period=period, epoch_time=transit_time)

mean, median, stddev = sigma_clipped_stats(ts_folded['pdcsap_flux'])
ts_folded['pdcsap_flux_norm'] = ts_folded['pdcsap_flux'] / median


ts_binned = aggregate_downsample(ts_folded, time_bin_size=0.03 * u.day)
# print(ts['pdcpdcsap_flux'])

# plt.plot(ts.time.jd, ts['pdcsap_flux'], 'k.', markersize=1)
# plt.xlabel('Julian Date')
# plt.ylabel('SAP Flux (e-/s)')

plt.plot(ts_folded.time.jd, ts_folded['pdcsap_flux_norm'], 'k.', markersize=1)
plt.plot(ts_binned.time_bin_start.jd, ts_binned['pdcsap_flux_norm'], 'r-', drawstyle='steps-post')
plt.xlabel('Time (days)')
plt.ylabel('Normalized flux')

# plt.plot(ts_folded.time.jd, ts_folded['pdcpdcsap_flux'], 'k.', markersize=1)
# plt.xlabel('Time (days)')
# plt.ylabel('SAP Flux (e-/s)')

plt.show()