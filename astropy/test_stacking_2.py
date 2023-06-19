# Important Links
# https://docs.astropy.org/en/stable/timeseries/analysis.html


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


main_file = '000757450/'
lc_files = os.listdir(main_file)


stacked_lc = TimeSeries.read('000757450/kplr000757450-2009166043257_llc.fits', format='kepler.fits')
mean, median, stddev = sigma_clipped_stats(stacked_lc['pdcsap_flux'])
stacked_lc['pdcsap_flux_norm'] = stacked_lc['pdcsap_flux'] / median
stacked_lc = aggregate_downsample(stacked_lc, time_bin_size= 0.03 * u.day)


for lc_f in lc_files:
    file_path = os.path.join(main_file, lc_f)
    temp_ts = TimeSeries.read(file_path, format='kepler.fits')
    mean, median, stddev = sigma_clipped_stats(temp_ts['pdcsap_flux'])
    temp_ts['pdcsap_flux_norm'] = temp_ts['pdcsap_flux'] / median
    temp_ts = aggregate_downsample(temp_ts, time_bin_size= 0.03 * u.day)

    stacked_lc = vstack([stacked_lc, temp_ts], join_type='inner', metadata_conflicts='silent')

    # period = 8.884920 * u.day
    # # stacked_lc = stacked_lc.fold(period=period)
    # mean, median, stddev = sigma_clipped_stats(stacked_lc['pdcsap_flux'])
    # stacked_lc['pdcsap_flux_norm'] = stacked_lc['pdcsap_flux'] / median
    # stacked_lc = aggregate_downsample(stacked_lc, time_bin_size= 0.03 * u.day)

    
    # stacked_lc = hstack([stacked_lc, temp_ts], join_type='outer', metadata_conflicts='silent')

plt.plot(stacked_lc.time.bkjd, stacked_lc['pdcsap_flux'], 'k.', markersize=1)

# periodogram = BoxLeastSquares.from_timeseries(stacked_lc, 'pdcsap_flux')

# results = periodogram.autopower(0.2 * u.day)

# best = np.argmax(results.power) 

# period = results.period[best]
# transit_time = results.transit_time[best]

period = 8.884920 * u.day
# transit_time = 134.452 * u.jd

print("The period is {}...Is it EQUAL To: 8.884920".format(period))
print("The transit period is {}... Is it equal to : 134.452")

# p, t0 = 8.884920, 134.452       # TCE_PERIOD, TCE_TIME0bK
# ts_folded = stacked_lc.fold(period=period, epoch_time=transit_time)

# ts_folded = stacked_lc.fold(period=period)

# mean, median, stddev = sigma_clipped_stats(ts_folded['pdcsap_flux'])
# ts_folded['pdcsap_flux_norm'] = ts_folded['pdcsap_flux'] / median


# ts_binned = aggregate_downsample(ts_folded, time_bin_size= 0.03 * u.day)

# plt.plot(ts_folded.time.jd, ts_folded['pdcsap_flux_norm'], 'k.', markersize=1)
# plt.plot(ts_binned.time_bin_start.jd, ts_binned['pdcsap_flux_norm'], 'r-', drawstyle='steps-post')
# plt.xlabel('Time (days)')
# plt.ylabel('Normalized flux')
# plt.show()