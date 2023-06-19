import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.timeseries import BoxLeastSquares
from astropy.stats import sigma_clipped_stats
from astropy.timeseries import aggregate_downsample
import os
import lightkurve as lk


# main_file = '000757450/'
# lcfs = os.listdir(main_file)

# # Use the normalized PDCSAP_FLUX
# lc = lcfs[0].PDCSAP_FLUX.normalize()

# # Loop through the rest of the LCFS, appending to the first light curve
# for lcf in lcfs[1:]:
#     lc = lc.append(lcf.PDCSAP_FLUX.normalize())

search_result = lk.search_lightcurve('757450', author='Kepler', cadence='long')
# period, t0, duration_hours = 8.88492, 134.452, 2.078
# Download all available Kepler light curves
lc_collection = search_result.download_all()
lc_collection_stitched = lc_collection.stitch()
lc_clean = lc_collection_stitched.PDCSAP_FLUX.normalize().flatten(window_length=21)
lc_clean.plot()

# Create array of periods to search
period_array = np.linspace(1, 20, 10000)
# Create a BLSPeriodogram
bls = lc_clean.to_periodogram(method='bls', period=period_array, frequency_factor=500);
bls.plot()

period = bls.period_at_max_power
t0 = bls.transit_time_at_max_power
duration_hours = bls.duration_at_max_power

folded_lc = lc_clean.fold(period=period, epoch_time=t0)
folded_lc_plot = folded_lc.scatter()
folded_lc_plot.set_xlim(-1, 1)
folded_lc_plot.set_ylim(0.97, 1.02)

ax = lc_clean.plot()
binned_lc = lc_clean.bin(time_bin_size=100).plot(ax=ax, c='red');

# Create a BLS model using the BLS parameters
planet_b_model = bls.get_transit_model(period=period, transit_time=t0)

ymin = lc_clean['flux'].min()
ymax = lc_clean['flux'].max()

ymean = lc_clean['flux'].mean()

ymin = ymean - 0.02
ymax = ymean + 0.02

ax = lc_clean.fold(period, t0).scatter()
planet_b_model.fold(period, t0).plot(ax=ax, c='r', lw=2)
ax.set_xlim(-1, 1)
ax.set_ylim(ymin, ymax)

# Create a cadence mask using the BLS parameters
# planet_b_mask = bls.get_transit_mask(period=period,
#                                      transit_time=t0,
#                                      duration=duration_hours)

# masked_lc = lc_clean*(~planet_b_mask)
# ax = masked_lc.scatter();
# lc_clean[planet_b_mask].scatter(ax=ax, c='r', label='Masked');





# folded_lc.plot_river()
# folded_lc.plot_river(bin_points=5, method='median');
# folded_lc.plot_river(bin_points=1, method='sigma');