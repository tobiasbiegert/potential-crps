"""
construct_era5_climatology_forecasts.py

Load ERA-5 climatology from WB2, reshape it to a time series, and build lead‐time forecasts of 1, 3, 5, 7, and 10 days by rolling to make the climatology compatible to compute_pc.py.
"""
import xarray as xr
import pandas as pd
import numpy as np

# Variables of interest
variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_wind_speed',
    'total_precipitation_24hr'
]

# Lead times as numpy timedeltas
lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
]

# Load climatology dataset from WB2
era5_climatology = xr.open_zarr(
    store='gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr', 
    storage_options={"token": "anon"}
    ).sel(hour=[0,12])[variables].load()
print('loading complete')

# Build time index for 2020 at 12 h resolution
time_index = pd.date_range(start='2020-01-01 00:00', end='2020-12-31 12:00', freq='12h')

# Combine dayofyear and hour into a single time dimension
era5_climatology = (
    era5_climatology
    .stack(datetime=('dayofyear', 'hour'))
    .assign_coords(time=('datetime', time_index))
    .swap_dims({'datetime': 'time'})
    .drop_vars(['datetime', 'dayofyear', 'hour'])
)

delta = era5_climatology.time.diff("time").isel(time=0)     # here 12 hours

# For each lead time, roll the data and tag it with its prediction_timedelta
rolled_list = []
for td in lead_times:
    shift_steps = int(td / delta)
    era5_climatology_shifted = era5_climatology.roll(time=-shift_steps, roll_coords=False)
    era5_climatology_shifted = era5_climatology_shifted.expand_dims(prediction_timedelta=[td])
    rolled_list.append(era5_climatology_shifted)
print('rolling complete')

# Concatenate lead time DataArrays
era5_climatology_forecasts = xr.concat(rolled_list, dim="prediction_timedelta")

# Save as Zarr
era5_climatology_forecasts.to_zarr('data/era5_climatology_forecasts.zarr', mode='w', consolidated=True)