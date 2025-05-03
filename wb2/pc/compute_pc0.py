"""
compute_pc0.py

Script to compute PC^(0) measure for ERA5 and the IFS analysis.
"""

import numpy as np
import xarray as xr
from tqdm import tqdm

# PC0 Functions
def pc0(y):
    """
    Compute PC^(0) = (1/(2*n^2)) * \sum_{i,j} |y[i] - y[j]| using a sorted-based O(n log n) approach for a 1D array y.

    Parameters:
        y (1D array-like): Observations or values along time.

    Returns:
        float: The computed PC^(0) value.
    """
    n = len(y)
    y_sorted = np.sort(y)
    ranks = np.arange(1, n + 1)          
    weights = 2 * ranks - n - 1          
    return np.sum(weights * y_sorted) / (n**2)


def pc0_along_time(da):
    """
    Apply pc0 along the 'time' dimension of an xarray DataArray.
    Reduces 'time' and returns a 2D field (latitude, longitude).

    Parameters:
        da (xarray.DataArray): Input data array with a 'time' dimension.

    Returns:
        xarray.DataArray: 2D output (lat, lon) after reducing time.
    """
    return xr.apply_ufunc(
        pc0,
        da,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
    )

# Main Script
def main():
    # Variables of interest
    variables = [
        '2m_temperature',
        'mean_sea_level_pressure',
        '10m_wind_speed',
        'total_precipitation_24hr'
    ]

    # Time range
    time_range = slice('2020-01-01', '2021-01-10')

    # Load ERA5 data
    era5 = xr.open_zarr(
        store='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr', 
        storage_options={"token": "anon"}
        ).sel(time=slice('2020-01-01','2021-01-10'))[variables].load()
    print('Loading ERA5 complete')

    # Load IFS high-resolution analysis data
    ifs_analysis = xr.open_zarr(
        store='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr', 
        storage_options={"token": "anon"}
        ).sel(time=slice('2020-01-01','2021-01-10'))[variables[:-1]].load()
    print('Loading IFS analysis complete')

    # Lead times of interest
    lead_times = [
        np.timedelta64(1, 'D'),
        np.timedelta64(3, 'D'),
        np.timedelta64(5, 'D'),
        np.timedelta64(7, 'D'),
        np.timedelta64(10, 'D')
    ]

    # Load 2020 GraphCast forecasts to get correct time and prediction_timedelta values
    forecasts = xr.open_zarr(
        store='gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr',
        storage_options={'token': 'anon'}
    ).sel(
        time=slice('2020-01-01', '2020-12-31'),
        prediction_timedelta=lead_times
    )[variables]

    # Compute PC0 for ERA5
    era5_pc0_list = []
    for lt in tqdm(forecasts.prediction_timedelta, desc='ERA5 PC0'):
        # align times for this lead time
        valid_time = forecasts.time + lt
        targets = era5.sel(time=valid_time)
        era5_pc0_list.append(targets.map(pc0_along_time))

    era5_pc0 = xr.concat(era5_pc0_list, dim='prediction_timedelta')

    # Compute PC0 for IFS analysis
    ifs_pc0_list = []
    for lt in tqdm(forecasts.prediction_timedelta, desc='IFS PC0'):
        valid_time = forecasts.time + lt
        targets = ifs_analysis.sel(time=valid_time)
        ifs_pc0_list.append(targets.map(pc0_along_time))

    ifs_analysis_pc0 = xr.concat(ifs_pc0_list, dim='prediction_timedelta')

    # Save results to disk or Zarr
    era5_pc0.to_netcdf('results/era5_pc0.nc', engine='h5netcdf')
    ifs_analysis_pc0.to_netcdf('results/ifs_analysis_pc0.nc', engine='h5netcdf')

    print('PC0 computation complete.')


if __name__ == '__main__':
    main()