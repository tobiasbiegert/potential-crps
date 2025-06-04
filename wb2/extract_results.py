import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

# Variables
variables = [
    'mean_sea_level_pressure',
    '2m_temperature',
    '10m_wind_speed',
    'total_precipitation_24hr',
]

# Lead times as numpy timedeltas for slicing
lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
]

# List out each model
models_vs_era5 = {
    'graphcast_vs_era5':             'gs://$BUCKET/easyuq/pc/graphcast_240x121_vs_era5.zarr',
    'graphcast_operational_vs_era5': 'gs://$BUCKET/easyuq/pc/graphcast_operational_240x121_vs_era5.zarr',
    'pangu_vs_era5':                 'gs://$BUCKET/easyuq/pc/pangu_240x121_vs_era5.zarr',
    'pangu_operational_vs_era5':     'gs://$BUCKET/easyuq/pc/pangu_operational_240x121_vs_era5.zarr',
    'ifs_hres_vs_era5':              'gs://$BUCKET/easyuq/pc/ifs_hres_240x121_vs_era5.zarr',
    'era5_climatology_vs_era5':      'results/era5_climatology_240x121_vs_era5.zarr',
}

models_vs_ifs_analysis = {
    'graphcast_vs_ifs_analysis':             'gs://$BUCKET/easyuq/pc/graphcast_240x121_vs_ifs_analysis.zarr',
    'graphcast_operational_vs_ifs_analysis': 'gs://$BUCKET/easyuq/pc/graphcast_operational_240x121_vs_ifs_analysis.zarr',
    'pangu_vs_ifs_analysis':                 'gs://$BUCKET/easyuq/pc/pangu_240x121_vs_ifs_analysis.zarr',
    'pangu_operational_vs_ifs_analysis':     'gs://$BUCKET/easyuq/pc/pangu_operational_240x121_vs_ifs_analysis.zarr',
    'ifs_hres_vs_ifs_analysis':              'gs://$BUCKET/easyuq/pc/ifs_hres_240x121_vs_ifs_analysis.zarr',
    'era5_climatology_vs_ifs_analysis':      'results/era5_climatology_240x121_vs_ifs_analysis.zarr',
}

# PC^0
era5_pc0 = xr.open_dataset('results/era5_pc0.nc', decode_timedelta=True).load()
ifs_analysis_pc0 = xr.open_dataset('results/ifs_analysis_pc0.nc', decode_timedelta=True).load()

def process_model(name, zarr_path, pc0):

    # Load
    with ProgressBar():
        ds = xr.open_zarr(zarr_path, decode_timedelta=True).load()

    # get only crps values
    crps_keys = [f'{var}_crps' for var in variables if f'{var}_crps' in ds.data_vars]
    ds_crps = ds[crps_keys]

    # same for the time‐metrics
    time_keys = [f'{var}_time' for var in variables if f'{var}_time' in ds.data_vars]
    ds_time = ds[time_keys]

    # build rename maps
    rename_crps = {k: k.rsplit('_',1)[0] for k in crps_keys}
    rename_time = {k: k.rsplit('_',1)[0] for k in time_keys}

    ds_crps = ds_crps.rename(rename_crps)
    ds_crps.to_netcdf(f'results/{name}_crps.nc')
    
    # Rename and switch “task” → “metric”
    ds_time = (
        ds_time
        .rename(rename_time)
        .rename({'task': 'metric'})
    )

    # Prepare PC slice
    ds_pc = (
        ds_crps
        .mean(dim='time')
        .expand_dims(metric=['pc'])
    )

    # Concat idr_time / eval_time with pc
    ds_combined = xr.concat([ds_time, ds_pc], dim='metric')

    # Compute PCS
    pcs = (
        (pc0 - ds_combined.sel(metric='pc'))
        / pc0
    ).expand_dims(metric=['pcs'])
    ds_combined = xr.concat([ds_combined, pcs], dim='metric')

    # Save final Dataset
    ds_combined.to_netcdf(f'results/{name}_pc.nc')

    print(f'{name} done')

# Run for every model in the list
for model_name, path in models_vs_era5.items():
    process_model(model_name, path, era5_pc0)

for model_name, path in models_vs_ifs_analysis.items():
    process_model(model_name, path, ifs_analysis_pc0)

# Get CRPS of IFS ENS for each grid point.
with ProgressBar():
    ifs_ens_vs_era5_crps = xr.open_zarr('gs://$BUCKET/baseline/ifs_ens_240x121_surface_vs_era5_2020_probabilistic_spatial.zarr', decode_timedelta=True).sel(lead_time=lead_times, metric='crps').load()
ifs_ens_vs_era5_crps.to_netcdf('results/ifs_ens_vs_era5_crps.nc')

with ProgressBar():
    ifs_ens_vs_ifs_analysis_crps = xr.open_zarr('gs://$BUCKET/baseline/ifs_ens_240x121_vs_analysis_2020_probabilistic_spatial.zarr', decode_timedelta=True)[variables[:-1]].sel(lead_time=lead_times, metric='crps').load()
ifs_ens_vs_ifs_analysis_crps.to_netcdf('results/ifs_ens_vs_ifs_analysis_crps.nc')