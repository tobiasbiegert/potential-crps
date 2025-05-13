import xarray as xr
from dask.diagnostics import ProgressBar

# 1) Settings
variables = [
    'mean_sea_level_pressure',
    '2m_temperature',
    '10m_wind_speed',
    'total_precipitation_24hr',
]

# List out each model
models_vs_era5 = {
    'graphcast_vs_era5':    'gs://$BUCKET/easyuq/pc/graphcast_240x121_vs_era5.zarr',
    'pangu_vs_era5':        'gs://$BUCKET/easyuq/pc/pangu_240x121_vs_era5.zarr',
    'ifs_hres_vs_era5':     'gs://$BUCKET/easyuq/pc/ifs_hres_240x121_vs_era5.zarr',
}

models_vs_ifs_analysis = {
    'graphcast_vs_ifs_analysis':    'gs://$BUCKET/easyuq/pc/graphcast_240x121_vs_ifs_analysis.zarr',
    'pangu_vs_ifs_analysis':        'gs://$BUCKET/easyuq/pc/pangu_240x121_vs_ifs_analysis.zarr',
    'ifs_hres_vs_ifs_analysis':     'gs://$BUCKET/easyuq/pc/ifs_hres_240x121_vs_ifs_analysis.zarr',
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
        .rename(rename_crps)
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

"""
Get CRPS of IFS ENS for each grid point using the official evaluation script from WeatherBench 2

python weatherbench2/scripts/evaluate.py \
--forecast_path=gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr \
--obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
--climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
--output_dir=gs://$BUCKET/baseline/ \
--output_file_prefix=ifs_ens_240x121_surface_vs_era5_2020_ \
--input_chunks=init_time=1,lead_time=1 \
--eval_configs=probabilistic_spatial \
--variables=2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr \
--use_beam=True \
--runner=DataflowRunner \
-- \
--project=$PROJECT \
--region=$REGION \
--job_name=evaluate-ifs-ens-240x121-vs-era5-surface \
--temp_location=gs://$BUCKET/tmp/ \
--setup_file=./wb2_eval_setup.py \
--worker_machine_type=c3-highmem-8 \
--autoscaling_algorithm=THROUGHPUT_BASED
"""
with ProgressBar():
    ifs_ens_vs_era5_crps = xr.open_zarr('gs://wb2_pp/baseline/ifs_ens_240x121_surface_vs_era5_2020_probabilistic_spatial.zarr').sel(lead_time=lead_times, metric='crps').load()
ifs_ens_vs_era5_crps.to_netcdf('results/ifs_ens_vs_era5_crps.nc')

# with ProgressBar():
#     ifs_ens_vs_ifs_analysis_crps = xr.open_zarr('gs://wb2_pp/baseline/ifs_ens_240x121_surface_vs_ifs_analysis_2020_probabilistic_spatial.zarr').sel(lead_time=lead_times, metric='crps').load()
# ifs_ens_vs_ifs_analysis_crps.to_netcdf('results/ifs_ens_vs_ifs_analysis_crps.nc')