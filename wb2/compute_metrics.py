'''
compute_metrics.py

Evaluate 72-hour 850 hPa temperature forecasts (IFS-HRES, Pangu, GraphCast, ERA5 climatology, persistence) against ERA5 in 2020, computing MSE, ACC, CPA, PC, and PCS metrics, and saving results to NetCDF.
'''
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata
from isodisreg import idr
from pc.compute_pc0 import pc0_along_time
from dask.diagnostics import ProgressBar

def apply_timeseries_metric(func, obs, pred, metric_name) -> xr.DataArray:
    '''
    Apply a 1-D (time-series) metric to each grid cell.
    '''    
    return xr.apply_ufunc(
        func, 
        obs, 
        pred,
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[]],
        vectorize=True,
        join='inner',       
        dask='parallelized',
        output_dtypes=[float]
    ).expand_dims(metric=[metric_name])

def mse_field(obs_ds, pred_ds):
    # Align on time
    obs, pred = xr.align(obs_ds, pred_ds, join='inner')
    
    # Errors
    err = pred - obs
    
    return (err ** 2).mean(dim='time').expand_dims(metric=['mse'])

def acc_field(obs_ds, pred_ds, clim_ds):
    '''ACC at each (lat, lon) over time dimension.'''
    # Align on time
    obs, pred, clim = xr.align(obs_ds, pred_ds, clim_ds, join='inner')
    
    # Anomalies
    pred_anomaly = pred - clim
    obs_anomaly = obs - clim

    # ACC
    num   = (pred_anomaly * obs_anomaly).sum('time')
    denom = np.sqrt((pred_anomaly**2).sum('time') * (obs_anomaly**2).sum('time'))

    return xr.where(denom == 0, 0.0, (num / denom)).expand_dims(metric=['acc'])

def _cpa(obs, pred):
    # check for NaNs and infs
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() <= 1:
        return np.nan
        
    o = obs[valid]
    f = pred[valid]

    obs_rank    = rankdata(o, method='average')
    pred_rank    = rankdata(f, method='average')
    obs_classes = rankdata(o, method='dense')

    try:
        var_cpa = np.cov(obs_classes, obs_rank)[0, 1]
        if var_cpa == 0:
            return np.nan
        return (np.cov(obs_classes, pred_rank)[0, 1] / var_cpa + 1) / 2
    except Exception:
        return np.nan

def _pc(obs, pred):
    # check for NaNs and infs
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() <= 1:
        return np.nan
        
    o = np.asarray(obs[valid], dtype=float)
    p = np.asarray(pred[valid], dtype=float)
    
    try:
        p_df = pd.DataFrame({'pred': p}, columns=['pred'])
        fitted_idr = idr(o, p_df)
        easyuq_pred = fitted_idr.predict(digits=8)
        return np.mean(easyuq_pred.crps(o))
    except Exception:
        return np.nan

def main():
    # Define variable, level, lead time, test time
    var = 'temperature'
    level = 850
    lead_time = np.timedelta64(3, 'D')
    test_time = slice('2020-01-01', '2020-12-31')
    chunks = {'time': -1, 'latitude': 11, 'longitude': 20}
 
    # Forecasts
    ifs_hres = xr.open_zarr(
        store='gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr', 
        decode_timedelta=True,
        storage_options={'token': 'anon'}
    ).sel(
        time=test_time,
        prediction_timedelta=lead_time,
        level=level
    )[[var]].chunk(chunks)
    
    pangu = xr.open_zarr(
        store='gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr', 
        decode_timedelta=True,
        storage_options={'token': 'anon'}
    ).sel(
        time=test_time,
        prediction_timedelta=lead_time,
        level=level
    )[[var]].chunk(chunks)
    
    graphcast = xr.open_zarr(
        store='gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr', 
        decode_timedelta=True,
        storage_options={'token': 'anon'}
    ).sel(
        time=test_time,
        prediction_timedelta=lead_time,
        level=level
    )[[var]].chunk(chunks)

    # Ground Truth
    era5_2020 = xr.open_zarr(
        store='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr',
        storage_options={'token': 'anon'}
    ).sel(
        time=slice(ifs_hres.time[0], ifs_hres.time[-1] + lead_time),
        level=level
    )[[var]]
    # only 00 UTC and 12 UTC needed
    era5 = era5_2020.sel(time=(era5_2020.time.dt.hour == 0) | (era5_2020.time.dt.hour == 12)).chunk(chunks)

    # Persistence forecast
    era5_persistence_valid = era5.sel(time=test_time).assign_coords(time=era5.sel(time=test_time).time + lead_time).chunk(chunks)

    # Assign valid time
    ifs_hres_valid = ifs_hres.assign_coords(time=ifs_hres.time + lead_time).drop_vars('prediction_timedelta', errors='ignore')
    pangu_valid = pangu.assign_coords(time=pangu.time + lead_time).drop_vars('prediction_timedelta', errors='ignore')
    graphcast_valid = graphcast.assign_coords(time=graphcast.time + lead_time).drop_vars('prediction_timedelta', errors='ignore')

    # ERA5 climatology forecast
    era5_climatology = xr.open_zarr(
        store='gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr', 
        storage_options={'token': 'anon'}
    ).sel(hour=[0,12], level=level)[[var]]

    time_index = pd.date_range(start='2020-01-01 00:00', end='2020-12-31 12:00', freq='12h')

    era5_climatology = (
        era5_climatology
        .stack(datetime=('dayofyear', 'hour'))
        .assign_coords(time=('datetime', time_index))
        .swap_dims({'datetime': 'time'})
        .drop_vars(['datetime', 'dayofyear', 'hour'])
    )

    delta = era5_climatology.time.diff('time').isel(time=0)
    shift_steps = int(lead_time / delta)
    era5_climatology_valid = era5_climatology.roll(time=-shift_steps, roll_coords=False).assign_coords(time=ifs_hres.time + lead_time).chunk(chunks)

    # PC^(0)
    era5_pc0 = era5.sel(time=ifs_hres.time + lead_time).drop_vars('prediction_timedelta', errors='ignore').map(pc0_along_time)

    # Compute metrics    
    ifs_hres_mse = mse_field(era5, ifs_hres_valid)
    ifs_hres_acc = acc_field(era5, ifs_hres_valid, era5_climatology_valid)
    ifs_hres_cpa = apply_timeseries_metric(_cpa, era5, ifs_hres_valid, 'cpa')
    ifs_hres_pc = apply_timeseries_metric(_pc, era5, ifs_hres_valid, 'pc')
    ifs_hres_pcs = ((era5_pc0 - ifs_hres_pc.sel(metric='pc')) / era5_pc0).expand_dims(metric=['pcs'])

    pangu_mse = mse_field(era5, pangu_valid)
    pangu_acc = acc_field(era5, pangu_valid, era5_climatology_valid)
    pangu_cpa = apply_timeseries_metric(_cpa, era5, pangu_valid, 'cpa')
    pangu_pc = apply_timeseries_metric(_pc, era5, pangu_valid, 'pc')
    pangu_pcs = ((era5_pc0 - pangu_pc.sel(metric='pc')) / era5_pc0).expand_dims(metric=['pcs'])

    graphcast_mse = mse_field(era5, graphcast_valid)
    graphcast_acc = acc_field(era5, graphcast_valid, era5_climatology_valid)
    graphcast_cpa = apply_timeseries_metric(_cpa, era5, graphcast_valid, 'cpa')
    graphcast_pc = apply_timeseries_metric(_pc, era5, graphcast_valid, 'pc')
    graphcast_pcs = ((era5_pc0 - graphcast_pc.sel(metric='pc')) / era5_pc0).expand_dims(metric=['pcs'])

    era5_climatology_mse = mse_field(era5, era5_climatology_valid)
    era5_climatology_cpa = apply_timeseries_metric(_cpa, era5, era5_climatology_valid, 'cpa')
    era5_climatology_pc = apply_timeseries_metric(_pc, era5, era5_climatology_valid, 'pc')
    era5_climatology_pcs = ((era5_pc0 - era5_climatology_pc.sel(metric='pc')) / era5_pc0).expand_dims(metric=['pcs'])

    era5_persistence_mse = mse_field(era5, era5_persistence_valid)
    era5_persistence_acc = acc_field(era5, era5_persistence_valid, era5_climatology_valid)
    era5_persistence_cpa = apply_timeseries_metric(_cpa, era5, era5_persistence_valid, 'cpa')
    era5_persistence_pc = apply_timeseries_metric(_pc, era5, era5_persistence_valid, 'pc')
    era5_persistence_pcs = ((era5_pc0 - era5_persistence_pc.sel(metric='pc')) / era5_pc0).expand_dims(metric=['pcs'])

    # Stack metrics for along the 'metric' dimension
    ifs_hres_metrics = xr.concat(
        [
            ifs_hres_mse, 
            ifs_hres_acc,
            ifs_hres_cpa,
            ifs_hres_pc,
            ifs_hres_pcs
        ], 
        dim='metric'
    )
    pangu_metrics = xr.concat(
        [
            pangu_mse, 
            pangu_acc,
            pangu_cpa,
            pangu_pc,
            pangu_pcs
        ], 
        dim='metric'
    )
    graphcast_metrics = xr.concat(
        [
            graphcast_mse, 
            graphcast_acc,
            graphcast_cpa,
            graphcast_pc,
            graphcast_pcs
        ], 
        dim='metric'
    )
    era5_climatology_metrics = xr.concat(
        [
            era5_climatology_mse,
            era5_climatology_cpa,
            era5_climatology_pc,
            era5_climatology_pcs
        ], 
        dim='metric'
    )
    era5_persistence_metrics = xr.concat(
        [
            era5_persistence_mse, 
            era5_persistence_acc,
            era5_persistence_cpa,
            era5_persistence_pc,
            era5_persistence_pcs
        ], 
        dim='metric'
    )

    # Save results
    with ProgressBar():
        ifs_hres_metrics.to_netcdf('results/ifs_hres_metrics_72h_T850.nc')
        pangu_metrics.to_netcdf('results/pangu_metrics_72h_T850.nc')
        graphcast_metrics.to_netcdf('results/graphcast_metrics_72h_T850.nc')
        era5_climatology_metrics.to_netcdf('results/era5_climatology_metrics_72h_T850.nc')
        era5_persistence_metrics.to_netcdf('results/era5_persistence_metrics_72h_T850.nc')
    
if __name__ == '__main__':
    main()