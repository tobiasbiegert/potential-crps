import numpy as np
import time
import xarray as xr
from isodisreg import idr

def compute_easyuq(preds, obs, var, lead_time, level=None):
    """
    Fit EasyUQ at each latitude and longitude point and compute PC.

    Parameters:
        preds (xarray.DataArray): Data array of model outputs for the variable of interest.
        obs (xarray.DataArray): Observed data array for the variable of interest.
        var (str): The variable name.
        lead_time (np.timedelta64): The prediction lead time.
        level (optional): The vertical level (if applicable) for the variable.

    Returns:
        xarray.Dataset: A dataset containing PC, training time and evaluation time for each point.
    """
    # Initialize an empty list to store the datasets for each grid point
    easyuq_datasets = []
    # Iterate over each grid point in chunk
    for lat in preds.latitude.values:
        for lon in preds.longitude.values:
            # Select data
            preds_point = preds.sel(latitude=lat, longitude=lon, prediction_timedelta=lead_time)

            # Get valid times
            valid_time = preds_point.time + lead_time

            # Extract corresponding observations       
            obs_point = obs.sel(latitude=lat, longitude=lon, time=valid_time)

            # Fit IDR
            start_time_idr = time.time()
            fitted_idr = idr(obs_point, preds_point.to_dataframe()[[var]])
            idr_time = np.float32(time.time() - start_time_idr)

            # Predict
            easyuq_preds_point = fitted_idr.predict(preds_point.to_dataframe()[[var]], digits=8)

            # Compute pc
            start_time_pc = time.time()
            crps = np.float32(easyuq_preds_point.crps(obs_point))
            # pc = np.float32(np.mean(crps))
            eval_time = np.float32(time.time() - start_time_pc)

            dims = ['time']
            coords = {
                'time': preds_point.time
            }
            da = xr.DataArray(crps, dims=dims, coords=coords)

            # Expand dimensions
            expand_dims = {
                'longitude': [lon],
                'latitude': [lat],
                'prediction_timedelta': preds.prediction_timedelta
            }
            if level is not None:
                expand_dims['level'] = [level]
            da = da.expand_dims(expand_dims)

            time_array = np.stack([idr_time, eval_time], axis=-1)
            time_dims = ['task']
            time_coords = {
                'task': ['idr_time', 'eval_time']
            }
            time_da = xr.DataArray(time_array, dims=time_dims, coords=time_coords)

            time_expand_dims = {
                'longitude': [lon],
                'latitude': [lat],
                'prediction_timedelta': preds.prediction_timedelta
            }
            if level is not None:
                time_expand_dims['level'] = [level]
            time_da = time_da.expand_dims(time_expand_dims)

            easyuq_datasets.append(
                xr.Dataset(
                    {
                        f'{var}_crps': da,
                        f'{var}_time': time_da
                    }
                )
            )
            
    # Merge datasets
    easyuq_dataset = xr.merge(easyuq_datasets)
    
    return easyuq_dataset