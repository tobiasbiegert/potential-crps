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
            crps = easyuq_preds_point.crps(obs_point)
            pc = np.float32(np.mean(crps))
            eval_time = np.float32(time.time() - start_time_pc)

            # Stack metrics into an array
            metrics_array = np.stack([pc, idr_time, eval_time], axis=-1)

            dims = ['metric']
            coords = {
                'metric': ['pc', 'idr_time', 'eval_time']
            }
            metrics_da = xr.DataArray(metrics_array, dims=dims, coords=coords)

            # Expand dimensions
            expand_dims = {
                'longitude': [lon],
                'latitude': [lat],
                'prediction_timedelta': preds.prediction_timedelta
            }
            if level is not None:
                expand_dims['level'] = [level]
            metrics_da = metrics_da.expand_dims(expand_dims)

            # Create and append Dataset
            easyuq_datasets.append(xr.Dataset({var: metrics_da}))
            
    # Merge datasets
    easyuq_dataset = xr.merge(easyuq_datasets)
    
    return easyuq_dataset