import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

def sign_flip_test(
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed_offset: int,
    *,
    b: int = 1000,
    root_seed: int = 42,
) -> np.float32:
    """
    One‑sided sign‑flip test.

    Parameters:
        score_a, score_b (1‑D array‑like): CRPS values from model A and model B
        seed_offset (int): Unique offset for the random seed of this grid point → independent but reproducible draws
        root_seed (int): Global seed
        b (int): Number of random sign permutations

    Returns:
        p (float32): Empirical one‑sided p‑value
    """
    # Check length and shape
    if score_a.shape != score_b.shape or score_a.ndim != 1:
        raise ValueError("inputs must be 1‑D arrays of identical length")

    # Compute score differences their mean
    d = score_a - score_b
    d_mean = d.mean()

    # draw a b×n matrix of ±1 signs with a deterministic seed
    rng = np.random.default_rng(root_seed + int(seed_offset))
    z = rng.choice((-1, 1), size=(b, d.size), replace=True)

    # elementwise product with differences
    m = (z * d).mean(axis=1)

    # empirical one‑sided p‑value
    p = np.mean(m <= d_mean)
    
    return p.astype("float32")

def sign_flip_test_dataset(
    ds_a: xr.Dataset,                 
    ds_b: xr.Dataset,                   
    *,                   
    b: int = 1000,                   
    seed: int = 42,                  
    variables: list[str] | None = None
) -> xr.Dataset:
    """
    Parameters:
        ds_a, ds_b (xr.Dataset): CRPS values from model A and model B
        b (int): Number of random sign permutations
        seed (int): Global seed
        variables (list): List of variables
        
    Returns:
        p_da (xr.Dataset): dataset of p‑values for each (lon, lat, lead_time) grid point
    """
    if variables is None:
        variables = list(ds_a.data_vars)

    # Align both datasets on all coords
    ds_a, ds_b = xr.align(ds_a[variables], ds_b[variables], join="exact")

    # build the seed offset array (unique per grid point)
    seed_da = xr.DataArray(
        np.arange(
            ds_a.sizes["longitude"]
            * ds_a.sizes["latitude"]
            * ds_a.sizes["prediction_timedelta"],
            dtype="uint32",
        ).reshape(
            ds_a.sizes["longitude"],
            ds_a.sizes["latitude"],
            ds_a.sizes["prediction_timedelta"],
        ),
        dims=("longitude", "latitude", "prediction_timedelta"),
        coords={
            "longitude": ds_a["longitude"],
            "latitude": ds_a["latitude"],
            "prediction_timedelta": ds_a["prediction_timedelta"],
        },
    )

    # vectorise the 1‑D test across all other dims
    p_da = xr.apply_ufunc(
        sign_flip_test,
        ds_a,
        ds_b,
        seed_da,
        kwargs=dict(b=b, root_seed=seed),
        input_core_dims=[["time"], ["time"], []],   
        output_core_dims=[[]],                  
        vectorize=True,                         
        dask="parallelized",                    
        output_dtypes=[np.float32],
    )

    return p_da

# Define lead times
lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
]
# Define variables
variables = [
    'mean_sea_level_pressure',
    '2m_temperature',
    '10m_wind_speed',
    'total_precipitation_24hr',
]
# Define chunking scheme
chunking_dict = {'longitude':8}

# Open CRPS data sets
graphcast_vs_era5_crps = xr.open_dataset('results/graphcast_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
pangu_vs_era5_crps = xr.open_dataset('results/pangu_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
ifs_hres_vs_era5_crps = xr.open_dataset('results/ifs_hres_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)

graphcast_vs_ifs_analysis_crps = xr.open_dataset('results/graphcast_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
pangu_vs_ifs_analysis_crps = xr.open_dataset('results/pangu_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
ifs_hres_vs_ifs_analysis_crps = xr.open_dataset('results/ifs_hres_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)

Run model-pair comparisons
precipitation is omitted if one of the models (depending on reference) does not include it

# GraphCast vs Pangu (ERA5 ground truth)
with ProgressBar():
    graphcast_pangu_vs_era5_p = sign_flip_test_dataset(
        graphcast_vs_era5_crps, 
        pangu_vs_era5_crps,
        variables=variables[:-1]
        ).compute()
graphcast_pangu_vs_era5_p.to_netcdf('results/graphcast_pangu_vs_era5_p.nc')

# GraphCast vs Pangu (IFS‑Analysis ground truth)
with ProgressBar():
    graphcast_pangu_vs_ifs_analysis_p = sign_flip_test_dataset(
        graphcast_vs_ifs_analysis_crps, 
        pangu_vs_ifs_analysis_crps,
        variables=variables[:-1]
        ).compute()
graphcast_pangu_vs_ifs_analysis_p.to_netcdf('results/graphcast_pangu_vs_ifs_analysis_p.nc')

# GraphCast vs IFS‑HRES (ERA5 ground truth) – only comparison with precipitation
with ProgressBar():
    graphcast_ifs_hres_vs_era5_p = sign_flip_test_dataset(
        graphcast_vs_era5_crps, 
        ifs_hres_vs_era5_crps,
        variables=variables
        ).compute()
graphcast_ifs_hres_vs_era5_p.to_netcdf('results/graphcast_ifs_hres_vs_era5_p.nc')

# GraphCast vs IFS‑HRES (IFS‑Analysis ground truth)
with ProgressBar():
    graphcast_ifs_hres_vs_ifs_analysis_p = sign_flip_test_dataset(
        graphcast_vs_ifs_analysis_crps, 
        ifs_hres_vs_ifs_analysis_crps,
        variables=variables[:-1]
        ).compute()
graphcast_ifs_hres_vs_ifs_analysis_p.to_netcdf('results/graphcast_ifs_hres_vs_ifs_analysis_p.nc')

# Pangu vs IFS‑HRES (ERA5 ground truth)
with ProgressBar():
    pangu_hres_vs_era5_p = sign_flip_test_dataset(
        pangu_vs_era5_crps, 
        ifs_hres_vs_era5_crps,
        variables=variables[:-1]
        ).compute()
pangu_hres_vs_era5_p.to_netcdf('results/pangu_hres_vs_era5_p.nc')

# Pangu vs IFS‑HRES (IFS‑Analysis ground truth)
with ProgressBar():    
    pangu_hres_vs_ifs_analysis_p = sign_flip_test_dataset(
        pangu_vs_ifs_analysis_crps, 
        ifs_hres_vs_ifs_analysis_crps,
        variables=variables[:-1]
        ).compute()
pangu_hres_vs_ifs_analysis_p.to_netcdf('results/pangu_hres_vs_ifs_analysis_p.nc')