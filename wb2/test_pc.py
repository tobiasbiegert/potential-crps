import xarray as xr
import numpy as np
from dask.diagnostics import ProgressBar

def block_permutation_test(
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed_offset: int,
    block_length: int,
    *,
    b: int = 1000,
    root_seed: int = 42,
) -> np.float32:
    """
    One-sided block permutation test.

    Parameters:
        score_a, score_b (1-D array-like): Score (here CRPS) values from model A and model B over initialization times
        seed_offset (int): Unique offset for random seed per grid point & lead time
        block_length (int): Number of consecutive init-time steps per block (here 2, 6, 10, 14, 20)
        b (int): Number of permutation replicates
        root_seed (int): Global RNG seed

    Returns:
        float32: Empirical one-sided p-value
    """
    if score_a.shape != score_b.shape or score_a.ndim != 1:
        raise ValueError('Inputs must be 1-D arrays of identical length')

    d = score_a - score_b
    n = d.size
    d_mean = d.mean()

    # Compute number of blocks accounting for possible offset up to block_length-1
    n_blocks = int(np.ceil((n + block_length - 1) / block_length))

    # draw a b×n_blocks matrix of +/- signs with a deterministic seed
    rng = np.random.default_rng(root_seed + int(seed_offset))
    block_signs = (1 - 2 * rng.integers(0, 2, size=(b, n_blocks), dtype=np.int8)) # {+1,-1}

    # Offsets to slide block boundaries across replicates
    offsets = np.arange(b, dtype=int) % block_length

    # Determine block index for each observation in each replicate
    block_indices = (np.arange(n)[None, :] + offsets[:, None]) // block_length

    # Expand block signs to full-series signs per replicate
    signs = block_signs[np.arange(b)[:, None], block_indices]

    # Compute replicate means
    m = np.einsum('ij,j->i', signs, d) / n # signs promoted to float on the fly
    
    # one‑sided p‑value
    n_lt = np.count_nonzero(m < d_mean)
    n_gt = np.count_nonzero(m > d_mean)
    n_eq = b - n_lt - n_gt                    # compute ties robustly as remainder
    p = (n_lt + 0.5 * n_eq + 0.5) / (b + 1)
    
    return np.float32(p)

def block_permutation_test_dataset(
    ds_a: xr.Dataset,                 
    ds_b: xr.Dataset,                   
    *,                   
    b: int = 1000,                   
    seed: int = 42,                  
    variables: list[str] | None = None
) -> xr.Dataset:
    """
    Apply block-permutation test to each (lon, lat, prediction_timedelta).
    
    Parameters:
        ds_a, ds_b (xr.Dataset): Score (here CRPS) values from model A and model B
        b (int): Number of permutation replicates
        seed (int): global RNG seed
        variables (list[str], optional): which data_vars to test
        
    Returns:
        p_da (xr.Dataset): dataset of p‑values for each (lon, lat, prediction_timedelta) grid point
    """
    if variables is None:
        variables = list(ds_a.data_vars)

    # Align both datasets on all coords
    ds_a, ds_b = xr.align(ds_a[variables], ds_b[variables], join='exact')

    # build a unique seed_offset per grid‐point & lead time
    seed_da = xr.DataArray(
        np.arange(
            ds_a.sizes['longitude']
            * ds_a.sizes['latitude']
            * ds_a.sizes['prediction_timedelta'],
            dtype='uint32',
        ).reshape(
            ds_a.sizes['longitude'],
            ds_a.sizes['latitude'],
            ds_a.sizes['prediction_timedelta'],
        ),
        dims=('longitude', 'latitude', 'prediction_timedelta'),
        coords={
            'longitude': ds_a['longitude'],
            'latitude': ds_a['latitude'],
            'prediction_timedelta': ds_a['prediction_timedelta'],
        },
    )

    # convert prediction_timedelta into block_length in steps (here 12h init_time steps)
    lead_days = (ds_a['prediction_timedelta'] / np.timedelta64(1, 'D')).astype(int)
    block_length = (lead_days * 2).rename('block_length')
    block_length_da = block_length.broadcast_like(seed_da)

    # vectorise the block_permutation_test along 'time'
    p_da = xr.apply_ufunc(
        block_permutation_test,
        ds_a,
        ds_b,
        seed_da,
        block_length_da,
        kwargs=dict(b=b, root_seed=seed),
        input_core_dims=[['time'], ['time'], [], []],   
        output_core_dims=[[]],                  
        vectorize=True,                         
        dask='parallelized',                    
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
chunking_dict = {'longitude': 60, 'latitude': 11, 'prediction_timedelta': 1, 'time': -1}

def main():
    # Open CRPS data sets
    graphcast_vs_era5_crps = xr.open_dataset('results/graphcast_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    graphcast_operational_vs_era5_crps = xr.open_dataset('results/graphcast_operational_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    pangu_vs_era5_crps = xr.open_dataset('results/pangu_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    pangu_operational_vs_era5_crps = xr.open_dataset('results/pangu_operational_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    ifs_hres_vs_era5_crps = xr.open_dataset('results/ifs_hres_vs_era5_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    
    graphcast_vs_ifs_analysis_crps = xr.open_dataset('results/graphcast_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    graphcast_operational_vs_ifs_analysis_crps = xr.open_dataset('results/graphcast_operational_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    pangu_vs_ifs_analysis_crps = xr.open_dataset('results/pangu_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    pangu_operational_vs_ifs_analysis_crps = xr.open_dataset('results/pangu_operational_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    ifs_hres_vs_ifs_analysis_crps = xr.open_dataset('results/ifs_hres_vs_ifs_analysis_crps.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times).chunk(chunking_dict)
    
    # Run model-pair comparisons
    # precipitation is omitted if one of the models (depending on reference) does not include it
    
    # GraphCast vs Pangu (ERA5 ground truth)
    with ProgressBar():
        graphcast_pangu_vs_era5_p = block_permutation_test_dataset(
            graphcast_vs_era5_crps, 
            pangu_vs_era5_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_pangu_vs_era5_p.to_netcdf('results/graphcast_pangu_vs_era5_p.nc')
    
    # GraphCast vs Pangu (IFS‑Analysis ground truth)
    with ProgressBar():
        graphcast_pangu_vs_ifs_analysis_p = block_permutation_test_dataset(
            graphcast_vs_ifs_analysis_crps, 
            pangu_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_pangu_vs_ifs_analysis_p.to_netcdf('results/graphcast_pangu_vs_ifs_analysis_p.nc')
    
    # GraphCast vs IFS‑HRES (ERA5 ground truth) – with precipitation
    with ProgressBar():
        graphcast_ifs_hres_vs_era5_p = block_permutation_test_dataset(
            graphcast_vs_era5_crps, 
            ifs_hres_vs_era5_crps,
            variables=variables
            ).compute()
    graphcast_ifs_hres_vs_era5_p.to_netcdf('results/graphcast_ifs_hres_vs_era5_p.nc')
    
    # GraphCast vs IFS‑HRES (IFS‑Analysis ground truth)
    with ProgressBar():
        graphcast_ifs_hres_vs_ifs_analysis_p = block_permutation_test_dataset(
            graphcast_vs_ifs_analysis_crps, 
            ifs_hres_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_ifs_hres_vs_ifs_analysis_p.to_netcdf('results/graphcast_ifs_hres_vs_ifs_analysis_p.nc')
    
    # Pangu vs IFS‑HRES (ERA5 ground truth)
    with ProgressBar():
        pangu_hres_vs_era5_p = block_permutation_test_dataset(
            pangu_vs_era5_crps, 
            ifs_hres_vs_era5_crps,
            variables=variables[:-1]
            ).compute()
    pangu_hres_vs_era5_p.to_netcdf('results/pangu_hres_vs_era5_p.nc')
    
    # Pangu vs IFS‑HRES (IFS‑Analysis ground truth)
    with ProgressBar():    
        pangu_hres_vs_ifs_analysis_p = block_permutation_test_dataset(
            pangu_vs_ifs_analysis_crps, 
            ifs_hres_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    pangu_hres_vs_ifs_analysis_p.to_netcdf('results/pangu_hres_vs_ifs_analysis_p.nc')

    # GraphCast Operational vs Pangu Operational (ERA5 ground truth)
    with ProgressBar():
        graphcast_operational_pangu_operational_vs_era5_p = block_permutation_test_dataset(
            graphcast_operational_vs_era5_crps, 
            pangu_operational_vs_era5_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_operational_pangu_operational_vs_era5_p.to_netcdf('results/graphcast_operational_pangu_operational_vs_era5_p.nc')
    
    # GraphCast Operational vs Pangu Operational (IFS‑Analysis ground truth)
    with ProgressBar():
        graphcast_operational_pangu_operational_vs_ifs_analysis_p = block_permutation_test_dataset(
            graphcast_operational_vs_ifs_analysis_crps, 
            pangu_operational_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_operational_pangu_operational_vs_ifs_analysis_p.to_netcdf('results/graphcast_operational_pangu_operational_vs_ifs_analysis_p.nc')
    
    # GraphCast Operational vs IFS‑HRES (ERA5 ground truth) – with precipitation
    with ProgressBar():
        graphcast_operational_ifs_hres_vs_era5_p = block_permutation_test_dataset(
            graphcast_operational_vs_era5_crps, 
            ifs_hres_vs_era5_crps,
            variables=variables
            ).compute()
    graphcast_operational_ifs_hres_vs_era5_p.to_netcdf('results/graphcast_operational_ifs_hres_vs_era5_p.nc')
    
    # GraphCast Operational vs IFS‑HRES (IFS‑Analysis ground truth)
    with ProgressBar():
        graphcast_operational_ifs_hres_vs_ifs_analysis_p = block_permutation_test_dataset(
            graphcast_operational_vs_ifs_analysis_crps, 
            ifs_hres_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    graphcast_operational_ifs_hres_vs_ifs_analysis_p.to_netcdf('results/graphcast_operational_ifs_hres_vs_ifs_analysis_p.nc')
    
    # Pangu Operational vs IFS‑HRES (ERA5 ground truth)
    with ProgressBar():
        pangu_operational_ifs_hres_vs_era5_p = block_permutation_test_dataset(
            pangu_operational_vs_era5_crps, 
            ifs_hres_vs_era5_crps,
            variables=variables[:-1]
            ).compute()
    pangu_operational_ifs_hres_vs_era5_p.to_netcdf('results/pangu_operational_ifs_hres_vs_era5_p.nc')
    
    # Pangu Operational vs IFS‑HRES (IFS‑Analysis ground truth)
    with ProgressBar():    
        pangu_operational_ifs_hres_vs_ifs_analysis_p = block_permutation_test_dataset(
            pangu_operational_vs_ifs_analysis_crps, 
            ifs_hres_vs_ifs_analysis_crps,
            variables=variables[:-1]
            ).compute()
    pangu_operational_ifs_hres_vs_ifs_analysis_p.to_netcdf('results/pangu_operational_ifs_hres_vs_ifs_analysis_p.nc')

if __name__ == '__main__':
    main()