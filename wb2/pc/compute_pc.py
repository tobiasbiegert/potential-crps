import logging
from absl import app
from absl import flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam
import numpy as np
import dask.array as da

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define command-line flags
PREDICTION_PATH = flags.DEFINE_string(
    'prediction_path',
    None,
    help='Path to forecasts to evaluate in Zarr format',
)
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to ground-truth to evaluate in Zarr format',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help='ISO 8601 timestamp (inclusive) at which to start',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help='ISO 8601 timestamp (inclusive) at which to stop',
)
VARIABLES = flags.DEFINE_list(
    'variables', 
    ['2m_temperature'], 
    help='Variables to compute PC for'
)
LEVELS = flags.DEFINE_list(
    'levels',
    [500, 700, 850],
    help='Comma delimited list of pressure levels to select.'
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path', 
    None, 
    help='Where to save results.'
)
CHUNK_SIZE_LAT = flags.DEFINE_integer(
    'chunk_size_lat', 
    -1, 
    help='Chunk size for latitude dimension'
)
CHUNK_SIZE_LON = flags.DEFINE_integer(
    'chunk_size_lon', 
    -1, 
    help='Chunk size for longitude dimension'
)
RUNNER = flags.DEFINE_string(
    'runner', 
    None, 
    help='beam.runners.Runner'
)
SKIP_NON_HEADLINE = flags.DEFINE_bool(
    'skip_non_headline',
    True,
    help='If True, skip variable-level combinations not defined in VARIABLE_LEVEL_MAPPING (WB2 headline scores).'
)
RECHUNK = flags.DEFINE_bool(
    'rechunk',
    False,
    help='Whether to apply rechunk transform to the dataset before processing.'
)
RESTRICT_LEAD_TIMES = flags.DEFINE_bool(
    'restrict_lead_times',
    True,
    help='If True, restrict computation to specific prediction_timedelta values (1, 3, 5, 7, and 10 days).'
)
VARIABLE_LEVEL_MAPPING = {
    '2m_temperature': None,
    'mean_sea_level_pressure': None,
    '10m_wind_speed': None,
    'temperature': 850,
    'specific_humidity': 700,
    'geopotential': 500,
    'wind_speed': 850
}

def compute_pc(key, prediction_chunk, targets=None, variable_level_mapping=None, skip_non_headline=True):
    """
    Compute PC for a single chunk of model output.
    
    Parameters:
        key (xarray_beam.ChunkKey): Metadata key identifying this chunk.
        prediction_chunk (xarray.Dataset): Forecast data for one chunk.
        targets (xarray.Dataset): Target data.
        ariable_level_mapping (dict): Maps variable names → headline level for WB2.
        skip_non_headline (bool): Skip non-headline levels if True.
    
    Returns:
        (key, xarray.Dataset): The same key, plus a Dataset containing PC (and timing) metrics.
    """
    # Import necessary modules
    import numpy as np
    import xarray as xr
    from pc.easyuq_helper import compute_easyuq

    # Drop any time-offset
    key = key.with_offsets(time=None)

    # Restrict forecast times so that forecast_time + lead_time lies within the target window (not necessary for 2020 data)
    start_time_bound = (targets.time[0] - prediction_chunk.prediction_timedelta).values[0]
    end_time_bound = (targets.time[-1] - prediction_chunk.prediction_timedelta).values[-1]
    prediction_chunk = prediction_chunk.sel(time=slice(start_time_bound, end_time_bound))

    # Extract the variable name from the key
    var, = key.vars
    
    # Get the lead time of the chunk. This assumes that 'prediction_timedelta' contains only a single lead time value. Therefore the chunk size along this dimension must be 1.
    lead_time = prediction_chunk.prediction_timedelta.values[0]

    # Skip TP24hr for lead times of less than 24 hours
    if (var == 'total_precipitation_24hr') & (lead_time < 86400000000000):
        logging.info(f'Skipping {key}')
        metrics = np.full((
            len(prediction_chunk.longitude), 
            len(prediction_chunk.latitude), 
            len(prediction_chunk.prediction_timedelta),
            3), np.nan, dtype=np.float32)
        coords = {
            'longitude': prediction_chunk.longitude,
            'latitude': prediction_chunk.latitude,
            'prediction_timedelta': prediction_chunk.prediction_timedelta,
            'metric': ['pc', 'idr_time', 'eval_time']
        }
        easyuq_dataset = xr.Dataset(
            {var: (('longitude', 'latitude', 'prediction_timedelta', 'metric'), metrics)},
            coords=coords
        )
        return key, easyuq_dataset
    
    # Align target chunk with the valid prediction times
    valid_time = prediction_chunk.time + lead_time
    target_chunk = targets.sel(time=valid_time, latitude=prediction_chunk.latitude, longitude=prediction_chunk.longitude)
    
    if 'level' in prediction_chunk.dims:
        # Upper-level variable
        level = prediction_chunk.level.values[0]
        headline_level = variable_level_mapping.get(var)
        # Skip if we only want the WB2 headline score levels
        if (level != headline_level) & skip_non_headline:
            logging.info(f'Skipping {key}')
            metrics = np.full((
                len(prediction_chunk.longitude), 
                len(prediction_chunk.latitude), 
                len(prediction_chunk.prediction_timedelta), 
                len(prediction_chunk.level),
                3), np.nan, dtype=np.float32)
            coords={
                'longitude': prediction_chunk.longitude,
                'latitude': prediction_chunk.latitude,
                'prediction_timedelta': prediction_chunk.prediction_timedelta,
                'level': prediction_chunk.level,
                'metric': ['pc', 'idr_time', 'eval_time']
            }
            easyuq_dataset = xr.Dataset(
                {var: (('longitude', 'latitude', 'prediction_timedelta', 'level', 'metric'), metrics)},
                coords=coords
            )
        else:
            logging.info(f'Processing {key}')
            preds = prediction_chunk.sel(level=level)[var].compute()
            obs = target_chunk.sel(level=level)[var].compute()
            logging.info(f'Loading complete for {key}')
            easyuq_dataset = compute_easyuq(preds, obs, var, lead_time, level=level)
            logging.info(f'PC results for {key}')
    else:
        # Surface variable
        logging.info(f'Processing {key}')
        preds = prediction_chunk[var].compute()
        obs = target_chunk[var].compute()
        logging.info(f'Loading complete for {key}')
        easyuq_dataset = compute_easyuq(preds, obs, var, lead_time)
        logging.info(f'PC results for {key}')

    return key, easyuq_dataset

def main(argv):
    # Open model outputs and get input chunks
    predictions, input_chunks = xbeam.open_zarr(PREDICTION_PATH.value)

    logging.info('Selecting variables, lead times and time.')
    
    predictions = predictions[VARIABLES.value].sel(time=slice(TIME_START.value, TIME_STOP.value))

    if RESTRICT_LEAD_TIMES.value:
        # Subset to only the desired lead times
        lead_times = [
            np.timedelta64(1, 'D'),
            np.timedelta64(3, 'D'),
            np.timedelta64(5, 'D'),
            np.timedelta64(7, 'D'),
            np.timedelta64(10, 'D')
        ]
        predictions = predictions.sel(prediction_timedelta=lead_times)

    # Define working chunks
    working_chunks = {'longitude': CHUNK_SIZE_LON.value, 'latitude': CHUNK_SIZE_LAT.value, 'prediction_timedelta': 1, 'level': 1, 'time': -1}

    if "level" not in predictions.dims:
        input_chunks.pop("level")
        working_chunks.pop("level")
    else:
        predictions = predictions.sel(level=[int(level) for level in LEVELS.value])

    # Remove time from output chunks
    output_chunks = working_chunks.copy()
    output_chunks.pop('time')
        
    # Open target data
    logging.info('Opening targets.')
    targets = xr.open_zarr(TARGET_PATH.value)[VARIABLES.value]

    # Create template
    templates = []

    for var in VARIABLES.value:
        if 'level' in predictions[var].dims:
            template = xr.Dataset({
                var: (('longitude', 'latitude', 'prediction_timedelta', 'level', 'metric'), da.full((
                    len(predictions.longitude), 
                    len(predictions.latitude), 
                    len(predictions.prediction_timedelta), 
                    len(predictions.level), 
                    3
                ), np.nan, dtype=np.float32)),
            }, coords={
                'longitude': predictions.longitude,
                'latitude': predictions.latitude,
                'prediction_timedelta': predictions.prediction_timedelta,
                'level': predictions.level,
                'metric': ['pc', 'idr_time', 'eval_time']
            }).chunk({'longitude': CHUNK_SIZE_LON.value, 'latitude': CHUNK_SIZE_LAT.value, 'prediction_timedelta': 1, 'level': 1})
            templates.append(template)
        else:
            template = xr.Dataset({
                var: (('longitude', 'latitude', 'prediction_timedelta', 'metric'), da.full((
                    len(predictions.longitude), 
                    len(predictions.latitude), 
                    len(predictions.prediction_timedelta), 
                    3), np.nan, dtype=np.float32)),
            }, coords={
                'longitude': predictions.longitude,
                'latitude': predictions.latitude,
                'prediction_timedelta': predictions.prediction_timedelta,
                'metric': ['pc', 'idr_time', 'eval_time']
            }).chunk({'longitude': CHUNK_SIZE_LON.value, 'latitude': CHUNK_SIZE_LAT.value, 'prediction_timedelta': 1})
            templates.append(template)

    final_template = xr.merge(templates)
    logging.info('Templates created.')

    # Define and run the Apache Beam pipeline
    with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:       
        if RECHUNK.value:
            (
                root
                | 'DatasetToChunks' >> xbeam.DatasetToChunks(
                    predictions,
                    split_vars=True,
                    chunks=input_chunks
                )
                | 'Rechunk' >> xbeam.Rechunk(
                    dim_sizes=predictions.sizes,
                    source_chunks=input_chunks,
                    target_chunks=working_chunks,
                    itemsize=4
                )
                | 'ComputePC' >> beam.MapTuple(
                    compute_pc,
                    targets=targets,
                    variable_level_mapping=VARIABLE_LEVEL_MAPPING,
                    skip_non_headline=SKIP_NON_HEADLINE.value
                )
                | 'ChunksToZarr' >> xbeam.ChunksToZarr(
                    OUTPUT_PATH.value,
                    template=xbeam.make_template(final_template),
                    zarr_chunks=output_chunks
                )
            )
        else:
            (
                root
                | 'DatasetToChunks' >> xbeam.DatasetToChunks(
                    predictions,
                    split_vars=True,
                    chunks=working_chunks
                )
                | 'ComputePC' >> beam.MapTuple(
                    compute_pc,
                    targets=targets,
                    variable_level_mapping=VARIABLE_LEVEL_MAPPING,
                    skip_non_headline=SKIP_NON_HEADLINE.value
                )
                | 'ChunksToZarr' >> xbeam.ChunksToZarr(
                    OUTPUT_PATH.value,
                    template=xbeam.make_template(final_template),
                    zarr_chunks=output_chunks
                )
            )

if __name__ == "__main__":
    app.run(main)