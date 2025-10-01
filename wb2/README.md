# WeatherBench 2 Analysis

This directory contains code for analyzing and comparing different WeatherBench 2 forecast models using PCRPS and PCRPS-S.

## Prerequisites

- Python 3.10
- (Optional) A Google Cloud Platform (GCP) project and bucket as well as `gcloud` CLI if you plan to run on Dataflow.

## Dependencies
The code is written in Python and uses the following libraries (as specified in requirements.txt):

- `absl-py==2.1.0`
- `apache-beam==2.64.0`
- `Cartopy==0.23.0`
- `cftime==1.6.4.post1`
- `dask==2024.4.2`
- `fsspec==2024.3.1`
- `gcsfs==2024.3.1`
- `h5netcdf==1.3.0`
- `isodisreg @ git+https://github.com/evwalz/isodisreg.git@c293eb3a126f200bf4894ecbab6157596c2a395e`
- `matplotlib==3.8.4`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `scipy==1.13.0`
- `shapely<2.0`
- `tqdm==4.66.4`
- `weatherbench2 @ git+https://github.com/google-research/weatherbench2.git@d29e2692ecce309b52407d7914af3baa62bbe2b9`
- `xarray==2025.4.0`
- `xarray-beam==0.8.0`
- `zarr==2.17.2`

## Directory Structure
- `./data/`: Contains the ERA5 deterministic climatology forecast.
- `./results/`: Contains the results.
- `./plots/`: Contains the visualizations.
- `./pc/`
  - `compute_pc0.py`: Computes $\text{PCRPS}^{(0)}$ for ERA5 and the IFS analysis.
  - `compute_pc.py`: Script to compute PCRPS for any model.
  - `easyuq_helper.py`: Helper module for computing PCRPS.
- `./construct_era5_climatology_forecasts.py`: Constructs ERA5 deterministic climatological forecasts.
- `./compute_metrics.py`: Script to compute different metrics for 3-day T850 forecasts.
- `./create_plots.py`: Generates plots.
- `./extract_results.py`: Downloads results from GCP and transforms them.
- `./requirements.txt`: Local installation requirements.
- `./setup.py`: Installation requirements for GCP workers.
- `./test_pc.py`: Runs block permutation tests for statistical significance between model pairs.

## Usage
1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Compute $\text{PCRPS}^{(0)}$ for ERA5 and the IFS analysis:
```bash
python pc/compute_pc0.py
```

3. Download and transform the ERA5 climatological forecast and save it to the `./data/` directory:
```bash
python construct_era5_climatology_forecasts.py
```

4. Fit EasyUQ and compute CRPS.
   
   Example usage on Google Cloud Platform (GCP) with Apache Beam and Dataflow (Replace `$PROJECT`, `$BUCKET`, and `$REGION` with your GCP project ID, bucket name, and region):
```bash
python pc/compute_pc.py \
--prediction_path=gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr \
--target_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
--output_path=gs://$BUCKET/easyuq/pc/graphcast_240x121_vs_era5.zarr \
--variables=2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr \
--time_start=2020-01-01 \
--time_stop=2020-12-31 \
--chunk_size_lon=8 \
--chunk_size_lat=11 \
--runner=DataflowRunner \
-- \
--project=$PROJECT \
--region=$REGION \
--job_name=compute-pc-graphcast-240x121-vs-era5 \
--temp_location=gs://$BUCKET/tmp/ \
--staging_location=gs://$BUCKET/staging/ \
--setup_file=./setup.py \
--worker_machine_type=c3-standard-8 \
--autoscaling_algorithm=THROUGHPUT_BASED
```
For the deterministic ERA5 climatology, this is done locally using DirectRunner since the forecasts are locally in `./data/`. 
But this can be done for every model if GCP is not accessible.
```bash
python pc/compute_pc.py \
--prediction_path=/data/era5_climatology_forecasts.zarr \
--target_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
--output_path=/results/era5_climatology_240x121_vs_era5.zarr \
--variables=2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr \
--time_start=2020-01-01 \
--time_stop=2020-12-31 \
--chunk_size_lon=8 \
--chunk_size_lat=11 \
--runner=DirectRunner \
-- \
--direct_num_workers 16 \
--direct_running_mode multi_processing
```

5. Compute CRPS of IFS ENS for each grid point using the official evaluation script from WeatherBench2:
```bash
python weatherbench2/scripts/evaluate.py \
--forecast_path=gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr \
--obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
--climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
--output_dir=gs://$BUCKET/baseline/ \
--output_file_prefix=ifs_ens_240x121_vs_analysis_2020_ \
--input_chunks=init_time=1,lead_time=1 \
--eval_configs=probabilistic_spatial \
--variables=2m_temperature,mean_sea_level_pressure,10m_wind_speed \
--use_beam=True \
--runner=DataflowRunner \
-- \
--project=$PROJECT \
--region=$REGION \
--job_name=evaluate-ifs-ens-240x121-vs-analysis-surface \
--temp_location=gs://$BUCKET/tmp/ \
--setup_file=./setup.py \
--worker_machine_type=c3-highmem-8 \
--autoscaling_algorithm=THROUGHPUT_BASED
```
Next, evaluate GenCast by computing per-grid-point CRPS:
```bash
python weatherbench2/scripts/evaluate.py \
--forecast_path=gs://weatherbench2/datasets/gencast/2020-240x121_equiangular_with_poles_conservative.zarr \
--obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
--climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
--output_dir=gs://$BUCKET/baseline/ \
--output_file_prefix=gencast_240x121_surface_vs_era5_2020_ \
--input_chunks=init_time=1,lead_time=1 \
--eval_configs=probabilistic_spatial \
--variables=2m_temperature,mean_sea_level_pressure,10m_wind_speed,total_precipitation_24hr \
--ensemble_dim=sample \
--use_beam=True \
--runner=DataflowRunner \
-- \
--project=$PROJECT \
--region=$REGION \
--job_name=evaluate-gencast-240x121-vs-era5-surface \
--temp_location=gs://$BUCKET/tmp/ \
--setup_file=./setup.py \
--worker_machine_type=c3-highmem-8 \
--autoscaling_algorithm=THROUGHPUT_BASED
```

6. Download CRPS results and time measurements from GCP bucket (or local `./results/` directory), compute PCRPS, PCRPS-S, and save results:
```bash
python extract_results.py
```

7. Run block permutation tests for specific model combinations:
```bash
python test_pc.py
```

8. Compute ACC, CPA, PCRPS, PCRPS-S, and RMSE for 3-day T850 forecasts. This is done to directly compare to the results in `wb1/`. This script can also be used to compute these metrics for any single combination of variable, level and lead time. But the PCRPS implementation is not as efficient and as general as the implementation using Apache Beam.
```bash
python compute_metrics.py
```

9. Generate plots:
```bash
python create_plots.py
```

## Models Analyzed
 
- IFS HRES
- GraphCast
- GraphCast (oper.)
- Pangu-Weather
- Pangu-Weather (oper.)


