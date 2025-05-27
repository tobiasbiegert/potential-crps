"""
Weather Forecast Metric Computation Script

This script computes various meteorological forecast verification metrics
including RMSE, CPA, PC, PCS, and ACC for different weather forecast models.
"""

import xarray as xr
import numpy as np
import isodisreg
from isodisreg import idr
from scipy import stats
import scipy
import pandas as pd
import properscoring as ps
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a forecast model."""
    name: str
    file_path: str
    lead_time: Optional[int] = None
    time_offset_hours: Optional[int] = None
    variable_name: str = 't'


class MetricComputer:
    """Class to compute meteorological forecast verification metrics."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the metric computer.
        
        Args:
            output_dir: Directory to save computed metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def cpa(obs: np.ndarray, fct: np.ndarray) -> float:
        """
        Compute Centered Pattern Accuracy.
        
        Args:
            obs: Observed values
            fct: Forecast values
            
        Returns:
            CPA value
        """
        if len(obs) <= 1 or len(fct) <= 1:
            return np.nan
            
        # Handle NaN values
        valid_idx = ~np.isnan(obs) & ~np.isnan(fct)
        if np.sum(valid_idx) <= 1:
            return np.nan
            
        obs_valid = obs[valid_idx]
        fct_valid = fct[valid_idx]
        
        obs_rank = rankdata(obs_valid, method='average')
        fct_rank = rankdata(fct_valid, method='average')
        obs_classes = rankdata(obs_valid, method='dense')
        
        try:
            var_cpa = np.cov(obs_classes, obs_rank)[0][1]
            if var_cpa == 0:
                return np.nan
            return (np.cov(obs_classes, fct_rank)[0][1] / var_cpa + 1) / 2
        except:
            return np.nan

    @staticmethod
    def rmse(obs: np.ndarray, fct: np.ndarray) -> float:
        """
        Compute Root Mean Square Error.
        
        Args:
            obs: Observed values
            fct: Forecast values
            
        Returns:
            RMSE value
        """
        valid_idx = ~np.isnan(obs) & ~np.isnan(fct)
        if np.sum(valid_idx) == 0:
            return np.nan
        return np.sqrt(np.mean((obs[valid_idx] - fct[valid_idx]) ** 2))

    @staticmethod
    def acc(obs: np.ndarray, fct: np.ndarray, clim: np.ndarray) -> float:
        """
        Compute Anomaly Correlation Coefficient using your original method.
        
        Args:
            obs: Observed values
            fct: Forecast values
            clim: Climatological values
            
        Returns:
            ACC value
        """
        # Handle NaN values
        valid_idx = ~np.isnan(obs) & ~np.isnan(fct) & ~np.isnan(clim)
        if np.sum(valid_idx) == 0:
            return np.nan
            
        obs_valid = obs[valid_idx]
        fct_valid = fct[valid_idx]
        clim_valid = clim[valid_idx]
        
        # Compute anomalies (your original method)
        forecast_anom = fct_valid - clim_valid
        truth_anom = obs_valid - clim_valid
        
        numerator = np.sum(forecast_anom * truth_anom)
        fct_std = np.sqrt(np.sum(forecast_anom ** 2))
        obs_std = np.sqrt(np.sum(truth_anom ** 2))
        
        if fct_std == 0 or obs_std == 0:
            return 0.0
            
        return numerator / (fct_std * obs_std)

    @staticmethod
    def pc(obs: np.ndarray, fct: np.ndarray) -> float:
        """
        Compute Pattern Correlation using isotonic distributional regression.
        
        Args:
            obs: Observed values
            fct: Forecast values
            
        Returns:
            PC value (CRPS-based)
        """
        try:
            # Handle NaN values
            valid_idx = ~np.isnan(obs) & ~np.isnan(fct)
            if np.sum(valid_idx) <= 1:
                return np.nan
                
            obs_valid = obs[valid_idx]
            fct_valid = fct[valid_idx]
            
            fct_test = pd.DataFrame({"fore": fct_valid}, columns=["fore"])
            out = idr(obs_valid, fct_test)
            out_fit = out.predict()
            return np.mean(out_fit.crps(obs_valid))
        except Exception as e:
            logger.warning(f"Error computing PC: {e}")
            return np.nan

    @staticmethod
    def pcs(obs: np.ndarray, fct: np.ndarray) -> float:
        """
        Compute Pattern Correlation Skill score.
        
        Args:
            obs: Observed values
            fct: Forecast values
            
        Returns:
            PCS value
        """
        try:
            # Handle NaN values
            valid_idx = ~np.isnan(obs) & ~np.isnan(fct)
            if np.sum(valid_idx) <= 1:
                return np.nan
                
            obs_valid = obs[valid_idx]
            fct_valid = fct[valid_idx]
            
            fct_test = pd.DataFrame({"fore": fct_valid}, columns=["fore"])
            
            def p03(y):
                """Compute reference skill score."""
                n = len(y)
                diff_matrix = np.abs(y[:, np.newaxis] - y)
                return np.sum(diff_matrix) / (2 * n * n)

            p0_val = p03(obs_valid)
            
            out = idr(obs_valid, fct_test)
            out_fit = out.predict()
            p1 = np.mean(out_fit.crps(obs_valid))
            
            if p0_val == 0:
                return 0.0
                
            return (p0_val - p1) / p0_val
        except Exception as e:
            logger.warning(f"Error computing PCS: {e}")
            return np.nan

    @staticmethod
    def pc_pcs(obs: np.ndarray, fct: np.ndarray) -> Tuple[float, float]:
        """
        Compute both PC and PCS in one call for efficiency.
        
        Args:
            obs: Observed values
            fct: Forecast values
            
        Returns:
            Tuple of (PC, PCS) values
        """
        try:
            # Handle NaN values
            valid_idx = ~np.isnan(obs) & ~np.isnan(fct)
            if np.sum(valid_idx) <= 1:
                return np.nan, np.nan
                
            obs_valid = obs[valid_idx]
            fct_valid = fct[valid_idx]
            
            fct_test = pd.DataFrame({"fore": fct_valid}, columns=["fore"])
            
            def p03(y):
                n = len(y)
                diff_matrix = np.abs(y[:, np.newaxis] - y)
                return np.sum(diff_matrix) / (2 * n * n)

            p0_val = p03(obs_valid)
            
            out = idr(obs_valid, fct_test)
            out_fit = out.predict()
            p1 = np.mean(out_fit.crps(obs_valid))
            
            pc_val = p1
            pcs_val = (p0_val - p1) / p0_val if p0_val != 0 else 0.0
            
            return pc_val, pcs_val
        except Exception as e:
            logger.warning(f"Error computing PC/PCS: {e}")
            return np.nan, np.nan

    def compute_metric_grid(self, obs: xr.Dataset, fct: xr.Dataset, 
                           metric_func: callable, clim: Optional[xr.Dataset] = None,
                           var_name: str = 't') -> np.ndarray:
        """
        Compute a metric across the spatial grid and average over longitude.
        
        Args:
            obs: Observed dataset
            fct: Forecast dataset
            metric_func: Metric function to compute
            clim: Climatological dataset (optional, needed for ACC)
            var_name: Variable name to use
            
        Returns:
            Array of metric values averaged over longitude
        """
        lats = obs.lat.values
        lons = obs.lon.values
        metric_vals = np.zeros((len(lats), len(lons)))

        # Check if metric requires climatology
        requires_clim = metric_func.__name__ == 'acc'
        
        if requires_clim and clim is None:
            raise ValueError("ACC metric requires climatology data")

        logger.info(f"Computing {metric_func.__name__} over {len(lats)} x {len(lons)} grid...")
        
        if requires_clim:
            try:
                clim_aligned = clim.sel(time=obs.time.values)
            except Exception as e:
                logger.error(f"Could not align climatology: {e}")
                raise

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                try:
                    obs_vals = obs.sel(lat=lat, lon=lon)[var_name].values
                    fct_vals = fct.sel(lat=lat, lon=lon)[var_name].values
                    
                    # Ensure same shape (safety check)
                    if obs_vals.shape != fct_vals.shape:
                        logger.warning(f"Shape mismatch at lat={lat}, lon={lon}: obs={obs_vals.shape}, fct={fct_vals.shape}")
                        metric_vals[i, j] = np.nan
                        continue
                    
                    if requires_clim:
                        clim_vals = clim_aligned.sel(lat=lat, lon=lon)[var_name].values
                        metric_vals[i, j] = metric_func(obs_vals, fct_vals, clim_vals)
                    else:
                        metric_vals[i, j] = metric_func(obs_vals, fct_vals)
                        
                except Exception as e:
                    logger.warning(f"Error at lat={lat}, lon={lon}: {e}")
                    metric_vals[i, j] = np.nan

        # Average over longitude dimension
        lon_mean = np.nanmean(metric_vals, axis=1)
        
        return lon_mean

    def process_model(self, model_name: str, obs: xr.Dataset, fct: xr.Dataset, 
                     clim: Optional[xr.Dataset] = None, var_name: str = 't') -> Dict[str, np.ndarray]:
        """
        Process all metrics for a given model and save results.
        
        Args:
            model_name: Name of the model
            obs: Observed dataset
            fct: Forecast dataset
            clim: Climatological dataset (optional)
            var_name: Variable name to use
            
        Returns:
            Dictionary of computed metrics
        """
        logger.info(f"Processing {model_name}...")
        
        # Define metrics to compute
        
        metrics = {
            'rmse': self.rmse,
            'cpa': self.cpa,
            'pc': self.pc,
            'pcs': self.pcs
        }

        # Add ACC if climatology is provided
        if clim is not None:
            metrics['acc'] = self.acc
        
        results = {}
        
        for metric_name, metric_func in metrics.items():
            logger.info(f"  Computing {metric_name}...")
            try:
                result = self.compute_metric_grid(obs, fct, metric_func, clim, var_name)
                results[metric_name] = result
                
                # Save to file
                filename = self.output_dir / f"{model_name}_{metric_name}.txt"
                np.savetxt(filename, result, fmt='%.6f')
                logger.info(f"  Saved to {filename}")
                
            except Exception as e:
                logger.error(f"  Error computing {metric_name} for {model_name}: {e}")
                results[metric_name] = None
        
        return results


class WeatherDataLoader:
    """Class to handle loading and preprocessing of weather data."""
    
    def __init__(self, base_dir: str):
        """
        Initialize the data loader.
        
        Args:
            base_dir: Base directory containing weather data
        """
        self.base_dir = Path(base_dir)
        
    def load_model_data(self, config: ModelConfig) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Load and align model data using your exact method.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (aligned_obs, aligned_fct) datasets
        """
        logger.info(f"Loading {config.name} data...")
        
        # Load validation data using your exact method
        dates = pd.date_range(start='2017-01-01', end='2018-12-31 12:00:00', freq='12H')
        t850_valid17 = xr.open_dataset(str(self.base_dir / "temperature_850hPa_2017_5.625deg.nc")).drop('level')
        t850_valid18 = xr.open_dataset(str(self.base_dir / "temperature_850hPa_2018_5.625deg.nc")).drop('level')
        t850_valid1718 = xr.concat([t850_valid17, t850_valid18], dim='time')
        t850_valid1718 = t850_valid1718.sel(time=dates)
        
        # Load forecast data
        file_path = Path(config.file_path)
        if not file_path.exists():
            potential_files = list(self.base_dir.glob(f"**/{file_path.name}"))
            if potential_files:
                file_path = potential_files[0]
            else:
                raise FileNotFoundError(f"Could not find {config.file_path}")
        
        fct_data = xr.open_dataset(file_path)
        
        # Apply model-specific processing and alignment using your exact methods
        if config.name == "CNN":
            # CNN alignment (your method)
            common_times = np.intersect1d(t850_valid1718.time.values, fct_data.time.values)
            obs_aligned = t850_valid1718.sel(time=common_times)
            fct_aligned = fct_data.sel(time=common_times)
            
        elif config.name == "Persistence":
            # Persistence alignment (your method)
            fct_data = fct_data.sel(lead_time=3*24)
            common_times = np.intersect1d(t850_valid1718.time.values, fct_data.time.values + pd.Timedelta(hours=3*24))
            fct_data = fct_data.assign_coords({'time': fct_data.time.values + pd.Timedelta(hours=3*24)})
            obs_aligned = t850_valid1718.sel(time=common_times)
            fct_aligned = fct_data.sel(time=common_times)
            
        elif config.name in ["T42", "T63"]:
            # T42/T63 alignment (similar to persistence)
            fct_data = fct_data.sel(lead_time=3*24)
            common_times = np.intersect1d(t850_valid1718.time.values, fct_data.time.values + pd.Timedelta(hours=3*24))
            fct_data = fct_data.assign_coords({'time': fct_data.time.values + pd.Timedelta(hours=3*24)})
            obs_aligned = t850_valid1718.sel(time=common_times)
            fct_aligned = fct_data.sel(time=common_times)
            
        else:
            # Default alignment (CNN method)
            common_times = np.intersect1d(t850_valid1718.time.values, fct_data.time.values)
            obs_aligned = t850_valid1718.sel(time=common_times)
            fct_aligned = fct_data.sel(time=common_times)
        
        logger.info(f"Aligned {config.name}: obs={len(obs_aligned.time)}, fct={len(fct_aligned.time)} time steps")
        
        return obs_aligned, fct_aligned


def main():
    """Main function to run the metric computation pipeline."""
    
    # Use relative paths
    base_dir = "./wb_data"
    output_dir = "./metrics"
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize components
    data_loader = WeatherDataLoader(base_dir)
    metric_computer = MetricComputer(output_dir)
    
    try:
        # Load climatology for ACC computation
        clim_data = None
        clim_file = Path(base_dir) / "weekly_climatology_5.625.nc"
        if clim_file.exists():
            logger.info("Loading climatology data...")
            clim_data = xr.open_dataset(clim_file)
        
        # Define model configurations based on downloaded files
        model_configs = [
            ModelConfig(
                name="CNN",
                file_path=str(Path(base_dir) / "fccnn_3d.nc"),
                variable_name='t'
            ),
            
            ModelConfig(
                name="Persistence",
                file_path=str(Path(base_dir) / "persistence_5.625.nc"),
                variable_name='t'
            ),
            
            ModelConfig(
                name="LinearRegression",
                file_path=str(Path(base_dir) / "lr_3d_t_t.nc"),
                variable_name='t'
            ),
            
            ModelConfig(
                name="T42",
                file_path=str(Path(base_dir) / "t42_5.625deg.nc"),
                variable_name='t'
            ),
            
            ModelConfig(
                name="T63",
                file_path=str(Path(base_dir) / "t63_5.625deg.nc"),
                variable_name='t'
            )
        ]
        
        logger.info(f"Processing {len(model_configs)} models...")
        
        # Process each model
        for config in model_configs:
            try:
                # Load and align model data using your exact method
                obs_aligned, fct_aligned = data_loader.load_model_data(config)
                
                # Compute metrics
                results = metric_computer.process_model(
                    model_name=config.name,
                    obs=obs_aligned,
                    fct=fct_aligned,
                    clim=clim_data,
                    var_name=config.variable_name
                )
                
                logger.info(f"Completed processing {config.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {config.name}: {e}")
                continue
        
        logger.info("All models processed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()