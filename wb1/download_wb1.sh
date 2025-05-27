#!/bin/bash
# Create directory for downloads

DOWNLOAD_PATH="/Volumes/My Passport for Mac/pred_codiff/temp_wb1/wb_data/"

mkdir -p "$DOWNLOAD_PATH"

# Download temperature data
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/5.625deg/temperature_850&files=temperature_850_5.625deg.zip" -O "$DOWNLOAD_PATH/temperature_850_5.625deg.zip"
# Download baselines
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=weekly_climatology_5.625.nc" -O "$DOWNLOAD_PATH/weekly_climatology_5.625.nc"
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=fccnn_3d.nc" -O "$DOWNLOAD_PATH/fccnn_3d.nc"
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=lr_3d_t_t.nc" -O "$DOWNLOAD_PATH/lr_3d_t_t.nc"
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=persistence_5.625.nc" -O "$DOWNLOAD_PATH/persistence_5.625.nc"
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=t42_5.625deg.nc" -O "$DOWNLOAD_PATH/t42_5.625deg.nc"
wget "https://dataserv.ub.tum.de/s/m1524895/download?path=/baselines&files=t63_5.625deg.nc" -O "$DOWNLOAD_PATH/t63_5.625deg.nc"

echo "All downloads complete!"

cd "$DOWNLOAD_PATH"
unzip temperature_850_5.625deg.zip