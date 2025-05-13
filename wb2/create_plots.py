import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

from weatherbench2.visualization import set_wb2_style
from weatherbench2.metrics import _spatial_average

# Apply the WeatherBench2 plotting style
set_wb2_style()

# Ensure LaTeX‐style fonts for all text in the plots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Define variables, variable abbreviations and units
variables = ['mean_sea_level_pressure', '2m_temperature', '10m_wind_speed', 'total_precipitation_24hr']
titles = ['MSLP', 'T2M', 'WS10', 'TP24hr']
units = [r'$\left[ \text{Pa} \right]$', r'$ \left[ \text{K} \right]$', r'$ \left[ \text{m}/\text{s}^{-1} \right] $', r'$\left[ \text{m}/\text{d}^{-1} \right]$']

# Lead times as numpy timedeltas for slicing
lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
]
# Convert timedeltas into integer days for plotting
lead_time_days = lead_times/np.timedelta64(1, 'D')

# Load in PC^0
era5_pc0 = xr.open_dataset('results/era5_pc0.nc', decode_timedelta=True).load()
era5_pc0_spatial_avg = era5_pc0.map(_spatial_average, region=None, skipna=False)
era5_pc0_mean = era5_pc0.mean(dim=['latitude', 'longitude'])

ifs_analysis_pc0 = xr.open_dataset('results/ifs_analysis_pc0.nc', decode_timedelta=True).load()
ifs_analysis_pc0_spatial_avg = ifs_analysis_pc0.map(_spatial_average, region=None, skipna=False)
ifs_analysis_pc0_mean = ifs_analysis_pc0.mean(dim=['latitude', 'longitude'])

# Load in PC and PCS results
graphcast_vs_era5_results = xr.open_dataset('results/graphcast_vs_era5_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()
pangu_vs_era5_results = xr.open_dataset('results/pangu_vs_era5_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()
ifs_hres_vs_era5_results = xr.open_dataset('results/ifs_hres_vs_era5_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()

graphcast_vs_ifs_analysis_results = xr.open_dataset('results/graphcast_vs_ifs_analysis_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()
pangu_vs_ifs_analysis_results = xr.open_dataset('results/pangu_vs_ifs_analysis_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()
ifs_hres_vs_ifs_analysis_results = xr.open_dataset('results/ifs_hres_vs_ifs_analysis_pc.nc', decode_timedelta=True).sel(prediction_timedelta=lead_times, metric=['pc', 'pcs']).load()

# Load in IFS ENS CRPS
ifs_ens_vs_era5 = xr.open_dataset('results/ifs_ens_vs_era5_crps.nc', decode_timedelta=True).sel(lead_time=lead_times).load()
ens_crps = ifs_ens_vs_era5.rename({'lead_time':'prediction_timedelta'}).transpose(*ifs_hres_vs_era5_results.sel(metric="pc").dims)

# Load in p-values
graphcast_pangu_vs_era5_p = xr.open_dataset('results/graphcast_pangu_vs_era5_p.nc', decode_timedelta=True).load()
graphcast_pangu_vs_ifs_analysis_p = xr.open_dataset('results/graphcast_pangu_vs_ifs_analysis_p.nc', decode_timedelta=True).load()
graphcast_ifs_hres_vs_era5_p = xr.open_dataset('results/graphcast_ifs_hres_vs_era5_p.nc', decode_timedelta=True).load()
graphcast_ifs_hres_vs_ifs_analysis_p = xr.open_dataset('results/graphcast_ifs_hres_vs_ifs_analysis_p.nc', decode_timedelta=True).load()
pangu_hres_vs_era5_p = xr.open_dataset('results/pangu_hres_vs_era5_p.nc', decode_timedelta=True).load()
pangu_hres_vs_ifs_analysis_p = xr.open_dataset('results/pangu_hres_vs_ifs_analysis_p.nc', decode_timedelta=True).load()

# Compute latitude weighted mean
graphcast_vs_era5_results_spatial_avg = graphcast_vs_era5_results.map(_spatial_average, region=None, skipna=False)
pangu_vs_era5_results_spatial_avg = pangu_vs_era5_results.map(_spatial_average, region=None, skipna=False)
ifs_hres_vs_era5_results_spatial_avg = ifs_hres_vs_era5_results.map(_spatial_average, region=None, skipna=False)

graphcast_vs_ifs_analysis_results_spatial_avg = graphcast_vs_ifs_analysis_results.map(_spatial_average, region=None, skipna=False)
pangu_vs_ifs_analysis_results_spatial_avg = pangu_vs_ifs_analysis_results.map(_spatial_average, region=None, skipna=False)
ifs_hres_vs_ifs_analysis_results_spatial_avg = ifs_hres_vs_ifs_analysis_results.map(_spatial_average, region=None, skipna=False)

# Compute mean
graphcast_vs_era5_results_mean = graphcast_vs_era5_results.mean(dim=['latitude', 'longitude'])
pangu_vs_era5_results_mean = pangu_vs_era5_results.mean(dim=['latitude', 'longitude'])
ifs_hres_vs_era5_results_mean = ifs_hres_vs_era5_results.mean(dim=['latitude', 'longitude'])

graphcast_vs_ifs_analysis_results_mean = graphcast_vs_ifs_analysis_results.mean(dim=['latitude', 'longitude'])
pangu_vs_ifs_analysis_results_mean = pangu_vs_ifs_analysis_results.mean(dim=['latitude', 'longitude'])
ifs_hres_vs_ifs_analysis_results_mean = ifs_hres_vs_ifs_analysis_results.mean(dim=['latitude', 'longitude'])

# Compute Pearson's r and Spearman's ρ for for each variable overall as well as stratified by lead time.
corr_df = pd.DataFrame(index=variables, columns=['pearson', 'spearman'])

for i, var in enumerate(variables):    
    x = ens_crps[var].values.flatten()
    y = ifs_hres_vs_era5_results.sel(metric="pc")[var].values.flatten()
    pearson_val, _ = pearsonr(x, y)
    spearman_val, _ = spearmanr(x, y)
    corr_df.loc[var, 'pearson'] = pearson_val
    corr_df.loc[var, 'spearman'] = spearman_val
corr_df.to_csv(f'results/corr_df.csv')

pearson_df_lt = pd.DataFrame(index=variables, columns=[int(lt) for lt in lead_time_days])
spearman_df_lt = pd.DataFrame(index=variables, columns=[int(lt) for lt in lead_time_days])
for i, var in enumerate(variables):    
    for j, lt in enumerate(lead_times):
        x = ens_crps.sel(prediction_timedelta=lt)[var].values.flatten()
        y = ifs_hres_vs_era5_results.sel(metric="pc").sel(prediction_timedelta=lt)[var].values.flatten()
        pearson_val, _ = pearsonr(x, y)
        spearman_val, _ = spearmanr(x, y)
        pearson_df_lt.iloc[i,j] = pearson_val
        spearman_df_lt.iloc[i,j] = spearman_val
pearson_df_lt.to_csv(f'results/pearson_df_lt.csv')
spearman_df_lt.to_csv(f'results/spearman_df_lt.csv')

# -------- Line plot for PC of WeatherBench 2 models with ERA5 (top) and IFS analysis (bottom) as ground truth, latitude weighted mean ---------- #
# Create the figure and subplots
fig, axes = plt.subplots(2, len(variables[:-1]), figsize=(16, 10))

# ----------------------- Ground Truth Era5 -----------------------
for ax, var, title, unit in zip(axes[0, :], variables[:-1], titles[:-1], units[:-1]):
    ax.plot(lead_time_days,
            era5_pc0_spatial_avg.sel(prediction_timedelta=lead_times)[var],
            label='ERA5 $\\text{PC}^{(0)}$', color='tab:red')
    ax.plot(lead_time_days,
            ifs_hres_vs_era5_results_spatial_avg.sel(metric='pc')[var], color='tab:blue')
    ax.plot(lead_time_days,
            pangu_vs_era5_results_spatial_avg.sel(metric='pc')[var], color='tab:orange')
    ax.plot(lead_time_days,
            graphcast_vs_era5_results_spatial_avg.sel(metric='pc', prediction_timedelta=lead_times)[var], color='tab:green')

    ax.set_xlabel('Lead Time [d]', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticks(lead_time_days)
    ax.set_ylim(0)
    ax.set_ylabel('Latitude Weighted Mean PC ' + unit, fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)

# ----------------------- Ground Truth IFS Analysis -----------------------
for ax, var, title, unit in zip(axes[1, :], variables[:-1], titles[:-1], units[:-1]):
    ax.plot(lead_time_days,
            ifs_analysis_pc0_spatial_avg.sel(prediction_timedelta=lead_times)[var],
            label='IFS Analysis $\\text{PC}^{(0)}$', color='tab:purple')
    ax.plot(lead_time_days,
            ifs_hres_vs_ifs_analysis_results_spatial_avg.sel(metric='pc')[var],
            label='HRES', color='tab:blue')
    ax.plot(lead_time_days,
            pangu_vs_ifs_analysis_results_spatial_avg.sel(metric='pc')[var],
            label='Pangu-Weather', color='tab:orange')
    ax.plot(lead_time_days,
            graphcast_vs_ifs_analysis_results_spatial_avg.sel(metric='pc', prediction_timedelta=lead_times)[var],
            label='GraphCast', color='tab:green')

    ax.set_xlabel('Lead Time [d]', fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.set_xticks(lead_time_days)
    ax.set_ylim(0)
    ax.set_ylabel('Latitude Weighted Mean PC ' + unit, fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

# ----------------------- Row Titles -----------------------
# Upper row:
row1_left = axes[0, 0].get_position().x0
row1_right = axes[0, -1].get_position().x1
row1_top = axes[0, 0].get_position().y1
row1_center_x = (row1_left + row1_right) / 2

# Lower row:
row2_left = axes[1, 0].get_position().x0
row2_right = axes[1, -1].get_position().x1
row2_top = axes[1, 0].get_position().y1
row2_center_x = (row2_left + row2_right) / 2

# Place the row titles at some vertical offset above each row
fig.text(row1_center_x, row1_top + 0.04, "Ground Truth ERA5", 
         ha='center', va='bottom', fontsize=22)
fig.text(row2_center_x, row2_top + 0.04, "Ground Truth IFS Analysis", 
         ha='center', va='bottom', fontsize=22)

# ----------------------- Combined Legend --------------------------
handles_upper, labels_upper = axes[0, 0].get_legend_handles_labels()
handles_lower, labels_lower = axes[1, 0].get_legend_handles_labels()

# Combine them into a dictionary to eliminate duplicates
combined_legend = {}
for h, l in zip(handles_upper, labels_upper):
    combined_legend[l] = h
for h, l in zip(handles_lower, labels_lower):
    combined_legend[l] = h

combined_handles = list(combined_legend.values())
combined_labels = list(combined_legend.keys())

fig.legend(combined_handles, combined_labels, loc='lower center', ncol=5, 
           bbox_to_anchor=(0.5, -0.075), fontsize=22)

plt.savefig('plots/lineplot_lat_mean_pc_without_precip.png', facecolor='white', edgecolor='none', bbox_inches='tight')

# ----------- Map plot for PCS of GraphCast with ERA5 as ground truth. ------------------ #
# Create a meshgrid for plotting
lon = era5_pc0['longitude'].values
lat = era5_pc0['latitude'].values
lon_grid, lat_grid = np.meshgrid(lon, lat)

n_rows = len(lead_times)
n_cols = len(variables[:-1])

# Create a grid of subplots with one row per lead time and one column per variable
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), subplot_kw={'projection': ccrs.Robinson()}, constrained_layout=True)

# Normalization
norm = mcolors.Normalize(vmin=0, vmax=1)

for i, lt in enumerate(lead_times):
    # Convert lead time to days for the row label
    lt_days = int(lt / np.timedelta64(1, 'D'))
    
    for j, (var, var_title) in enumerate(zip(variables[:-1], titles[:-1])):
        ax = axes[i, j]
        ax.coastlines()
        
        # Select the data for the variable at the given lead time
        data = graphcast_vs_era5_results.sel(metric='pcs', prediction_timedelta=lt)[var]
        
        # Plot the data
        mesh = ax.pcolormesh(lon_grid, lat_grid, data.T, cmap='viridis', norm=norm, transform=ccrs.PlateCarree(), linewidth=0.5)
        
        # Every subplot gets its variable title on top
        ax.set_title(var_title, fontsize=24)
        
        # If this is the bottom row, add an x-axis label and the colorbar
        if i == n_rows - 1:
            ax.set_xlabel('Longitude')
            cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.75, pad=0.1)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label('$\\text{PCS}^{(\\text{GraphCast})}$', fontsize=22)
    
    # Annotate the left-most subplot of the current row with the lead time information.
    axes[i, 0].annotate(
        f"Lead Time: {lt_days} day" if i==0 else f"Lead Time: {lt_days} days",
        fontsize=22,
        xy=(-0.1, 0.5), 
        xycoords='axes fraction',
        rotation=90, 
        va='center'
    )

plt.savefig('plots/maps_graphcast_240x121_vs_era5_pcs_without_precip.png', facecolor='white', edgecolor='none', bbox_inches='tight')

# --------- Scatterplot for in-sample PC of IFS HRES vs out-of-sample CRPS of IFS ENS with ERA5 as ground truth. ---------- #
xlims = [(0, 830), (0, 4.3), (0, 3.2)] 
ylims = [(0, 830), (0, 4.3), (0, 3.2)]  

fig, axes = plt.subplots(1, len(variables[:-1]), figsize=(24, 8))

for ax, var, title, xlim_val, ylim_val, unit in zip(axes, variables[:-1], titles[:-1], xlims, ylims, units[:-1]):
    
    # Stack (prediction_timedelta, latitude, longitude) into one dimension "points"
    ens_data_stacked = ifs_ens_vs_era5[var].rename({'lead_time':'prediction_timedelta'}).stack(points=('prediction_timedelta', 'latitude', 'longitude'))
    hres_data_stacked = ifs_hres_vs_era5_results.sel(metric='pc')[var].stack(points=('prediction_timedelta', 'latitude', 'longitude'))
    
    # Align the two DataArrays so that they share matching coordinates
    ens_aligned, hres_aligned = xr.align(ens_data_stacked, hres_data_stacked, join="inner")
    
    # Convert to NumPy arrays
    x = hres_aligned.values
    y = ens_aligned.values
    
    # Extract lead times (in days) from the 'prediction_timedelta' coordinate
    lt_days = ens_aligned['prediction_timedelta'].dt.days.values
    
    # Use discrete, standard colors (one per unique lead time)
    unique_leads = np.unique(lt_days)
    standard_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    lead_to_color = dict(zip(unique_leads, standard_colors))
    
    # Map the lead time values to the corresponding colors
    point_colors = np.array([lead_to_color[lt] for lt in lt_days])
    
    # Create the scatter plot
    ax.scatter(x[::-1], y[::-1], c=point_colors[::-1], s=2, alpha=0.7)
    
    # Add a reference line from the origin to the top-right corner of the limits
    ax.plot([xlim_val[0], xlim_val[1]], [ylim_val[0], ylim_val[1]], color='k', linestyle='--', linewidth=1.5)
    
    ax.set_ylabel("ENS $\overline{\\text{CRPS}}$ " + unit, fontsize=20)
    ax.set_xlabel("HRES PC " + unit, fontsize=20)
    ax.set_title(title, fontsize=28)
    ax.set_xlim(xlim_val)
    ax.set_ylim(ylim_val)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Build legend for Pearson's r in UL corner
    pearson_handles = []
    pearson_labels  = []
    for j, lt in enumerate(unique_leads):
        pearson_handles.append(Line2D([], [], color=lead_to_color[lt], marker='o', linestyle='None', markersize=6))
        pearson_labels.append(f"{pearson_df_lt.loc[var,lt]:.3f}")

    pearson_full = corr_df.loc[var, 'pearson']
    leg1 = ax.legend(
        pearson_handles, pearson_labels,
        title=f"$r$ = {pearson_full:.3f}",
        loc="upper left",
        fontsize=20, title_fontsize=20,
        framealpha=0.5,
        borderpad=0.2,
        borderaxespad=0.2,
        handletextpad=0.4
    )
    # Box the overall r
    title1 = leg1.get_title()
    title1.set_bbox(
        dict(
            mutation_aspect=0.33,
            fill=False,
            edgecolor='black',
            boxstyle='round,pad=0.55,rounding_size=0.3'
        )
    )
    ax.add_artist(leg1)

    # Build legend for Spearman's ρ in LR corner
    spearman_handles = []
    spearman_labels  = []
    for j, lt in enumerate(unique_leads):
        spearman_handles.append(Line2D([], [], color=lead_to_color[lt], marker='o', linestyle='None', markersize=6))
        spearman_labels.append(f"{spearman_df_lt.loc[var,lt]:.3f}")
        
    spearman_full = corr_df.loc[var, 'spearman']
    leg2 = ax.legend(
        spearman_handles, spearman_labels,
        title=f"$\\rho$ = {spearman_full:.3f}",
        loc="lower right",
        fontsize=20, title_fontsize=20,
        framealpha=0.5,
        borderpad=0.2,
        borderaxespad=0.2,
        handletextpad=0.4
    )
    # Box the overall ρ
    title2 = leg2.get_title()
    title2.set_bbox(
        dict(
            mutation_aspect=0.37,
            fill=False,
            edgecolor='black',
            boxstyle='round,pad=0.45,rounding_size=0.3'
        )
    )

# Create combined legend
legend_handles = []
for lt in unique_leads:
    color = lead_to_color[lt]
    handle = Line2D([], [], color=color, marker='o', linestyle='None', markersize=6, label=int(lt))
    legend_handles.append(handle)

# Add the legend at the bottom center of the figure.
fig.legend(
    legend_handles, 
    [handle.get_label() for handle in legend_handles],
    loc='lower center', 
    ncol=len(unique_leads), 
    bbox_to_anchor=(0.5175, 0.0), 
    fontsize=18, 
    title='Lead Time [d]', 
    title_fontsize=20, 
    prop={'size': 20}, 
    handletextpad=0.5
)

plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('plots/scatterplot_ens_vs_hres.png', facecolor='white', edgecolor='none', bbox_inches='tight')

# ----------------------------- Boxplots of p-values with ERA5 as ground truth ----------------------------- #
# Create a figure with one subplot per variable
fig, axes = plt.subplots(1, len(variables[:-1]), figsize=(16, 5))

for ax, var, title in zip(axes, variables[:-1], titles[:-1]):
    # Lists for boxplot positions and data per model
    positions_graphcast_pangu = []
    positions_graphcast_hres = []
    positions_pangu_hres = []
    data_graphcast_pangu = []
    data_graphcast_hres = []
    data_pangu_hres = []
    x_ticks = []
    offset = 0.3  

    # Loop over each lead time
    for lt in lead_times:
        lt_days = int(lt / np.timedelta64(1, 'D'))
        x_ticks.append(lt_days)
        
        # Get p-values
        d1 = graphcast_pangu_vs_era5_p.sel(prediction_timedelta=lt)[var].values.flatten()
        d2 = graphcast_ifs_hres_vs_era5_p.sel(prediction_timedelta=lt)[var].values.flatten()
        d3 = pangu_hres_vs_era5_p.sel(prediction_timedelta=lt)[var].values.flatten()
        
        data_graphcast_pangu.append(d1)
        data_graphcast_hres.append(d2)
        data_pangu_hres.append(d3)
        
        # Define positions for each model's boxplot
        positions_graphcast_pangu.append(lt_days - offset)
        positions_graphcast_hres.append(lt_days)
        positions_pangu_hres.append(lt_days + offset)

    boxplot_kwargs = dict(
        widths=0.2,
        patch_artist=True,
        medianprops=dict(color='black'),
        showfliers=False  # hide outliers
    )
    
    # Plot the boxplots for each model
    ax.boxplot(data_graphcast_pangu, positions=positions_graphcast_pangu, boxprops=dict(facecolor='tab:green'), **boxplot_kwargs)
    ax.boxplot(data_graphcast_hres, positions=positions_graphcast_hres, boxprops=dict(facecolor='tab:blue'), **boxplot_kwargs)
    ax.boxplot(data_pangu_hres, positions=positions_pangu_hres, boxprops=dict(facecolor='tab:orange'), **boxplot_kwargs)
    
    ax.set_xlabel('Lead Time [days]', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_ylabel('p-values', fontsize=16)

# Shared legend
legend_handles = [
    Patch(facecolor='tab:green', label='$H_0: \\text{PC}^{(\\text{GraphCast})} \geq \\text{PC}^{(\\text{Pangu})}$'),
    Patch(facecolor='tab:blue', label='$H_0: \\text{PC}^{(\\text{GraphCast})} \geq \\text{PC}^{(\\text{HRES})}$'),
    Patch(facecolor='tab:orange', label='$H_0: \\text{PC}^{(\\text{Pangu})} \geq \\text{PC}^{(\\text{HRES})}$')
]
# Add shared legend to the full figure
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=16)

# Add main title
fig.suptitle("Ground Truth: ERA5", fontsize=20)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.savefig('plots/boxplot_pvalues_era5.png', facecolor='white', edgecolor='none', bbox_inches='tight')

# ----------------------------- Boxplots of p-values with IFS analysis as ground truth ----------------------------- #
# Create a figure with one subplot per variable (sharing y-axis for consistency)
fig, axes = plt.subplots(1, len(variables[:-1]), figsize=(16, 5))

for ax, var, title in zip(axes, variables[:-1], titles[:-1]):
    # Lists for boxplot positions and data per model
    positions_graphcast_pangu = []
    positions_graphcast_hres = []
    positions_pangu_hres = []
    data_graphcast_pangu = []
    data_graphcast_hres = []
    data_pangu_hres = []
    x_ticks = []
    offset = 0.3  

    # Loop over each lead time
    for lt in lead_times:
        lt_days = int(lt / np.timedelta64(1, 'D'))
        x_ticks.append(lt_days)
        
        # Get p-values
        d1 = graphcast_pangu_vs_ifs_analysis_p.sel(prediction_timedelta=lt)[var].values.flatten()
        d2 = graphcast_ifs_hres_vs_ifs_analysis_p.sel(prediction_timedelta=lt)[var].values.flatten()
        d3 = pangu_hres_vs_ifs_analysis_p.sel(prediction_timedelta=lt)[var].values.flatten()
        
        data_graphcast_pangu.append(d1)
        data_graphcast_hres.append(d2)
        data_pangu_hres.append(d3)
        
        # Define positions for each model's boxplot
        positions_graphcast_pangu.append(lt_days - offset)
        positions_graphcast_hres.append(lt_days)
        positions_pangu_hres.append(lt_days + offset)

    boxplot_kwargs = dict(
        widths=0.2,
        patch_artist=True,
        medianprops=dict(color='black'),
        showfliers=False  # hide outliers
    )
    
    # Plot the boxplots for each model
    ax.boxplot(data_graphcast_pangu, positions=positions_graphcast_pangu, boxprops=dict(facecolor='tab:green'), **boxplot_kwargs)
    ax.boxplot(data_graphcast_hres, positions=positions_graphcast_hres, boxprops=dict(facecolor='tab:blue'), **boxplot_kwargs)
    ax.boxplot(data_pangu_hres, positions=positions_pangu_hres, boxprops=dict(facecolor='tab:orange'), **boxplot_kwargs)
    
    ax.set_xlabel('Lead Time [days]', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_ylabel('p-values', fontsize=16)

# Shared legend
legend_handles = [
    Patch(facecolor='tab:green', label='$H_0: \\text{PC}^{(\\text{GraphCast})} \geq \\text{PC}^{(\\text{Pangu})}$'),
    Patch(facecolor='tab:blue', label='$H_0: \\text{PC}^{(\\text{GraphCast})} \geq \\text{PC}^{(\\text{HRES})}$'),
    Patch(facecolor='tab:orange', label='$H_0: \\text{PC}^{(\\text{Pangu})} \geq \\text{PC}^{(\\text{HRES})}$')
]
# Add shared legend to the full figure
fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=16)

# Add main title
fig.suptitle("Ground Truth: IFS Analysis", fontsize=20)

# Adjust layout to make room for legend
plt.tight_layout(rect=[0, 0.07, 1, 1])

plt.savefig('plots/boxplot_pvalues_ifs_analysis.png', facecolor='white', edgecolor='none', bbox_inches='tight')