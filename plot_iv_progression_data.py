# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:32:50 2025

@author: Titan.Hartono
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

# Define the folder containing the files
jv_folder_path = r'U:/20241105_microMPPT/data/20250206_light/jv'

# Channels of interest
ch_interest = ['CH3', 'CH8', 'CH16']

# Constants
irradiance = 1000  # W/m^2
area = 1e-4  # 1 cm^2 in m^2

# Create the directory for saving plots
plot_dir = os.path.join(jv_folder_path, 'plots', 'overview')
os.makedirs(plot_dir, exist_ok=True)

# Function to parse filename and extract metadata
def parse_filename(filename):
    parts = filename.split('_')
    channel = parts[1]  # e.g., CH3
    timestamp_str = '_'.join(parts[2:8])  # YYYY_MM_DD_HH_MM_SS
    scan_direction = parts[8].split('.')[0]  # fwd or bwd
    timestamp = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
    return channel, timestamp, scan_direction

# Read all valid files
files = [f for f in os.listdir(jv_folder_path) if f.startswith('jv_') and f.endswith('.txt')]

# Create a dictionary to store data for each channel and scan direction
data_dict = {}

for file in sorted(files):
    channel, timestamp, scan_direction = parse_filename(file)
    file_path = os.path.join(jv_folder_path, file)

    # Read data correctly
    df = pd.read_csv(file_path, sep=',', skiprows=0)  

    # Standardize column names (remove spaces, convert to lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Ensure required columns exist
    if 'voltage' not in df.columns or 'current' not in df.columns:
        print(f"Skipping {file}: Missing required columns.")
        continue

    df['Timestamp'] = timestamp

    # Store in dictionary
    key = (channel, scan_direction)
    if key not in data_dict:
        data_dict[key] = []
    data_dict[key].append(df)
    
# Store efficiency, Voc, Isc, and FF results
results = {key: [] for key in data_dict}

for (channel, scan_direction), data_list in data_dict.items():
    for df in data_list:
        voltage = df['voltage'] # in volt
        current = df['current'] # in ampere
        power = -1*voltage * current # watt
        power_density = voltage*current/area # watt/m2
        
        voc = voltage[current <= 0].max() # because iv is from neg current to pos current
        # isc = -1*(current[voltage <= 0].min()) # A/m2 because iv is from neg current to pos current
        isc = -1*current[voltage ==0].iloc[0]
        jsc = isc/(10*area) # in mA/cm2
        pmax = power.max()
        
        efficiency = (pmax / (irradiance * area)) * 100
        ff = (pmax*100 / (voc * isc)) if (voc * isc) != 0 else 0
        
        results[(channel, scan_direction)].append({
            'Timestamp': df['Timestamp'].iloc[0],
            'Voc': voc,
            'Isc': isc,
            'Jsc': jsc,
            'FF': ff,
            'Efficiency': efficiency
        })
        if channel in ch_interest:
            print(f"Channel {channel}, voc={voc}, isc={isc}, jsc={jsc}, pmax={pmax}, ff={ff}, pce={efficiency}")

# Plot data
for (channel, scan_direction), data_list in data_dict.items():
    plt.figure(figsize=(4,3), dpi=300)
    cmap = cm.get_cmap('viridis')#, len(data_list))  # Color progression
    
    num_curves = len(data_list)
    selected_indices = [0, num_curves // 2, num_curves - 1]  # First, middle, last
    
    # Determine y-limits based on last 3 rows of all datasets
    all_y_values = np.concatenate([df['current'].iloc[-3:].values for df in data_list])
    y_min, y_max = all_y_values.min(), all_y_values.max()
    
    for idx, df in enumerate(data_list):
        color = cmap(idx / num_curves)  # Normalize colors

        # Only label selected timestamps for better readability
        if idx in selected_indices:
            label = df['Timestamp'].iloc[0].strftime('%H:%M:%S')
        else:
            label = None

        plt.plot(df['voltage'], df['current'], label=label, color=color, alpha=0.45)
    
    # plt.ylim(y_min-0.001, 0.0001)  # Apply dynamic y-limits
    # plt.xlim(-0.05, 1.75)

    plt.xlabel('Voltage')
    plt.ylabel('Current')
    plt.title(f'{channel} ({scan_direction})')
    plt.legend(title="Time", loc="best")  # Only shows 3 timestamps now
    plt.grid()
    # plt.show()
    save_path = os.path.join(plot_dir, f'{channel}_{scan_direction}_IV_curve_overall.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
# Plot efficiency, Voc, Isc, and FF
metrics = ['Voc', 'Jsc', 'FF', 'Efficiency']
metric_labels = {'Voc':'Voc (V)', 'Jsc':'Jsc (mA/cm2)', 'FF': 'FF (%)', 'Efficiency':'PCE (%)'}
colors = sns.color_palette("tab20", n_colors=len(ch_interest) * 2)  

# Create a mapping from channels to paired colors
channel_color_map = {
    channel: {'fwd': colors[i * 2], 'bwd': colors[i * 2 + 1]}  # Dark for fwd, light for bwd
    for i, channel in enumerate(ch_interest)
}

# Iterate over each metric
for metric in metrics:
    plt.figure(figsize=(5, 4), dpi=300)

    # Iterate over channels and scan directions
    for channel in ch_interest:
        for scan_direction in ['fwd', 'bwd']:
            key = (channel, scan_direction)
            if key in results:
                values = results[key]
                timestamps = [entry['Timestamp'] for entry in values]
                metric_values = [entry[metric] for entry in values]

                # Use tab20 paired colors (dark for 'fwd', light for 'bwd')
                plt.plot(
                    timestamps, metric_values,
                    color=channel_color_map[channel][scan_direction],  # Select correct shade
                    linestyle='-',  # Keep solid lines, remove markers
                    label=f"{channel} ({scan_direction})"
                )

    # Plot settings
    plt.xlabel('Time')
    plt.ylabel(metric_labels[metric])
    
    if metric=='FF':
        plt.ylim(-1,100)
    
    plt.title(f'{metric} Over Time for Selected Channels')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    # plt.show()
    save_path = os.path.join(plot_dir, f'_overview_{metric}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()