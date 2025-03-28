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

# Define the folders containing the files
mppt_folder_path = r'U:/20241105_microMPPT/data/20250206_light/mppt'
temperature_folder_path = r'U:/20241105_microMPPT/data/20250206_light/temperature'

# Initialize storage for MPPT and temperature data
mppt_data_dict = {}
temperature_data = None  # Will store the combined temperature dataframe

# Function to load and preprocess MPPT data
def load_mppt_data(folder_path, required_columns):
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            channel = filename.split('_')[7].split('.')[0]  # Extract channel
            
            # Load the data
            df = pd.read_csv(filepath, header=0, parse_dates=['time'])
            
            # Strip leading and trailing spaces from column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Debug: Print cleaned column names
            print(f"Cleaned columns for {filename}: {df.columns}")
            
            # Check required columns
            if not all(col in df for col in required_columns):
                print(f"Missing required columns in {filename}")
                continue
            
            # Parse 'time' column as datetime
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            
            # Drop rows where 'time' is NaT
            df = df.dropna(subset=['time'])
            
            # Convert numeric columns to numeric, force errors to NaN
            for col in required_columns:
                if col != 'time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows where any required column is NaN
            df = df.dropna(subset=required_columns)
            
            data_dict[channel] = df
    return data_dict

# Load MPPT data (voltage, current)
mppt_data_dict = load_mppt_data(mppt_folder_path, ['time', 'voltage', 'current'])

# Add computed power to MPPT data
for channel, df in mppt_data_dict.items():
    df['power'] = df['voltage'] * df['current']

# Load temperature data
for filename in os.listdir(temperature_folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(temperature_folder_path, filename)
        
        # Load the temperature data
        temperature_data = pd.read_csv(filepath, header=0, parse_dates=['time'])
        
        # Strip leading and trailing spaces from column names
        temperature_data.columns = [col.strip() for col in temperature_data.columns]
        
        # Parse 'time' column as datetime
        temperature_data['time'] = pd.to_datetime(temperature_data['time'], errors='coerce')
        
        # Drop rows where 'time' is NaT
        temperature_data = temperature_data.dropna(subset=['time'])
        
        # Convert sensor columns to numeric, force errors to NaN
        for col in temperature_data.columns:
            if col != 'time':
                temperature_data[col] = pd.to_numeric(temperature_data[col], errors='coerce')
        
        # Drop rows with NaN in any sensor column
        temperature_data = temperature_data.dropna()
        break  # Assume one file in the temperature folder

# Find the last timestamp of any channel
latest_timestamp = None
for channel, df in mppt_data_dict.items():
    last_timestamp = df['time'].iloc[-1]  # Get the timestamp of the last row in each channel
    if latest_timestamp is None or last_timestamp > latest_timestamp:
        latest_timestamp = last_timestamp

# Format the latest timestamp as 'yyyy_mm_dd_HH_MM_SS'
timestamp_str = latest_timestamp.strftime("%Y_%m_%d_%H_%M_%S")

cmap = plt.get_cmap("tab20")

# Create a list of colors, each assigned to a channel or sensor
colors = [cmap(i) for i in range(max(len(mppt_data_dict), len(temperature_data.columns) - 1))]

# Plot settings
plt.figure(figsize=(15, 20))

# Plot 1: Time vs Voltage
plt.subplot(5, 1, 1)
for i, (channel, df) in enumerate(mppt_data_dict.items()):
    plt.plot(df['time'], df['voltage'], label=channel, color=colors[i])
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.title('Time vs Voltage')
plt.legend()
plt.grid()

# Plot 2: Time vs Current
plt.subplot(5, 1, 2)
for i, (channel, df) in enumerate(mppt_data_dict.items()):
    plt.plot(df['time'], df['current'], label=channel, color=colors[i])
plt.xlabel('Time')
plt.ylabel('Current (A)')
plt.title('Time vs Current')
plt.legend()
plt.grid()

# Plot 3: Time vs Power
plt.subplot(5, 1, 3)
for i, (channel, df) in enumerate(mppt_data_dict.items()):
    plt.plot(df['time'], df['power'], label=channel, color=colors[i])
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.title('Time vs Power')
plt.legend()
plt.grid()

# Plot 3: Time vs PCE
plt.subplot(5, 1, 4)
for i, (channel, df) in enumerate(mppt_data_dict.items()):
    plt.plot(df['time'], df['pce'], label=channel, color=colors[i])
plt.xlabel('Time')
plt.ylabel('PCE (%)')
plt.title('Time vs PCE')
plt.legend()
plt.grid()

# Plot 5: Time vs Temperature
plt.subplot(5, 1, 5)
for i, sensor in enumerate(temperature_data.columns[1:]):  # Skip 'time'
    plt.plot(temperature_data['time'], temperature_data[sensor], label=sensor, color=colors[i])
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Time vs Temperature')
plt.legend()
plt.grid()

# Adjust layout and display the plots
plt.tight_layout()

# Construct the filename with the latest timestamp
filename = os.path.join(mppt_folder_path, f"{timestamp_str}_mppt_temperature.png")

# Save the figure
plt.savefig(filename, dpi=300)

# Optionally, print the filename to confirm
print(f"Figure saved as {filename}")

# Show the plot
plt.show()

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime
# import matplotlib.cm as cm

# # Define the folder containing the files
# folder_path = r'U:/20241105_microMPPT/data/20250122_D/mppt'

# # Initialize storage for data and filenames
# data_files = []
# data_dict = {}

# # Iterate through all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):# and filename.startswith('mppt_'):
#         filepath = os.path.join(folder_path, filename)
#         channel = filename.split('_')[7].split('.')[0]  # Extract channel (e.g., 'ch4')
        
#         # Load the data
#         df = pd.read_csv(filepath, header=0, parse_dates=['time'])
        
#         # Strip leading and trailing spaces from column names
#         df.columns = [col.strip().lower() for col in df.columns]
        
#         # Debug: Print cleaned column names
#         print(f"Cleaned columns for {filename}: {df.columns}")
        
#         # Check required columns
#         if 'voltage' not in df or 'current' not in df:
#             print(f"Missing required columns in {filename}")
#             continue
        
#         # Parse 'time' column as datetime
#         df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
#         # Drop rows where 'time' is NaT
#         df = df.dropna(subset=['time'])
        
#         # Convert voltage and current to numeric, force errors to NaN
#         df['voltage'] = pd.to_numeric(df['voltage'], errors='coerce')
#         df['current'] = pd.to_numeric(df['current'], errors='coerce')
        
#         # Drop rows where either voltage or current is NaN
#         df = df.dropna(subset=['voltage', 'current'])
        
#         # Compute power
#         df['power'] = df['voltage'] * df['current']
        
#         data_dict[channel] = df
#         data_files.append(channel)

# # Sort channels for consistent plotting
# data_files.sort()

# # Find the last timestamp of any channel
# latest_timestamp = None
# for channel, df in data_dict.items():
#     last_timestamp = df['time'].iloc[-1]  # Get the timestamp of the last row in each channel
#     if latest_timestamp is None or last_timestamp > latest_timestamp:
#         latest_timestamp = last_timestamp

# # Format the latest timestamp as 'yyyy_mm_dd_HH_MM_SS'
# timestamp_str = latest_timestamp.strftime("%Y_%m_%d_%H_%M_%S")

# cmap = plt.get_cmap("tab20")

# # Create a list of colors, each assigned to a channel
# colors = [cmap(i) for i in range(24)]

# # Plot settings
# plt.figure(figsize=(15, 10))

# # Plot 1: Time vs Voltage
# plt.subplot(3, 1, 1)
# for i, (channel, df) in enumerate(data_dict.items()):
#     plt.plot(df['time'], df['voltage'], label=channel, color=colors[i])
# plt.xlabel('Time')
# plt.ylabel('Voltage (V)')
# plt.title('Time vs Voltage')
# plt.legend()
# plt.grid()

# # Plot 2: Time vs Current
# plt.subplot(3, 1, 2)
# for i, (channel, df) in enumerate(data_dict.items()):
#     plt.plot(df['time'], df['current'], label=channel, color=colors[i])
# plt.xlabel('Time')
# plt.ylabel('Current (A)')
# plt.title('Time vs Current')
# plt.legend()
# plt.grid()

# # Plot 3: Time vs Power
# plt.subplot(3, 1, 3)
# for i, (channel, df) in enumerate(data_dict.items()):
#     plt.plot(df['time'], df['power'], label=channel, color=colors[i])
# plt.xlabel('Time')
# plt.ylabel('Power (W)')
# plt.title('Time vs Power')
# plt.legend()
# plt.grid()

# # Adjust layout and display the plots
# plt.tight_layout()

# # Construct the filename with the latest timestamp
# filename = os.path.join(folder_path, f"{timestamp_str}_mppt.png")

# # Save the figure
# plt.savefig(filename, dpi=300)

# # Optionally, print the filename to confirm
# print(f"Figure saved as {filename}")

# # Show the plot
# plt.show()
