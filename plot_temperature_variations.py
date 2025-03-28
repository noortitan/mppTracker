# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:42:33 2025

@author: Titan.Hartono
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "U:/20241105_microMPPT/data/20250206_light/"
temperature_folder_path = os.path.join(BASE_DIR, "temperature")

# Load temperature data
temperature_data = None  # Initialize variable

for filename in os.listdir(temperature_folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(temperature_folder_path, filename)
        
        temp_df = pd.read_csv(filepath, header=0, parse_dates=['time'])
        temp_df.columns = [col.strip() for col in temp_df.columns]  # Clean column names
        temp_df['time'] = pd.to_datetime(temp_df['time'], errors='coerce')  # Convert time
        temp_df = temp_df.dropna(subset=['time'])  # Remove rows with NaT in time

        # Convert all temperature columns to numeric
        for col in temp_df.columns:
            if col != 'time':
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

        # Drop rows with any NaN values
        temp_df = temp_df.dropna()

        # Concatenate multiple temperature files (if needed)
        if temperature_data is None:
            temperature_data = temp_df
        else:
            temperature_data = pd.concat([temperature_data, temp_df])

# Load Excel information data
cells_info = pd.read_excel(os.path.join(BASE_DIR, '20250206_run_info.xlsx'))

cells_info["Cell name"] = cells_info["Cell name"].fillna("no_cell")

# Get all unique cell names
unique_cell_names = cells_info["Cell name"].unique()
board_names = cells_info["Board"].unique()

# Calculate global min and max temperature across all cells
global_temp_min = temperature_data.iloc[:, 1:].min().min()  # Minimum of all temperature columns
global_temp_max = temperature_data.iloc[:, 1:].max().max()  # Maximum of all temperature columns

#%% Plot based on the unique cell names
# Loop through each unique cell name and create a plot
for cell_name in unique_cell_names:
    # Get channels corresponding to the current cell name
    cell_channels = cells_info.loc[cells_info["Cell name"] == cell_name, "Channel"].astype(str)
    cell_columns = ["time"] + [f"CH{ch}" for ch in cell_channels if f"CH{ch}" in temperature_data.columns]

    # Extract data for the current cell
    cell_data = temperature_data[cell_columns]

    # Create a plot for the current cell
    plt.figure(figsize=(8,4))
    for ch in cell_columns[1:]:  # Exclude 'time' column
        plt.plot(cell_data["time"], cell_data[ch], label=ch)

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Temperature vs. Time for {cell_name}")
    plt.legend(loc="best", ncol=2)  # Two-column legend
    
    # Set the y-axis limits based on global min and max temperature
    plt.ylim(global_temp_min-1, global_temp_max+1)
    
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

#%% Plot based on the location of the boards

# Loop through each unique board and create a plot
for board in board_names:
    # Get channels corresponding to the current cell name
    cell_board = cells_info.loc[cells_info["Board"] == board, "Channel"].astype(str)
    cell_columns = ["time"] + [f"CH{ch}" for ch in cell_board if f"CH{ch}" in temperature_data.columns]

    # Extract data for the current cell
    cell_data = temperature_data[cell_columns]

    # Create a plot for the current cell
    plt.figure(figsize=(8,4))
    for ch in cell_columns[1:]:  # Exclude 'time' column
        plt.plot(cell_data["time"], cell_data[ch], label=ch)

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Temperature vs. Time for board {board}")
    plt.legend(loc="best", ncol=2)  # Two-column legend
    
    # Set the y-axis limits based on global min and max temperature
    plt.ylim(global_temp_min-1, global_temp_max+1)
    
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

#%% Plot IV progression over time

channel_of_interest = [3, 8, 16]

directory_iv = os.path.join(BASE_DIR, "jv")


