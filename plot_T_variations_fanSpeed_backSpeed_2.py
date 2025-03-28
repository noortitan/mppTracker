# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:35:05 2025

@author: Titan.Hartono
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the specified file path
file_path = r"G:/aging_setup/temp_fan_test_2/back_fan_saftly_test/20250314_1635/2025_03_14_16_38_09_temperature.txt"
save_dir = r"G:/aging_setup/temp_fan_test_2/back_fan_saftly_test/20250314_1635/2025_03_14_16_38_09/plots/"
excel_file = os.path.join(save_dir, "temperature_summary.xlsx")


# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

data = pd.read_csv(file_path)

data['time'] = pd.to_datetime(data['time'])  # Convert time to datetime format

# Identify speed-related columns
speed_columns = ['front1_speed', 'front2_speed', 'back_speed', 'inside_back_speed', 'inside_front_speed']

# Identify channel columns
channel_columns = [col for col in data.columns if col.startswith("CH")]

# Group by unique combinations of speed settings
grouped = data.groupby(speed_columns)

# List to store summary data
summary_data = []

# Iterate through each group
for speed_values, group in grouped:
    plt.figure(figsize=(8, 5), dpi=300)
    
    for ch in channel_columns:
        plt.plot(group['time'], group[ch], label=ch)
    
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title(f"{dict(zip(speed_columns, speed_values))}")
    plt.legend(ncol=3)
    plt.xticks(rotation=45)
    plt.grid()
    
    # Generate filename based on speed values
    speed_str = "_".join(str(v) for v in speed_values)  # Convert speed values to a string
    filename = f"{speed_str}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, bbox_inches="tight")  # Save the figure
    plt.close()  # Close the figure to free memory

    # Extract last row temperatures for each channel
    last_temperatures = group[channel_columns].iloc[-1]
    
    # Find highest and lowest last temperature for this speed group
    highest_last_temp = last_temperatures.max()
    lowest_last_temp = last_temperatures.min()

    # Append data to summary list
    summary_data.append(list(speed_values) + [highest_last_temp, lowest_last_temp])

# Create DataFrame from summary data
summary_df = pd.DataFrame(summary_data, columns=speed_columns + ["Highest_Last_Temperature", "Lowest_Last_Temperature"])
summary_df['Delta_Last_Temperature'] = summary_df['Highest_Last_Temperature']-summary_df['Lowest_Last_Temperature']

# Save DataFrame to Excel
summary_df.to_excel(excel_file, index=False)

print(f"Excel summary saved at: {excel_file}")