# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:24:14 2025

@author: Titan.Hartono
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the TXT file
file_path = "G:/aging_setup/temp_fan_test/temperature/2025_02_14_14_29_57_temperature.txt"  # Change this to your actual file path

# correct_headers = ["timestamp", "elapsed_time", "fan_back_speed", "fan_front_speed"] + [f"CH{i}" for i in range(1, 25)]
# df = pd.read_csv(file_path, names=correct_headers, skiprows=1, parse_dates=["timestamp"])

# Read the file to get the exact header and use it for columns
df_raw = pd.read_csv(file_path, header=None, nrows=1)  # Only read the first row for the header
columns_from_file = df_raw.iloc[0].tolist()  # Get the first row as a list

# Define new header: insert 'elapsed_time', 'fan_back_speed', 'fan_front_speed' after 'timestamp'
new_header = ["timestamp", "elapsed_time", "fan_back_speed", "fan_front_speed"] + columns_from_file[1:]

# Load the data with the newly created header
df = pd.read_csv(file_path, names=new_header, skiprows=1, parse_dates=["timestamp"])


# Recalculate elapsed_time based on fan speed group
df["elapsed_time"] = df.groupby(["fan_back_speed", "fan_front_speed"])["timestamp"].transform(
    lambda x: ((x - x.min()).dt.total_seconds() / 60)  # Keep in minutes
)

# df = df.sort_index().groupby(["fan_back_speed", "fan_front_speed"]).tail(20)

# Calculate global max and min temperature across all channels (preserving order from the file)
channels = df.columns[4:]  # Starting from CH1 to CH24 based on your original file structure
global_max_temp = df[channels].max().max()
global_min_temp = df[channels].min().min()

# Define board channel groupings
board_channels = {
    'A': ["CH19", "CH20", "CH21", "CH22", "CH23", "CH24"],  # Channels 18-24
    'B': ["CH13", "CH14", "CH15", "CH16", "CH17", "CH18"],           # Channels 13-18
    'C': ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6"],                 # Channels 1-6
    'D': ["CH7", "CH8", "CH9", "CH10", "CH11", "CH12"]               # Channels 7-12
}

# # Do it differently
# board_channels = {
#     'X': ["CH5", "CH6", "CH11", "CH12", "CH17", "CH18", "CH23", "CH24"],  # Channels 18-24
#     'Y': ["CH3", "CH4", "CH9", "CH10", "CH15", "CH16", "CH21", "CH22"],           # Channels 13-18
#     'Z': ["CH1", "CH2", "CH7", "CH8", "CH13", "CH14", "CH19", "CH20"],                 # Channels 1-6
#     # 'D': ["CH7", "CH8", "CH9", "CH10", "CH11", "CH12"]               # Channels 7-12
# }

# Get unique speed combinations
speed_groups = df.groupby(["fan_back_speed", "fan_front_speed"])

#%% Each group separately

# Plot each group separately
for (back_speed, front_speed), group in speed_groups:
    plt.figure(figsize=(6, 3))

    # Plot all CH columns in the original order
    for ch in channels:
        plt.plot(group["elapsed_time"], group[ch], label=ch, alpha=0.7)

    # Set consistent y-limits based on global max/min temperatures
    plt.ylim(global_min_temp - 2, global_max_temp + 2)  # Adjust the range a bit for better visualization
    # plt.ylim(global_min_temp - 2, 68)  # Adjust the range a bit for better visualization

    # Labels and title
    plt.xlabel("Elapsed Time (minutes)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Back={back_speed}, Front={front_speed}")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2)  # Place legend outside
    plt.grid(True)
    
    # Show the plot
    plt.show()
    
#%% Plot each group separately by board
for (back_speed, front_speed), group in speed_groups:
    for board, channels in board_channels.items():
        plt.figure(figsize=(5,3))

        # Plot all channels for the specific board
        for ch in channels:
            if ch in group.columns:  # Ensure the channel is present in the data
                plt.plot(group["elapsed_time"], group[ch], label=ch, alpha=0.7)

        # Set consistent y-limits based on global max/min temperatures
        plt.ylim(global_min_temp - 2, global_max_temp + 2)  # Adjust the range a bit for better visualization

        # Labels and title
        plt.xlabel("Elapsed Time (minutes)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Back={back_speed}, Front={front_speed}, Board={board}")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Place legend outside
        plt.grid(True)

        # Show the plot
        plt.show()
        
#%% calculate the delta

# Initialize an empty list to store the results
results = []

# Get unique speed combinations
speed_groups = df.groupby(["fan_back_speed", "fan_front_speed"])

# Step 1: For each speed pair, calculate the average temperature for each channel
for (back_speed, front_speed), group in speed_groups:
    
    # Step 1: Calculate average temperature over time for each channel
    T_avg = group[board_channels['A'] + board_channels['B'] + board_channels['C'] + board_channels['D']].mean()
    # T_avg = group[board_channels['X'] + board_channels['Y'] + board_channels['Z']].mean()
    
    # Step 2: For each board, calculate the difference between the max and min T_avg
    for board, channels in board_channels.items():
        board_T_avg = T_avg[channels]  # Average temperatures for the current board
        
        # Calculate max, min, and delta for the board
        max_channel_avg = board_T_avg.idxmax()  # Channel with maximum average temperature
        min_channel_avg = board_T_avg.idxmin()  # Channel with minimum average temperature
        max_T = board_T_avg.max()  # Maximum T_avg
        min_T = board_T_avg.min()  # Minimum T_avg
        delta_T_avg = max_T - min_T  # Delta between max and min
        
        # Step 3: Store results
        results.append({
            'fan_back_speed': back_speed,
            'fan_front_speed': front_speed,
            'board': board,
            'max_channel_avg': max_channel_avg,
            'min_channel_ave': min_channel_avg,
            'max_T_avg': max_T,
            'min_T_avg': min_T,
            'delta_T_avg': delta_T_avg
        })

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# Initialize an empty list to store results for the last temperature analysis
last_temp_results = []

for (back_speed, front_speed), group in speed_groups:
    
    # Step 2: For each board, calculate the last temperature (last_T), max, min, and delta
    for board, channels in board_channels.items():
        # Get the last temperature for the current board
        # last_T = group[channels].iloc[-1]  # Get the last row temperature values for the board
        last_T_avg = group[channels].tail(5).mean()
        
        # Calculate max, min, and delta for the last temperature (last_T)
        # max_channel_last = last_T.idxmax()  # Channel with maximum average temperature
        # min_channel_last = last_T.idxmin()  # Channel with minimum average temperature
        # max_T_last = last_T.max()  # Maximum last temperature
        # min_T_last = last_T.min()  # Minimum last temperature
        # delta_T_last = max_T_last - min_T_last  # Delta between max and min for the last temperature
        
        max_channel_last = last_T_avg.idxmax()  # Channel with maximum average temperature
        min_channel_last = last_T_avg.idxmin()  # Channel with minimum average temperature
        max_T_last = last_T_avg.max()  # Maximum last temperature
        min_T_last = last_T_avg.min()  # Minimum last temperature
        delta_T_last = max_T_last - min_T_last  # Delta between max and min for the last temperature
        
        # Step 3: Store the results for this speed pair and board
        last_temp_results.append({
            'fan_back_speed': back_speed,
            'fan_front_speed': front_speed,
            'board': board,
            'max_channel_last_avg': max_channel_last,
            'min_channel_last_avg': min_channel_last,
            'max_T_last_avg': max_T_last,
            'min_T_last_avg': min_T_last,
            'delta_T_last_avg': delta_T_last
        })


# Convert the results into a DataFrame
last_temp_results_df = pd.DataFrame(last_temp_results)

# Combine the original results with last temperature data (if needed)
final_results_df = pd.merge(results_df, last_temp_results_df, on=["fan_back_speed", "fan_front_speed", "board"], how="left")


threshold = 8
final_results_df[f'pass_threshold_avg_{threshold}'] = (final_results_df['delta_T_avg']<=threshold)
final_results_df[f'pass_threshold_last_avg_{threshold}'] = (final_results_df['delta_T_last_avg']<=threshold)

threshold = 10
final_results_df[f'pass_threshold_avg_{threshold}'] = (final_results_df['delta_T_avg']<=threshold)
final_results_df[f'pass_threshold_last_avg_{threshold}'] = (final_results_df['delta_T_last_avg']<=threshold)

# Get the current timestamp to generate the filename
timestamp_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Define the filename with the current timestamp
output_file_path = f"G:/aging_setup/temp_fan_test/temperature/{timestamp_now}_temperatures_processed.xlsx"

# # Save the DataFrame to an Excel file
# final_results_df.to_excel(output_file_path, index=False)

#%% Now look at the specific speed, channel, board, see if it falls under the temperature we're interested in

def calculate_T_threshold(T_min_threshold, T_max_threshold, df):
    # Initialize an empty list to store the results
    result_T_threshold = []

    # Get unique speed combinations
    speed_groups = df.groupby(["fan_back_speed", "fan_front_speed"])

    # Step 1: For each speed pair, calculate the average temperature for each channel
    for (back_speed, front_speed), group in speed_groups:

        # Flatten the list of channels from all boards
        all_channels = [ch for channels in board_channels.values() for ch in channels]
        
        # Compute mean over the last 5 rows for all channels
        last_5_avg = group[all_channels].tail(5).mean()

        for ch in all_channels:
            avg_T = last_5_avg[ch]  # Get the averaged temperature for the channel
            within_threshold = T_min_threshold <= avg_T <= T_max_threshold  # Check if within threshold

            # Store the result
            result_T_threshold.append({
                'fan_back_speed': back_speed,
                'fan_front_speed': front_speed,
                'channel': ch,
                'avg_last_5_T': avg_T,
                'within_threshold': within_threshold
            })

    # Convert results into a DataFrame
    result_T_threshold_df = pd.DataFrame(result_T_threshold)

    # Count how many channels are within the threshold for each fan speed pair
    threshold_summary_df = (
        result_T_threshold_df
        .groupby(["fan_back_speed", "fan_front_speed"])["within_threshold"]
        .sum()  # Count the number of True values
        .reset_index()
    )

    # Rename column for clarity
    threshold_summary_df.rename(columns={"within_threshold": "num_within_threshold"}, inplace=True)

    return result_T_threshold_df, threshold_summary_df

result_T_threshold_55_60, threshold_summary_55_60 = calculate_T_threshold(55, 60, df)
result_T_threshold_60_65, threshold_summary_60_65 = calculate_T_threshold(60, 65, df)
result_T_threshold_65_70, threshold_summary_65_70 = calculate_T_threshold(65, 70, df)
result_T_threshold_60_70, threshold_summary_60_70 = calculate_T_threshold(60, 70, df)
result_T_threshold_55_65, threshold_summary_55_65 = calculate_T_threshold(55, 65, df)





# # fig, ax = plt.subplots(figsize=(12,8))
# # df_tmp.plot(ax=ax)
# # fig.savefig("result.png")
