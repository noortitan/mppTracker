# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:55:58 2025

@author: Titan.Hartono
"""

import serial
import time
import os
import re
import numpy as np
from binascii import hexlify
from keithley2600 import Keithley2600, ResultTable
import matplotlib.pyplot as plt
import pandas as pd
# import matplotlib.cm as cm


def reverse_bits(value, bit_width=8):
    return int(f"{value:0{bit_width}b}"[::-1], 2)


def crc16_ibm(data: bytes, poly=0x8005, init=0x0000, final_xor=0x0000):
    crc = init
    for byte in data:
        byte = reverse_bits(byte)
        crc ^= byte << 8
        for _ in range(8):
            crc = (crc << 1) ^ poly if crc & 0x8000 else crc << 1
            crc &= 0xFFFF
    return reverse_bits(crc, 16) ^ final_xor


def hex_to_bytes(hex_string):
    return bytes.fromhex(hex_string.replace(" ", ""))


def create_umppt_message(message, des=1, source=0):
    message_bytes = message.encode()
    header = f"{des:02X}{source:02X}{len(message_bytes):02X}"
    payload = hexlify(message_bytes).decode()
    full_message = header + payload
    crc = crc16_ibm(hex_to_bytes(full_message))
    return f"55{full_message}{crc:04X}AA"


def send_and_receive_data(port, baudrate, data, end_marker=b'\xAA', timeout=1):
    try:
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            ser.write(data)
            print(f"Data sent: {data}")
            reply = b''
            while not reply.endswith(end_marker):
                reply += ser.read(ser.in_waiting or 1)
            print(f"Reply received: {reply}")
            return reply
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None
    
def send_data(port, baudrate, data, end_marker=b'\xaa', timeout=1):
    """
    Sends a byte string over a specified COM port and reads the reply until an end marker is detected.

    Args:
        port (str): The COM port to use (e.g., 'COM3', '/dev/ttyUSB0').
        baudrate (int): The baud rate for the serial connection.
        data (bytes): The byte string to send.
        end_marker (bytes): The marker that indicates the end of the response.
        timeout (float): Timeout in seconds for reading the reply.

    Returns:
        bytes: The full reply received from the COM port.
    """
    try:
        # Initialize serial connection
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            # Send the byte string
            ser.write(data)
            print(f"Data sent: {data}")

    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def parse_reply(reply):
    try:
        decoded = reply[1:-1].decode("ascii", errors="ignore")
        matches = re.findall(r'[+-]?\d+\.\d+E[+-]?\d+', decoded)
        if len(matches) < 2:
            raise ValueError("Insufficient data in reply")
        return {"voltage": float(matches[0]), "current": float(matches[1])}
    except Exception as e:
        print(f"Parsing error: {e}")
        return None


def setup_directories(base_dir, subdirs):
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)


def initialize_files(base_dir, channels, start_time, formatted_time_start_universal):
    # Create file for saving jv params
    # formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(start_time))
    filepath = os.path.join(base_dir, "jv", f"{formatted_time_start_universal}_jv_params.txt")
    with open(filepath, "w") as file:
        file.write("time, channel, dir, pce, voc, jsc, ff\n")
    
    # Create files for saving mppt for each channel
    for ch in channels:
        filepath = get_filename_for_channel(base_dir, ch, start_time)
        with open(filepath, "w") as file:
            file.write("time, voltage, current, area, irradiance, pce\n")


def append_data_to_file(filepath, timestamp, data, CELL_AREA, IRRADIANCE):
    pce = data['voltage']*data['current']*100/(CELL_AREA*IRRADIANCE)
    with open(filepath, "a") as file:
        file.write(f"{timestamp}, {data['voltage']}, {data['current']}, {CELL_AREA}, {IRRADIANCE}, {pce}\n")

# Update how the files are named
def get_filename_for_channel(base_dir, channel, start_time):
    # Format the start_time as YYYY_mm_dd_HH_MM_SS
    formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(start_time))
    # Create the filename using the formatted time and channel number
    filename = os.path.join(base_dir, "mppt", f"{formatted_time}_mppt_ch{channel}.txt")
    return filename


def calculate_solar_cell_params(data, irradiance=1000):
    """
    Calculate efficiency, Jsc, Voc, and FF from IV data.

    Parameters:
    data (numpy.ndarray): IV data with columns [Voltage (V), Current (A), Current Density (A/m²)]
    irradiance (float): Incident power density in W/m² (default is 1000 W/m² for 1 Sun illumination)

    Returns:
    dict: Dictionary with efficiency (%), Jsc (mA/cm²), Voc (V), and FF.
    """
    voltage = data[:, 0]  # Voltage (V)
    current = data[:, 1]  # Current (A)
    current_density_A_m2 = data[:, 2]  # Current Density (A/m²)

    # Convert Current Density from A/m² to mA/cm²
    current_density_mA_cm2 = current_density_A_m2 * 0.1  # (1 A/m² = 0.1 mA/cm²)

    # Convert Pin from W/m² to mW/cm²
    irradiance_mW_cm2 = irradiance * 0.1  # (1 W/m² = 0.1 mW/cm²)

    # Short-Circuit Current Density (Jsc) → max current at V=0
    jsc = np.abs(np.interp(0, voltage, current_density_mA_cm2)) # make it positive, otherwise jsc is negative
    
    # np.interp needs to be monotonically increasing for interp to work
    if current_density_mA_cm2[0] > current_density_mA_cm2[-1]:  
        # Flip arrays if necessary
        current_density_mA_cm2 = current_density_mA_cm2[::-1]
        voltage = voltage[::-1]

    # Open-Circuit Voltage (Voc) → voltage where current is zero
    voc = np.interp(0, current_density_mA_cm2, voltage)

    # Maximum Power Point (MPP)
    power_density = voltage * current_density_mA_cm2  # Power density (mW/cm²)
    max_power_index = np.argmax(power_density)
    pmp = power_density[max_power_index]  # Max power density (mW/cm²)

    # Fill Factor (FF)
    ff = pmp / (jsc * voc) if (jsc * voc) > 0 else 0

    # Efficiency (η) = Pmp / Pin (in %)
    efficiency = (pmp / irradiance_mW_cm2) * 100  # Convert to percentage

    return {
        "pce": efficiency,
        "jsc": jsc,
        "voc": voc,
        "ff": ff
    }

def perform_iv_scan(keithley, channel, base_directory, formatted_current_time_underscore, 
                    CELL_AREA, IRRADIANCE, formatted_time_start_universal,formatted_current_time_normal):
    bypass_ch_msg = f"MODE{channel} BYP"
    lnbp_ch_msg = f"MODE{channel} LNBP"

    hex_msg_bypass_ch = create_umppt_message(bypass_ch_msg)
    hex_msg_lnbp_ch = create_umppt_message(lnbp_ch_msg)

    send_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(hex_msg_lnbp_ch))
    send_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(hex_msg_bypass_ch))
    
    # Forward first
    rt_fwd = ResultTable(['Voltage', 'Current', 'Current Density'], ['V', 'A', 'A/m2'], params={'sweep_type': 'iv_fwd'})
    for v in range(0, 220):
        keithley.apply_voltage(keithley.smua, v / 100)
        time.sleep(0.01)
        i = keithley.smua.measure.i()
        rt_fwd.append_row([v / 100, i, i/CELL_AREA])
    
    rt_bwd = ResultTable(['Voltage', 'Current', 'Current Density'], ['V', 'A', 'A/m2'], params={'sweep_type': 'iv_bwd'})
    for v in range(220, -1, -1):
        keithley.apply_voltage(keithley.smua, v / 100)
        time.sleep(0.01)
        i = keithley.smua.measure.i()
        rt_bwd.append_row([v / 100, i, i/CELL_AREA])
        
    # bwd_data = rt_bwd.data
    # fwd_data = rt_fwd.data
    
    # Calculate solar cell params
    cell_params_bwd = calculate_solar_cell_params(rt_bwd.data, IRRADIANCE)
    cell_params_fwd = calculate_solar_cell_params(rt_fwd.data, IRRADIANCE)
    
    # Save solar cells params into txt file
    filepath_cell_params = os.path.join(base_directory, "jv", f"{formatted_time_start_universal}_jv_params.txt")
    with open(filepath_cell_params, "a") as file:
        file.write(f"{formatted_current_time_normal}, {channel}, fwd, {cell_params_fwd['pce']}, {cell_params_fwd['voc']}, {cell_params_fwd['jsc']}, {cell_params_fwd['ff']}\n")
        file.write(f"{formatted_current_time_normal}, {channel}, bwd, {cell_params_bwd['pce']}, {cell_params_bwd['voc']}, {cell_params_bwd['jsc']}, {cell_params_bwd['ff']}\n")

    save_path = os.path.join(base_directory, "jv", "plots")
    os.makedirs(save_path, exist_ok=True)
    file_name = f"jv_CH{channel}_{formatted_current_time_underscore}"

    plt.plot(rt_bwd.data[:, 0], rt_bwd.data[:, 1], label='bwd')
    plt.plot(rt_fwd.data[:, 0], rt_fwd.data[:, 1], label='fwd')
    plt.title(f"CH{channel} IV Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, file_name + ".png"))
    plt.close()

    np.savetxt(os.path.join(base_directory, "jv", file_name + "_bwd.txt"), rt_bwd.data, delimiter=",", header="Voltage, Current, Current Density", comments="")
    np.savetxt(os.path.join(base_directory, "jv", file_name + "_fwd.txt"), rt_fwd.data, delimiter=",", header="Voltage, Current, Current Density", comments="")

    mppt_msg = f"MODE{channel} MPPT"
    hex_msg_mppt = create_umppt_message(mppt_msg)
    send_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(hex_msg_mppt))
    
    return rt_bwd, rt_fwd
    
#--------------------------------------------------------------------------------------------------------------------------

def count_sensors(ser):
    response = send_command_T(ser, "COUNT")
    for line in response:
        print(line)

def read_response_until_marker_T(ser, marker="Transmission finished", timeout=10):
    """Read lines from the serial port until the specified marker is received or timeout occurs."""
    ser.reset_input_buffer()  
    response = []
    start_time = time.time()
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            response.append(line)
            if marker in line:
                break
        if time.time() - start_time > timeout:
            print("Warning: Timeout reached before marker was received.")
            break
    return response

def send_command_T(ser, command, expect_multiple_lines=False, marker="Transmission finished", timeout=10):
    """Send a command over serial and retrieve the response with a timeout."""
    ser.reset_input_buffer()
    ser.write((command + "\n").encode())
    time.sleep(0.1)  # Allow Arduino to process

    response = []
    start_time = time.time()

    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            response.append(line)
            if marker in line:
                break

        if time.time() - start_time > timeout:  # Timeout condition
            print(f"WARNING: Timeout waiting for response to {command}")
            break

    return response

def list_sensor_ids(ser):
    """Retrieve and return a list of sensor IDs in the correct order."""
    response = send_command_T(ser, "LIST", expect_multiple_lines=True, timeout=10)
    sensor_ids = []
    for line in response:
        if line.startswith("Sensor "):  
            parts = line.split(": ")
            if len(parts) == 2:
                sensor_ids.append(parts[1].strip())  
    return sensor_ids  


def get_sensor_temp(ser, sensor_id):
    """Retrieve temperature for a single sensor and extract the numeric value."""
    response = send_command_T(ser, f"GET {sensor_id}")

    for line in response:
        print(f"DEBUG: Response from {sensor_id}: {line}")  # Debugging

        # Extract temperature using regex
        match = re.search(r"([-+]?\d*\.\d+|\d+) C", line)
        if match:
            return match.group(1)  # Return only the numeric temperature value

    print(f"WARNING: No valid temperature from {sensor_id}, returning 'NaN'")
    return "NaN"  # Return NaN if no valid temperature is found

def initialize_T_sensors(ser, BASE_DIR, save_as_channels=True):
    sensor_ids = list_sensor_ids(ser)
    if not sensor_ids:
        print("No sensors detected.")
        return
    
    dict_cells = { # new version
        '28B6F4B60F0000D5':'A1', # A
        '28E6D7B60F00004D':'A2',
        '28D4DFB60F0000F0':'A3',
        '28FCC3B60F0000BA':'A4',
        '288958B60F00000E':'A5',
        '28655AA30F000060':'A6',
        '28091DB60F0000DF':'B1', # B
        '28C531B70F00006B':'B2',
        '285359B70F000047':'B3',
        '28E48FB60F000088':'B4',
        '287C9FB60F0000E4':'B5',
        '28868FB60F000025':'B6',
        '28F55CB60F0000F4':'C1', # C
        '28B71FB70F0000C7':'C2',
        '287EA4B60F00007E':'C3',
        '28727BB60F000032':'C4',
        '28005AB60F0000F1':'C5',
        '28991DB70F0000E1':'C6',
        '281698A70F00000E':'D1', # D
        '28A063A80F0000F7':'D2',
        '280362A80F00003F':'D3',
        '28BBDAA70F000072':'D4',
        '28C4A2A70F00009D':'D5',
        '287C77A80F00002D':'D6',
        '287A3CA8040000C7':'ambient',
        }
    
    dict_channels = { # new version
        '28B6F4B60F0000D5':24, # A
        '28E6D7B60F00004D':23,
        '28D4DFB60F0000F0':22,
        '28FCC3B60F0000BA':21,
        '288958B60F00000E':20,
        '28655AA30F000060':19,
        '28091DB60F0000DF':18, # B
        '28C531B70F00006B':17,
        '285359B70F000047':16,
        '28E48FB60F000088':15,
        '287C9FB60F0000E4':14,
        '28868FB60F000025':13,
        '28F55CB60F0000F4':6, # C
        '28B71FB70F0000C7':5,
        '287EA4B60F00007E':4,
        '28727BB60F000032':3,
        '28005AB60F0000F1':2,
        '28991DB70F0000E1':1,
        '281698A70F00000E':12, # D
        '28A063A80F0000F7':11,
        '280362A80F00003F':10,
        '28BBDAA70F000072':9,
        '28C4A2A70F00009D':8,
        '287C77A80F00002D':7,
        '287A3CA8040000C7':'ambient',
        }
    
    # dict_cells = { # old version
    #     '281254A30F0000DF':'A1', # A
    #     '289658A30F0000C8':'A2',
    #     '286E52A30F0000A6':'A3',
    #     '286B62A30F0000C9':'A4',
    #     '288D4CA30F000008':'A5',
    #     '288A5BA30F0000A0':'A6',
    #     '281652A30F00009F':'B1', # B
    #     '28B97AA30F000021':'B2',
    #     '2827B4A20F00008C':'B3',
    #     '282064A30F0000D8':'B4',
    #     '284B2CA30F000034':'B5',
    #     '289255A30F0000F8':'B6',
    #     '28665EA30F000026':'C1', # C
    #     '281A5CA30F000040':'C2',
    #     '28165AA30F0000A1':'C3',
    #     '285237A30F0000F5':'C4',
    #     '28005DA30F000019':'C5',
    #     '28E433A30F00005F':'C6',
    #     '28F24BA30F0000E5':'D1', # D
    #     '28716AA30F000063':'D2',
    #     '281A57A30F000030':'D3',
    #     '2885BBA20F0000D1':'D4',
    #     '28CB5FA30F0000FD':'D5',
    #     '28FA64A30F0000D3':'D6',
    #     }
    
    # dict_channels = { # old version
    #     '281254A30F0000DF':24, # A
    #     '289658A30F0000C8':23,
    #     '286E52A30F0000A6':22,
    #     '286B62A30F0000C9':21,
    #     '288D4CA30F000008':20,
    #     '288A5BA30F0000A0':19,
    #     '281652A30F00009F':18, # B
    #     '28B97AA30F000021':17,
    #     '2827B4A20F00008C':16,
    #     '282064A30F0000D8':15,
    #     '284B2CA30F000034':14,
    #     '289255A30F0000F8':13,
    #     '28665EA30F000026':6, # C
    #     '281A5CA30F000040':5,
    #     '28165AA30F0000A1':4,
    #     '285237A30F0000F5':3,
    #     '28005DA30F000019':2,
    #     '28E433A30F00005F':1,
    #     '28F24BA30F0000E5':12, # D
    #     '28716AA30F000063':11,
    #     '281A57A30F000030':10,
    #     '2885BBA20F0000D1':9,
    #     '28CB5FA30F0000FD':8,
    #     '28FA64A30F0000D3':7,
    #     }

    # filename = "temperature_log_2.txt"
    filename = os.path.join(BASE_DIR, "temperature", time.strftime("%Y_%m_%d_%H_%M_%S") + "_temperature.txt")
    
    # Generate the header row with channel names (CH1, CH2, etc.)
    channel_names = [f"CH{dict_channels[sensor_id]}" for sensor_id in sensor_ids]
    
    if save_as_channels == False:
        file_exists = os.path.exists(filename)
        with open(filename, "a") as file:
            if not file_exists:
                file.write("time," + ",".join(sensor_ids) + "\n")  
                
    if save_as_channels == True:
        file_exists = os.path.exists(filename)
        with open(filename, "a") as file:
            if not file_exists:
                file.write("time," + ",".join(channel_names) + "\n")
    
    return sensor_ids, filename

def record_temperatures(ser, BASE_DIR, sensor_ids, filename_T, save_as_channels):

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 🚀 Send one command to get all temperatures at once
    response = send_command_T(ser, "ALL", expect_multiple_lines=True)

    temperatures = []
    for sensor_id in sensor_ids:
        matched_temp = None
        for line in response:
            if sensor_id in line:
                match = re.search(r"([-+]?\d*\.\d+|\d+) C", line)
                if match:
                    matched_temp = match.group(1)
                    break
        if matched_temp:
            temperatures.append(matched_temp)
        else:
            temperatures.append("NaN")  # Handle missing data

    row = timestamp + "," + ",".join(temperatures)
    with open(filename_T, "a") as file:
        file.write(row + "\n")

    print(f"Recorded at {timestamp}: {row}")  
    # time.sleep(1)  # Reduce sleep to 1s for faster updates


def all_sensors_temp(ser):
    response = send_command_T(ser, "ALL", expect_multiple_lines=True)
    for line in response:
        print(line)

def set_fan1_speed(ser, speed):
    response = send_command_T(ser, f"FAN1 {speed}")
    for line in response:
        print(line)

def set_fan2_speed(ser, speed):
    response = send_command_T(ser, f"FAN2 {speed}")
    for line in response:
        print(line)

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

#--------------------------------------------------------------------
    
def load_temperature_data(folder_path):
    """Loads temperature data from a single .txt file in the given folder."""
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, header=0, parse_dates=['time'])
            df.columns = [col.strip() for col in df.columns]  # Clean column names
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            
            # Convert sensor data to numeric
            for col in df.columns:
                if col != 'time':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna()  # Return cleaned data
    return None  # Return None if no temperature file is found

def plot_mppt_temperature(mppt_folder_path, temperature_folder_path):
    """Plots MPPT data (Voltage, Current, Power, PCE) and temperature over time."""
    mppt_data = load_mppt_data(mppt_folder_path, ['time', 'voltage', 'current'])
    
    # Compute power
    for df in mppt_data.values():
        df['power'] = df['voltage'] * df['current']
    
    temperature_data = load_temperature_data(temperature_folder_path)
    
    # Find the latest timestamp across all MPPT channels
    latest_timestamp = max(df['time'].iloc[-1] for df in mppt_data.values())
    timestamp_str = latest_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(max(len(mppt_data), len(temperature_data.columns) - 1))]

    plt.figure(figsize=(15, 20))

    plot_configs = [
        ("Voltage (V)", "voltage", 1),
        ("Current (A)", "current", 2),
        ("Power (W)", "power", 3),
        ("PCE (%)", "pce", 4)
    ]

    for title, key, idx in plot_configs:
        plt.subplot(5, 1, idx)
        for i, (channel, df) in enumerate(mppt_data.items()):
            plt.plot(df['time'], df[key], label=channel, color=colors[i])
        plt.xlabel('Time')
        plt.ylabel(title)
        plt.title(f'Time vs {title}')
        plt.legend()
        plt.grid()

    # Temperature Plot
    plt.subplot(5, 1, 5)
    for i, sensor in enumerate(temperature_data.columns[1:]):  
        plt.plot(temperature_data['time'], temperature_data[sensor], label=sensor, color=colors[i])
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.title('Time vs Temperature')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    filename = os.path.join(mppt_folder_path, f"{timestamp_str}_mppt_temperature.png")
    plt.savefig(filename, dpi=300)
    print(f"Figure saved as {filename}")
    plt.show()

if __name__ == "__main__":
    COM_PORT_MPPT, BAUD_RATE_MPPT = "COM3", 125000
    COM_PORT_ARDUINO, BAUD_RATE_ARDUINO = "COM7", 9600

    keithley = Keithley2600('TCPIP0::169.254.0.1::inst0::INSTR')

    ACTIVE_CHANNELS = [3, 4, 8, 9, 12, 14, 16, 19, 20, 21, 22, 23, 24]
    CELL_AREA, IRRADIANCE = 1E-4, 1000 # irradiance in W/m2, cell area in m2
    BASE_DIR = "U:/20241105_microMPPT/data/20250206_light/"
    SUBDIRS = ["jv", "mppt", "jv/plots", "temperature"]
    TIME_IV, TIME_SLEEP = 300, 5 # seconds
    SPEED_FAN1, SPEED_FAN2 = 80, 80 # percent

    # Initialize MPPT connection
    idn_message = hex_to_bytes("550100052a49444e3fA6FBaa")
    print(send_and_receive_data(COM_PORT_MPPT, BAUD_RATE_MPPT, idn_message))

    for channel in ACTIVE_CHANNELS:
        msg = create_umppt_message(f"MODE{channel} MPPT")
        send_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(msg))

    # Initialize Arduino connection
    try:
        ser = serial.Serial(COM_PORT_ARDUINO, BAUD_RATE_ARDUINO, timeout=5)
    except serial.SerialException as e:
        print(f"Error opening serial port {COM_PORT_ARDUINO}: {e}")
        exit()

    # Setup fan speeds
    set_fan1_speed(ser, SPEED_FAN1)
    set_fan2_speed(ser, SPEED_FAN2)

    # Time tracking
    start_time_universal = time.time()
    start_time_change = time.time()
    formatted_time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time_universal))

    setup_directories(BASE_DIR, SUBDIRS)
    initialize_files(BASE_DIR, ACTIVE_CHANNELS, start_time_universal, formatted_time_start)
    sensor_ids, filename_T = initialize_T_sensors(ser, BASE_DIR, save_as_channels=True)

    while True:
        try:
            # Trigger MPPT measurement
            send_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(create_umppt_message("*TRG")))
            
            # Measure temperatures
            record_temperatures(ser, BASE_DIR, sensor_ids, filename_T, save_as_channels=True)

            # Measure MPPT data
            for channel in ACTIVE_CHANNELS:
                msg_fetc = create_umppt_message(f"FETC{channel}?")
                reply = send_and_receive_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(msg_fetc))
                parsed = parse_reply(reply)
                if parsed:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    filepath = get_filename_for_channel(BASE_DIR, channel, start_time_universal)
                    append_data_to_file(filepath, timestamp, parsed, CELL_AREA, IRRADIANCE)
                    print(f"Channel {channel}: {parsed}")

            # IV sweep check
            if (time.time() - start_time_change) >= TIME_IV:
                for channel in ACTIVE_CHANNELS:
                    perform_iv_scan(keithley, channel, BASE_DIR, time.strftime("%Y_%m_%d_%H_%M_%S"),
                                    CELL_AREA, IRRADIANCE, formatted_time_start, time.strftime("%Y-%m-%d %H:%M:%S"))

                    # Continue MPPT during IV sweep
                    for channel_mppt in ACTIVE_CHANNELS:
                        msg_fetc = create_umppt_message(f"FETC{channel_mppt}?")
                        reply = send_and_receive_data(COM_PORT_MPPT, BAUD_RATE_MPPT, hex_to_bytes(msg_fetc))
                        parsed = parse_reply(reply)
                        if parsed:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            filepath = get_filename_for_channel(BASE_DIR, channel_mppt, start_time_universal)
                            append_data_to_file(filepath, timestamp, parsed, CELL_AREA, IRRADIANCE)
                            print(f"Channel {channel_mppt}: {parsed}")

                start_time_change = time.time()

            time.sleep(TIME_SLEEP)

        except KeyboardInterrupt:
            ser.close()
            print("Data acquisition stopped, now plotting...")
            plot_mppt_temperature(os.path.join(BASE_DIR, "mppt"), os.path.join(BASE_DIR, "temperature"))
            print("Plotting is done")
            break