# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:49:10 2025

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

    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_lnbp_ch))
    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_bypass_ch))
    
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
    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_mppt))
    
    return rt_bwd, rt_fwd
    

def sanitize_ids(ids_list):
    """Sanitize device IDs by keeping only valid alphanumeric characters."""
    sanitized_ids = []
    for id_str in ids_list:
        sanitized_id = ''.join(c for c in id_str if c.isalnum())  # Keep only alphanumeric characters
        sanitized_ids.append(sanitized_id)
    return sanitized_ids

def write_header_T(file_path, ids_list):
    """Write the header to the file if it doesn't exist."""
    global header_written_T

    # Sanitize IDs to remove non-ASCII characters
    sanitized_ids = sanitize_ids(ids_list)

    if not header_written_T:
        # Open the file with utf-8 encoding
        with open(file_path, "w", encoding="utf-8") as file:
            header = "time," + ",".join(sanitized_ids) + "\n"
            file.write(header)
        header_written_T = True

def append_data_T(file_path, current_time, meas_list):
    """Append cleaned and formatted data to the file."""
    try:
        # Clean and format each measurement
        cleaned_meas_list = []
        for value in meas_list:
            # Remove invalid characters
            cleaned_value = re.sub(r"[^0-9E\+\.\-]", "", value)
            # Ensure proper scientific notation with E+01
            if "E+" in cleaned_value:
                base, exponent = cleaned_value.split("E+")
                cleaned_value = f"{float(base):.6f}E+01"  # Reformat to 6 decimals and enforce E+01
            cleaned_meas_list.append(cleaned_value)

        # Open the file in append mode with UTF-8 encoding
        with open(file_path, "a", encoding="utf-8") as file:
            # Create the data line
            data_line = current_time + "," + ",".join(cleaned_meas_list) + "\n"
            # Write the data line
            file.write(data_line)
    except Exception as e:
        print(f"Error writing data to file: {e}")

def measure_temperature(com_port, baud_rate, base_dir, start_time):
    """Measure temperatures and append data to the temperature file."""
    formatted_time_start_universal = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
    file_path = os.path.join(base_dir + "temperature", f"{formatted_time_start_universal}_temperature.txt")
    
    # Enumerate devices
    msg_enum = create_umppt_message("ONEW:ENUM?")
    send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_enum))
    
    # Get device IDs
    msg_ids = create_umppt_message("ONEW:IDS?")
    response_ids = send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_ids))
    # ids_response = response_ids.decode('utf-8', errors='ignore')  # Decode bytes to string
    # ids_list = ids_response[5:].split(",")  # Remove the prefix and split the IDs
    
    # response_ids_cleaned = re.sub(rb'[^\x20-\x7E]', b'', response_ids)  # Keep only printable ASCII characters
    # ids_response = response_ids_cleaned.decode('utf-8', errors='ignore')  # Decode the cleaned byte string
    # ids_list = re.findall(r'[A-Fa-f0-9]{16}', ids_response)  # Extract valid 16-character hex IDs using regex
    
    # Decode and clean response
    response_ids_cleaned = re.sub(rb'[^\x20-\x7E]', b'', response_ids)  # Keep only printable ASCII
    ids_response = response_ids_cleaned.decode('utf-8', errors='ignore')  # Decode cleaned response
    
    # Debugging: Check cleaned response
    print(f"Cleaned Response: {ids_response}")
    
    # Explicitly remove known prefixes ('U\x00\x01e') if present
    if ids_response.startswith('U'):
        ids_response = ids_response[1:]
    if ids_response.startswith('\x00'):
        ids_response = ids_response[1:]
    if ids_response.startswith('e'):
        ids_response = ids_response[1:]

    # Use regex to extract valid 16-character hexadecimal IDs
    ids_list = re.findall(r'\b[A-Fa-f0-9]{16}\b', ids_response)
    
    # Clean up device IDs
    # ids_list = [id_str.strip() for id_str in ids_list if id_str.strip()]  # Remove empty strings
    # ids_list = sanitize_ids(ids_list)  # Sanitize device IDs
    print(f"Sanitized Device IDs: {ids_list}")  # Debugging
    
    # Write header if not already written
    write_header_T(file_path, ids_list)
    
    # Get measurements
    msg_meas = create_umppt_message("ONEW:MEAS?")
    response_meas = send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_meas))
    meas_response = response_meas.decode('utf-8', errors='ignore')  # Decode bytes to string
    meas_list = meas_response[5:].split(",")  # Remove the prefix and split the measurements
    meas_list = [meas_str.strip() for meas_str in meas_list if meas_str.strip()]  # Clean up measurements
    print(f"Cleaned Measurements: {meas_list}")  # Debugging
    
    # Get the current timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Append the data
    append_data_T(file_path, current_time, meas_list)

if __name__ == "__main__":
    COM_PORT = "COM3"
    BAUD_RATE = 125000
    ACTIVE_CHANNELS = [13, 14, 15, 16, 17, 18] #[1, 2, 3, 4, 5, 6]
    CELL_AREA = 4E-4 # m2, cm2 -> m2 E-4
    IRRADIANCE = 300 # in W/m2, 1-Sun: 100 mW/cm2 = 1000 W/m2
    BASE_DIR = "U:/20241105_microMPPT/data/20250204_B/"
    SUBDIRS = ["jv", "mppt", "jv/plots", "temperature"]
    TIME_IV = 120 #seconds, do IV every 2 minutes
    TIME_SLEEP = 5 #seconds

    start_time_universal = time.time()
    start_time_change = time.time()
    formatted_time_start_universal = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time_universal))
    
    setup_directories(BASE_DIR, SUBDIRS)
    initialize_files(BASE_DIR, ACTIVE_CHANNELS, start_time_universal, formatted_time_start_universal)

    # Initialize the T file with the header
    header_written_T = False

    keithley = Keithley2600('TCPIP0::169.254.0.1::inst0::INSTR')

    idn_message = hex_to_bytes("550100052a49444e3fA6FBaa") # *IDN?
    print(send_and_receive_data(COM_PORT, BAUD_RATE, idn_message))

    for channel in ACTIVE_CHANNELS:
        msg = create_umppt_message(f"MODE{channel} MPPT")
        send_data(COM_PORT, BAUD_RATE, hex_to_bytes(msg))

    
    while True:
        try:
            msg_trg = create_umppt_message("*TRG")
            send_data(COM_PORT, BAUD_RATE, hex_to_bytes(msg_trg))
            
            # Measure temperatures
            measure_temperature(COM_PORT, BAUD_RATE, BASE_DIR, start_time_universal)

            # Measure MPPT, break if the time has reached to measure IV
            for channel_mppt in ACTIVE_CHANNELS:                
                msg_fetc = create_umppt_message(f"FETC{channel_mppt}?")
                reply = send_and_receive_data(COM_PORT, BAUD_RATE, hex_to_bytes(msg_fetc))
                parsed = parse_reply(reply)
                if parsed:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # filepath = os.path.join(BASE_DIR, "mppt", f"mppt_ch{channel_mppt}.txt")
                    filepath = get_filename_for_channel(BASE_DIR, channel_mppt, start_time_universal)
                    append_data_to_file(filepath, timestamp, parsed, CELL_AREA, IRRADIANCE)
                    print(f"Channel {channel_mppt}: {parsed}")
                    
            # If it reaches the time point interest, do IV sweep, but in the mean time also do MPPT
            if (time.time() - start_time_change) >= TIME_IV:
                for channel_iv in ACTIVE_CHANNELS:
                    rt_bwd, rt_fwd = perform_iv_scan(keithley, channel_iv, BASE_DIR, 
                                                     time.strftime("%Y_%m_%d_%H_%M_%S"), CELL_AREA, 
                                                     IRRADIANCE, formatted_time_start_universal,
                                                     time.strftime("%Y-%m-%d %H:%M:%S"))
                    
                    # Enter MPPT loop in the meantime
                    for channel_mppt in ACTIVE_CHANNELS:                
                        msg_fetc = create_umppt_message(f"FETC{channel_mppt}?")
                        reply = send_and_receive_data(COM_PORT, BAUD_RATE, hex_to_bytes(msg_fetc))
                        parsed = parse_reply(reply)
                        if parsed:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            # filepath = os.path.join(BASE_DIR, "mppt", f"mppt_ch{channel_mppt}.txt")
                            filepath = get_filename_for_channel(BASE_DIR, channel_mppt, start_time_universal)
                            append_data_to_file(filepath, timestamp, parsed, CELL_AREA, IRRADIANCE)
                            print(f"Channel {channel_mppt}: {parsed}")
                    
                start_time_change = time.time()

            time.sleep(TIME_SLEEP)
            
        except KeyboardInterrupt:
            print("Data acquisition stopped.")
            break
