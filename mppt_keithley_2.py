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


def initialize_files(base_dir, channels, start_time):
    for ch in channels:
        filepath = get_filename_for_channel(base_dir, ch, start_time)
        with open(filepath, "w") as file:
            file.write("time, voltage, current\n")


def append_data_to_file(filepath, timestamp, data):
    with open(filepath, "a") as file:
        file.write(f"{timestamp}, {data['voltage']}, {data['current']}\n")

# Update how the files are named
def get_filename_for_channel(base_dir, channel, start_time):
    # Format the start_time as YYYY_mm_dd_HH_MM_SS
    formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime(start_time))
    # Create the filename using the formatted time and channel number
    filename = os.path.join(base_dir, "mppt", f"{formatted_time}_mppt_ch{channel}.txt")
    return filename


def perform_iv_scan(keithley, channel, base_directory, formatted_current_time):
    bypass_ch_msg = f"MODE{channel} BYP"
    lnbp_ch_msg = f"MODE{channel} LNBP"

    hex_msg_bypass_ch = create_umppt_message(bypass_ch_msg)
    hex_msg_lnbp_ch = create_umppt_message(lnbp_ch_msg)

    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_lnbp_ch))
    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_bypass_ch))

    rt_bwd = ResultTable(['Voltage', 'Current'], ['V', 'A'], params={'sweep_type': 'iv_bwd'})
    for v in range(220, 0, -1):
        keithley.apply_voltage(keithley.smua, v / 100)
        time.sleep(0.01)
        i = keithley.smua.measure.i()
        rt_bwd.append_row([v / 100, i])

    rt_fwd = ResultTable(['Voltage', 'Current'], ['V', 'A'], params={'sweep_type': 'iv_fwd'})
    for v in range(0, 220):
        keithley.apply_voltage(keithley.smua, v / 100)
        time.sleep(0.01)
        i = keithley.smua.measure.i()
        rt_fwd.append_row([v / 100, i])

    save_path = os.path.join(base_directory, "jv", "plots")
    os.makedirs(save_path, exist_ok=True)
    file_name = f"jv_CH{channel}_{formatted_current_time}"

    plt.plot(rt_bwd.data[:, 0], rt_bwd.data[:, 1], label='bwd')
    plt.plot(rt_fwd.data[:, 0], rt_fwd.data[:, 1], label='fwd')
    plt.title(f"CH{channel} IV Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, file_name + ".png"))
    plt.close()

    np.savetxt(os.path.join(base_directory, "jv", file_name + "_bwd.txt"), rt_bwd.data, delimiter=",", header="Voltage, Current", comments="")
    np.savetxt(os.path.join(base_directory, "jv", file_name + "_fwd.txt"), rt_fwd.data, delimiter=",", header="Voltage, Current", comments="")

    mppt_msg = f"MODE{channel} MPPT"
    hex_msg_mppt = create_umppt_message(mppt_msg)
    send_data(COM_PORT, BAUD_RATE, hex_to_bytes(hex_msg_mppt))
    

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

# def sanitize_ids(ids_list):
#     # Remove non-ASCII characters and ensure valid data
#     sanitized_ids = []
#     for id_str in ids_list:
#         sanitized_id = ''.join(c for c in id_str if c.isalnum())  # Keep only alphanumeric characters
#         sanitized_ids.append(sanitized_id)
#     return sanitized_ids

# def write_header_T(file_path, ids_list):
#     """Write the header to the file if it doesn't exist."""
#     global header_written_T

#     # Sanitize IDs to remove non-ASCII characters
#     sanitized_ids = [''.join(c for c in device_id if c.isalnum()) for device_id in ids_list]

#     if not header_written_T:
#         # Open the file with utf-8 encoding
#         with open(file_path, "w", encoding="utf-8") as file:
#             header = "time," + ",".join(sanitized_ids) + "\n"
#             file.write(header)
#         header_written_T = True

# def append_data_T(file_path, current_time, meas_list):
#     """Append cleaned and formatted data to the file."""
#     try:
#         # Clean and format each measurement
#         cleaned_meas_list = []
#         for value in meas_list:
#             # Remove invalid characters
#             cleaned_value = re.sub(r"[^0-9E\+\.\-]", "", value)
#             # Ensure proper scientific notation with E+01
#             if "E+" in cleaned_value:
#                 base, exponent = cleaned_value.split("E+")
#                 cleaned_value = f"{float(base):.6f}E+01"  # Reformat to 6 decimals and enforce E+01
#             cleaned_meas_list.append(cleaned_value)

#         # Open the file in append mode with UTF-8 encoding
#         with open(file_path, "a", encoding="utf-8") as file:
#             # Create the data line
#             data_line = current_time + "," + ",".join(cleaned_meas_list) + "\n"
#             # Write the data line
#             file.write(data_line)
#     except Exception as e:
#         print(f"Error writing data to file: {e}")
        
# def measure_temperature(com_port, baud_rate, base_dir, start_time):
#     """Measure temperatures and append data to the temperature file."""
#     formatted_time_start_universal = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time))
#     file_path = os.path.join(BASE_DIR + "temperature", f"{formatted_time_start_universal}_temperature.txt")
    
#     # Enumerate devices
#     msg_enum = create_umppt_message("ONEW:ENUM?")
#     send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_enum))
    
#     # Get device IDs
#     msg_ids = create_umppt_message("ONEW:IDS?")
#     response_ids = send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_ids))
#     ids_response = response_ids.decode('utf-8', errors='ignore')  # Decode bytes to string
#     ids_list = ids_response[5:].split(",")  # Remove the prefix and split the IDs
#     # ids_list = [id_str.strip() for id_str in ids_list if id_str.strip()]  # Clean up IDs
#     ids_list = [re.sub(r'[^\x20-\x7E]', '', id_str) for id_str in ids_list if id_str.strip()]
    
#     print(f"Device IDs: {ids_list}")  # Debugging
    
#     # Write header if not already written
#     global header_written_T
#     if not header_written_T:
#         with open(file_path, "w", encoding="utf-8") as file:
#             header = "time," + ",".join(ids_list) + "\n"
#             file.write(header)
#         header_written_T = True
    
#     # Get measurements
#     msg_meas = create_umppt_message("ONEW:MEAS?")
#     response_meas = send_and_receive_data(com_port, baud_rate, hex_to_bytes(msg_meas))
#     meas_response = response_meas.decode('utf-8', errors='ignore')  # Decode bytes to string
#     meas_list = meas_response[5:].split(",")  # Remove the prefix and split the IDs
#     meas_list = [meas_str.strip() for meas_str in meas_list if meas_str.strip()]  # Clean up measurements
    
#     print(f"Measurements: {meas_list}")  # Debugging
    
#     # Get the current timestamp
#     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    
#     # Append the data to the file
#     append_data_T(file_path, current_time, meas_list)

if __name__ == "__main__":
    COM_PORT = "COM3"
    BAUD_RATE = 125000
    ACTIVE_CHANNELS = [1, 2, 3, 4, 5, 6]
    BASE_DIR = "U:/20241105_microMPPT/data/20250123/"
    SUBDIRS = ["jv", "mppt", "jv/plots", "temperature"]
    TIME_IV = 180 #seconds, do IV every 3 minutes

    start_time_universal = time.time()
    start_time_change = time.time()
    formatted_time_start_universal = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(start_time_universal))
    
    setup_directories(BASE_DIR, SUBDIRS)
    initialize_files(BASE_DIR, ACTIVE_CHANNELS, start_time_universal)

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

            # Measure MPPT
            for channel in ACTIVE_CHANNELS:                
                msg_fetc = create_umppt_message(f"FETC{channel}?")
                reply = send_and_receive_data(COM_PORT, BAUD_RATE, hex_to_bytes(msg_fetc))
                parsed = parse_reply(reply)
                if parsed:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    # filepath = os.path.join(BASE_DIR, "mppt", f"mppt_ch{channel}.txt")
                    filepath = get_filename_for_channel(BASE_DIR, channel, start_time_universal)
                    append_data_to_file(filepath, timestamp, parsed)
                    print(f"Channel {channel}: {parsed}")
                    
            # If it reaches the time point interest, do IV sweep
            if (time.time() - start_time_change) >= TIME_IV:
                for channel in ACTIVE_CHANNELS:
                    perform_iv_scan(keithley, channel, BASE_DIR, time.strftime("%Y%m%d_%H%M%S"))
                start_time_change = time.time()

            time.sleep(5)
        except KeyboardInterrupt:
            print("Data acquisition stopped.")
            break
