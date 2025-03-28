# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:54:06 2025

@author: Titan.Hartono
"""

import serial
import time
from  keithley2600 import Keithley2600, ResultTable
import matplotlib.pyplot as plt
import re
import numpy as np
import os

from binascii import hexlify

def reverse_bits(value, bit_width=8):
    """
    Reverse the bits of a value (e.g., for reflection).
    
    :param value: Value to reverse (e.g., a byte or a word)
    :param bit_width: Number of bits in the value
    :return: Bit-reversed value
    """
    result = 0
    for _ in range(bit_width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result

def crc16_ibm(data: bytes, poly=0x8005, init=0x0000, final_xor=0x0000):
    """
    Calculate the CRC-16-IBM checksum for a given data byte array.

    :param data: Byte array of data to calculate CRC for
    :param poly: Polynomial (default: 0x8005, CRC-16-IBM)
    :param init: Initial value (default: 0x0000)
    :param final_xor: Final XOR value (default: 0x0000)
    :return: CRC-16 checksum as an integer
    """
    # Initialize CRC with the initial value
    crc = init

    # Process each byte with bit reflection
    for byte in data:
        byte = reverse_bits(byte)  # Reflect the input byte
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFFFF  # Ensure CRC remains 16-bit

    # Reflect the final CRC and apply the final XOR
    crc = reverse_bits(crc, bit_width=16)
    return crc ^ final_xor

def hex_to_bytes(hex_string):
    """
    Convert a hex string to a byte array.
    
    :param hex_string: String of hex data (e.g., "55 52 00 05 2A 49 44 4E 3F")
    :return: Byte array
    """
    return bytes.fromhex(hex_string.replace(" ", ""))

def text_to_umppt(message, des=1, source=0):
    mess_len = len(message)
    
    des = f"{des:02X}"
    source = f"{source:02X}"
    mess_len = f"{mess_len:02X}"
    
    extender = des + source + mess_len
    
    message_extended = extender + hexlify(message.encode()).decode()
    
    ba_message_extended = hex_to_bytes(message_extended)
    
    crc = crc16_ibm(ba_message_extended)
    
    full_message = "55" + message_extended + f"{crc:04X}" + "aa"
    
    return full_message

def send_and_receive_data(port, baudrate, data, end_marker=b'\xaa', timeout=1):
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

            # Read the reply until the end marker is detected
            full_reply = b''
            while True:
                # Read available bytes
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    full_reply += chunk
                    print(f"Received chunk: {chunk}")
                    # Check if the end marker is in the accumulated reply
                    if full_reply.endswith(end_marker):
                        print("End marker detected.")
                        break

            print(f"Full reply received: {full_reply}")
            return full_reply
    except serial.SerialException as e:
        print(f"Error: {e}")
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

            # Read the reply until the end marker is detected
            # full_reply = b''
            # while True:
            #     # Read available bytes
            #     chunk = ser.read(ser.in_waiting or 1)
            #     if chunk:
            #         full_reply += chunk
            #         print(f"Received chunk: {chunk}")
            #         # Check if the end marker is in the accumulated reply
            #         if full_reply.endswith(end_marker):
            #             print("End marker detected.")
            #             break

            # print(f"Full reply received: {full_reply}")
            # return full_reply
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def parse_reply(reply):
    """
    Parse the received reply and extract numeric values.

    Args:
        reply (bytes): The full reply received from the device.

    Returns:
        dict: Parsed data with 'value1' and 'value2', or None if parsing fails.
    """
    try:
        # Decode the reply to a string (excluding the first and last bytes: 'U' and '\xaa')
        decoded = reply[1:-1].decode("ascii", errors="ignore")

        # Find all numeric patterns in the reply using regex
        numeric_matches = re.findall(r'[+-]?\d+\.\d+E[+-]?\d+', decoded)

        if len(numeric_matches) < 2:
            raise ValueError(f"Not enough numeric data found. Extracted: {numeric_matches}")

        # Convert the first two matches to floats
        value1 = float(numeric_matches[0])
        value2 = float(numeric_matches[1])

        # Return as a dictionary
        return {"voltage": value1, "current": value2}

    except Exception as e:
        print(f"Error parsing reply: {e}")
        return None


def create_directory_with_subdirs(base_directory):
    """
    Creates a directory with the given name and specified subdirectories.
    If the base directory already exists, no new directories are created.
    
    Args:
        base_directory (str): The name of the main directory to create.
        subdirectories (list): A list of subdirectory names to create inside the main directory.
    
    Returns:
        str: A message indicating the result of the operation.
    """
    
    subdirectories = ["jv", "mppt", "jv/plots"]
    
    if os.path.exists(base_directory):
        return f"Directory '{base_directory}' already exists. No subdirectories were created."
    else:
        # Create the base directory
        os.makedirs(base_directory)

        # Create subdirectories
        for subdir in subdirectories:
            path = os.path.join(base_directory, subdir)
            os.makedirs(path)

        return f"Directory '{base_directory}' with subdirectories {subdirectories} created successfully!"
    
def create_txt_file(base_directory, ch):
    file_path = os.path.join(base_directory,"mppt",f"mppt_ch{ch}.txt")
    
    # Ensure the directory exists
    os.makedirs(base_directory, exist_ok=True)
    
    # Create and initialize the file
    with open(file_path, "w") as file:
        # file.write("New Data Acquisition\n")  # Optional header
        # file.write(f"Channel {channel_number}\n")  # Channel-specific header
        file.write("time, voltage, current\n")  # Column headers for clarity
        file.flush()
        file.close()
        
    return file_path

def append_data_to_txt(file_path, formatted_current_time, parsed_fetc):
    with open(file_path, "a") as file:
        # Acquire a new row of data
        # new_row = acquire_data()
        
        # Add a timestamp to each row
        # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the row with the timestamp and data
        row_string = f"{formatted_current_time}, {parsed_fetc['voltage']}, {parsed_fetc['current']}"
        
        # Write the row to the file, followed by a newline
        file.write(row_string + "\n")
        
        # Flush the buffer to ensure data is written to disk
        file.flush()
        
        file.close()


if __name__ == "__main__":
    # Specify COM port and baud rate
    com_port = "COM3"  # Change to your COM port (e.g., 'COM4', '/dev/ttyUSB0' for Linux/Mac)
    baud_rate = 125000   # Common baud rates: 9600, 115200
    
    active_channels = [4, 6, 10, 12]
    
    # Specify Keithley 2600
    keithley = Keithley2600('TCPIP0::169.254.0.1::inst0::INSTR')
    
    # Directory name
    base_directory = "U:/20241105_microMPPT/data/20250121/"
    create_directory_with_subdirs(base_directory)
    
    # Create txt files
    for ch in active_channels:
        create_txt_file(base_directory, ch)

    # Data to send
    message_idn = "550100052a49444e3fA6FBaa" # *IDN?
    
    # hex_message = hex_to_bytes(message)
    
    # ask what the mode is
    message_ask_mode4 = "550100064d4f4445343f7662aa" # *MODE4?
    
    # MPPT
    message_trg = "55 01 00 04 2a 54 52 47 9a d5 aa" # *TRG
    message_fetc4 = "55 01 00 06 46 45 54 43 34 3f cd 1f aa" # FETC4?
    message_fetc6 = "55 01 00 06 46 45 54 43 36 3f ad 1e aa" # FETC6?
    message_fetc10 = "55 01 00 07 46 45 54 43 31 30 3f e5 18 aa" # FETC10?
    message_fetc12 = "55 01 00 07 46 45 54 43 31 32 3f 85 19 aa" # FETC12?
    
    # Temperature sensors
    message_enum = "55 01 00 0a 4f 4e 45 57 3a 45 4e 55 4d 3f 39 34 aa" # ONEW:ENUM?
    message_ids = "55 01 00 09 4f 4e 45 57 3a 49 44 53 3f ec 33 aa"  #ONEW:IDS?
    message_meas = "55 01 00 0a 4f 4e 45 57 3a 4d 45 41 53 3f 78 9e aa" # ONEW:MEAS?
    
    # Bypass
    message_lnbp = "55 01 00 09 4d 4f 44 45 20 4c 4e 42 50 82 cd aa" # MODE LNBP
    message_mode12_byp = "55 01 00 0a 4d 4f 44 45 31 32 20 42 59 50 1e 7d aa" # MODE12 BYP
    message_mode12_lnbp = "55 01 00 0a 4d 4f 44 45 31 32 20 42 59 50 1e 7d aa" # MODE12 LNBP, returning to MPPT
    
    # Initialize
    print(send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_idn)))
    start_time = time.time()
    
    print(send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_ask_mode4))) # ask mode of channel 4
    
    # Make sure all the active channels are in MPPT mode
    for channel in active_channels:
        msg_change_MPPT = "MODE"+str(channel)+" MPPT"
        
        hex_msg_change_MPPT = text_to_umppt(message=msg_change_MPPT)
        send_data(com_port, baud_rate, hex_to_bytes(hex_msg_change_MPPT))
        # print(msg_change_MPPT)
    
    data_fetc4 = np.empty((0, 3))  # Empty array with shape (0, 2)
    data_fetc6 = np.empty((0, 3))
    data_fetc10 = np.empty((0, 3))
    data_fetc12 = np.empty((0, 3))
    
    data_temp = np.empty((0,5))
    
    initiate_T = send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_enum)) # can't do, need the T sensors connected to SCSI

    # Start time for the while true loop
    start_while_time = time.time()
    
    # Entering the loop for MPPT
    while True:
        
        try:
            send_data(com_port, baud_rate, hex_to_bytes(message_trg))
            reply_fetc4 = send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_fetc4))
            reply_fetc6 = send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_fetc6))
            reply_fetc10 = send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_fetc10))
            reply_fetc12 = send_and_receive_data(com_port, baud_rate, hex_to_bytes(message_fetc12))
    
            # Parse replies
            parsed_fetc4 = parse_reply(reply_fetc4)
            parsed_fetc6 = parse_reply(reply_fetc6)
            parsed_fetc10 = parse_reply(reply_fetc10)
            parsed_fetc12 = parse_reply(reply_fetc12)
    
            current_time = time.time() # to be appended
            local_current_time = time.localtime(current_time)
            formatted_current_time = time.strftime("%Y-%m-%d %H:%M:%S", local_current_time)
            undrscr_current_time = time.strftime("%Y_%m_%d_%H_%M_%S", local_current_time)
            
            # Validate parsed results
            if parsed_fetc4 and parsed_fetc6 and parsed_fetc10 and parsed_fetc12:
                
                # Append both values as rows to their respective numpy arrays
                # # Create a new array by appending a row with value1 and value2
                # new_data_fetc4 = np.array([[formatted_current_time, parsed_fetc4["voltage"], parsed_fetc4["current"]]])
                # new_data_fetc6 = np.array([[formatted_current_time, parsed_fetc6["voltage"], parsed_fetc6["current"]]])
                # new_data_fetc10 = np.array([[formatted_current_time, parsed_fetc10["voltage"], parsed_fetc10["current"]]])
                # new_data_fetc12 = np.array([[formatted_current_time, parsed_fetc12["voltage"], parsed_fetc12["current"]]])
    
                # # Concatenate the new data to the existing data arrays
                # data_fetc4 = np.vstack([data_fetc4, new_data_fetc4])
                # data_fetc6 = np.vstack([data_fetc6, new_data_fetc6])
                # data_fetc10 = np.vstack([data_fetc10, new_data_fetc10])
                # data_fetc12 = np.vstack([data_fetc12, new_data_fetc12])
                
                file_path_ch4 = os.path.join(base_directory,"mppt","mppt_ch4.txt")
                file_path_ch6 = os.path.join(base_directory,"mppt","mppt_ch6.txt")
                file_path_ch10 = os.path.join(base_directory,"mppt","mppt_ch10.txt")
                file_path_ch12 = os.path.join(base_directory,"mppt","mppt_ch12.txt")
                
                # Save as txt file
                append_data_to_txt(file_path_ch4, formatted_current_time, parsed_fetc4)
                append_data_to_txt(file_path_ch6, formatted_current_time, parsed_fetc6)
                append_data_to_txt(file_path_ch10, formatted_current_time, parsed_fetc10)
                append_data_to_txt(file_path_ch12, formatted_current_time, parsed_fetc12)
    
                # Log parsed results
                print(f"FETC4: {parsed_fetc4}")
                print(f"FETC6: {parsed_fetc6}")
                print(f"FETC10: {parsed_fetc10}")
                print(f"FETC12: {parsed_fetc12}")
            else:
                print("Warning: Parsing failed for one or more messages. Skipping this iteration.")
            
            # Do an IV scan for all channels if it goes beyond certain minutes
            if (current_time-start_while_time) >= 180: # 1 minute = 60 s
                # ch = 4
                for ch in active_channels:
                
                    bypass_ch_msg = "MODE"+str(ch)+" BYP"
                    lnbp_ch_msg = "MODE"+str(ch)+" LNBP"
                    
                    hex_msg_bypass_ch = text_to_umppt(message=bypass_ch_msg)
                    hex_msg_lnbp_ch = text_to_umppt(message=lnbp_ch_msg)
                    
                    initiate_IV_scan = send_data(com_port, baud_rate, hex_to_bytes(message_lnbp))
                    # bypass_ch12 = send_data(com_port, baud_rate, hex_to_bytes(message_mode12_byp))
                    bypass_ch = send_data(com_port, baud_rate, hex_to_bytes(hex_msg_bypass_ch))
                    
                    # create ResultTable with two columns
                    rt = ResultTable(
                        column_titles=['Voltage', 'Current'],
                        units=['V', 'A'],
                        params={'recorded': time.asctime(), 'sweep_type': 'iv'},
                    )
                    
                    # measure currents bwd
                    for v in range(220,0,-1):
                        keithley.apply_voltage(keithley.smua, v/100)
                        time.sleep(0.01)
                        i = keithley.smua.measure.i()
                        rt.append_row([v/100, i])
                        
                    iv_data_bwd = rt.data
                    
                    # measure some currents fwd
                    rt = ResultTable(
                        column_titles=['Voltage', 'Current'],
                        units=['V', 'A'],
                        params={'recorded': time.asctime(), 'sweep_type': 'iv'},
                    )
                    
                    for v in range(0, 220):
                        keithley.apply_voltage(keithley.smua, v/100)
                        time.sleep(0.01)
                        i = keithley.smua.measure.i()
                        rt.append_row([v/100, i])
                        
                    iv_data_fwd = rt.data
                    
                    plt.plot(iv_data_bwd[:,0], iv_data_bwd[:,1], label='bwd')
                    plt.plot(iv_data_fwd[:,0], iv_data_fwd[:,1], label='fwd')
                    plt.title("CH"+str(ch)+" "+formatted_current_time)
                    plt.grid()
                    plt.legend(loc="upper left")
                    
                    save_path = os.path.join(base_directory, "jv", "plots")
                    os.makedirs(save_path, exist_ok=True)  # Creates the directories if they don't exist
                    file_name = f"jv_CH{str(ch)}_{undrscr_current_time}"
                    plt.savefig(os.path.join(save_path, file_name+".png"))
                    
                    plt.close()
                    
                    # Save files as txt 
                    np.savetxt(os.path.join(base_directory, "jv", file_name+"_bwd.txt"), iv_data_bwd, delimiter=",", header="voltage, current", comments="") #fmt="%.6f",
                    np.savetxt(os.path.join(base_directory, "jv", file_name+"_fwd.txt"), iv_data_fwd, delimiter=",", header="voltage, current", comments="")
                        
                    # Return back to MPPT
                    return_ch = send_data(com_port, baud_rate, hex_to_bytes(hex_msg_lnbp_ch))
                    msg_change_MPPT = "MODE"+str(ch)+" MPPT"
                    hex_msg_change_MPPT = text_to_umppt(message=msg_change_MPPT)
                    send_data(com_port, baud_rate, hex_to_bytes(hex_msg_change_MPPT))
                
                    
                # Reset start_while_time
                start_while_time = time.time()
    
                   
            
           
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("Exiting loop.")
            break
    
    # # Send the data
    # reply = send_and_receive_data(com_port, baud_rate, hex_message)
    
    # print(reply)
    