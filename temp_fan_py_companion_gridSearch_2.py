# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:53:13 2025

@author: peter.tillmann
"""

#!/usr/bin/env python3
import serial
import time
import os
import re

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
    time.sleep(0.2)  # Allow Arduino to process

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
        # match = re.search(r"([-+]?\d*\.\d+|\d+) C", line)
        match = re.search(r"([-+]?\d*\.\d+|\d+)", line)  # More flexible regex
        if match:
            return match.group(1)  # Return only the numeric temperature value

    print(f"WARNING: No valid temperature from {sensor_id}, returning 'NaN'")
    return "NaN"  # Return NaN if no valid temperature is found

def initialize_T_sensors(ser, BASE_DIR, save_as_channels=True):
    sensor_ids = list_sensor_ids(ser)
    if not sensor_ids:
        print("No sensors detected.")
        return
    
    dict_cells = {
        '281254A30F0000DF':'A1', # A
        '289658A30F0000C8':'A2',
        '286E52A30F0000A6':'A3',
        '286B62A30F0000C9':'A4',
        '288D4CA30F000008':'A5',
        '288A5BA30F0000A0':'A6',
        '281652A30F00009F':'B1', # B
        '28B97AA30F000021':'B2',
        '2827B4A20F00008C':'B3',
        '282064A30F0000D8':'B4',
        '284B2CA30F000034':'B5',
        '289255A30F0000F8':'B6',
        '28665EA30F000026':'C1', # C
        '281A5CA30F000040':'C2',
        '28165AA30F0000A1':'C3',
        '285237A30F0000F5':'C4',
        '28005DA30F000019':'C5',
        '28E433A30F00005F':'C6',
        '28F24BA30F0000E5':'D1', # D
        '28716AA30F000063':'D2',
        '281A57A30F000030':'D3',
        '2885BBA20F0000D1':'D4',
        '28CB5FA30F0000FD':'D5',
        '28FA64A30F0000D3':'D6',
        }
    
    dict_channels = {
        '281254A30F0000DF':24, # A
        '289658A30F0000C8':23,
        '286E52A30F0000A6':22,
        '286B62A30F0000C9':21,
        '288D4CA30F000008':20,
        '288A5BA30F0000A0':19,
        '281652A30F00009F':18, # B
        '28B97AA30F000021':17,
        '2827B4A20F00008C':16,
        '282064A30F0000D8':15,
        '284B2CA30F000034':14,
        '289255A30F0000F8':13,
        '28665EA30F000026':6, # C
        '281A5CA30F000040':5,
        '28165AA30F0000A1':4,
        '285237A30F0000F5':3,
        '28005DA30F000019':2,
        '28E433A30F00005F':1,
        '28F24BA30F0000E5':12, # D
        '28716AA30F000063':11,
        '281A57A30F000030':10,
        '2885BBA20F0000D1':9,
        '28CB5FA30F0000FD':8,
        '28FA64A30F0000D3':7,
        }

    # filename = "temperature_log_2.txt"
    # os.makedirs(os.path.dirname)
    filename = os.path.join(BASE_DIR, "temperature", time.strftime("%Y_%m_%d_%H_%M_%S") + "_temperature.txt")
    
    
    # Ensure the directory exists before trying to write the file
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # ✅ Fix added
    
    # Generate the header row with channel names (CH1, CH2, etc.)
    channel_names = [f"CH{dict_channels[sensor_id]}" for sensor_id in sensor_ids]
    # channel_names = sensor_ids
    
    if save_as_channels == False:
        file_exists = os.path.exists(filename)
        with open(filename, "a") as file:
            if not file_exists:
                # file.write("time," + ",".join(sensor_ids) + "\n")  
                file.write("time,minutes,front1_speed,front2_speed,back_speed,inside_back_speed,inside_front_speed," + ",".join(channel_names) + "\n")
                
    if save_as_channels == True:
        file_exists = os.path.exists(filename)
        with open(filename, "a") as file:
            if not file_exists:
                # file.write("time," + ",".join(channel_names) + "\n")
                file.write("time,minutes,front1_speed,front2_speed,back_speed,inside_back_speed,inside_front_speed," + ",".join(channel_names) + "\n")
    
    return sensor_ids, filename

def record_temperatures(ser, BASE_DIR, save_as_channels):
    sensor_ids, filename = initialize_T_sensors(ser, BASE_DIR, save_as_channels)
    
    while True:
        try:
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
            with open(filename, "a") as file:
                file.write(row + "\n")

            print(f"Recorded at {timestamp}: {row}")  
            time.sleep(1)  # Reduce sleep to 1s for faster updates

        except KeyboardInterrupt:
            ser.close()
            print("\nStopping temperature logging.")
            break
    

def all_sensors_temp(ser):
    response = send_command_T(ser, "ALL", expect_multiple_lines=True)
    for line in response:
        print(line)

def set_front1_speed(ser, speed):
    response = send_command_T(ser, f"FRONT1 {speed}")
    for line in response:
        print(line)

def set_front2_speed(ser, speed):
    response = send_command_T(ser, f"FRONT2 {speed}")
    for line in response:
        print(line)
        
def set_back_speed(ser, speed):
    response = send_command_T(ser, f"BACK {speed}")
    for line in response:
        print(line)

def set_inside_back_speed(ser, speed):
    response = send_command_T(ser, f"INSIDE_BACK {speed}")
    for line in response:
        print(line)

def set_inside_front_speed(ser, speed):
    response = send_command_T(ser, f"INSIDE_FRONT {speed}")
    for line in response:
        print(line)

# def main():
#     # port = input("Enter the serial port (e.g., COM3 or /dev/ttyACM0): ").strip()
#     port = "COM6"
#     baud_rate = 9600
#     try:
#         ser = serial.Serial(port, baud_rate, timeout=1)
#     except serial.SerialException as e:
#         print(f"Error opening serial port {port}: {e}")
#         return

#     # Wait for the Arduino to reset after serial connection
#     time.sleep(2)
#     print("Connected to Arduino.")

#     while True:
#         try:
#             sensor_ids = list_sensor_ids(ser)
            
#             time.sleep(3)

#     # while True:
#     #     print("\nMenu:")
#     #     print("1. Count sensors")
#     #     print("2. List sensor IDs")
#     #     print("3. Get sensor temperature")
#     #     print("4. Get all sensors temperature")
#     #     print("5. Set Fan 1 speed")
#     #     print("6. Set Fan 2 speed")
#     #     print("7. Exit")
#     #     choice = input("Enter your choice: ").strip()
#     #     if choice == "1":
#     #         count_sensors(ser)
#     #     elif choice == "2":
#     #         list_sensor_ids(ser)
#     #     elif choice == "3":
#     #         sensor_id = input("Enter sensor ID (16 hex digits): ").strip()
#     #         get_sensor_temp(ser, sensor_id)
#     #     elif choice == "4":
#     #         all_sensors_temp(ser)
#     #     elif choice == "5":
#     #         speed = input("Enter Fan 1 speed (0-100): ").strip()
#     #         set_fan1_speed(ser, speed)
#     #     elif choice == "6":
#     #         speed = input("Enter Fan 2 speed (0-100): ").strip()
#     #         set_fan2_speed(ser, speed)
#     #     elif choice == "7":
#     #         break
#     #     else:
#     #         print("Invalid choice. Please try again.")

            
#         except KeyboardInterrupt:
#             ser.close()
#             print("Connection closed to temperature sensor and fan.")
#             break

def set_fan_speed(ser, front1_speed, front2_speed, back_speed, inside_back_speed, inside_front_speed):
    set_front1_speed(ser, front1_speed)
    set_front2_speed(ser, front2_speed)
    set_back_speed(ser, back_speed)
    set_inside_back_speed(ser, inside_back_speed)
    set_inside_front_speed(ser, inside_front_speed)
    print(f"Fan speeds set: front1_speed={front1_speed}, front2_speed={front1_speed}, back_speed={back_speed}, inside_back_speed={inside_back_speed}, inside_front_speed = {inside_front_speed}")

def grid_search_fan_speeds(ser, BASE_DIR, save_as_channels=True):
    sensor_ids, filename = initialize_T_sensors(ser, BASE_DIR, save_as_channels)
    fan_speeds = [100]  # Range of speeds
    back_fan_speeds = [50]
    front1_speed = 100
    front2_speed = 100
    
    file_exists = os.path.exists(filename)
    if not file_exists:
        with open(filename, "w") as file:
            # file.write("time,minutes,fan1_speed,fan2_speed," + ",".join(sensor_ids) + "\n")
            file.write("time,minutes,front1_speed,front2_speed,back_speed,inside_back_speed,inside_front_speed," + ",".join(sensor_ids) + "\n")
    
    for back_speed in back_fan_speeds:
        
        for inside_front_speed in fan_speeds:
            for inside_back_speed in fan_speeds:
                # set_fan_speed(ser, fan1_speed, fan2_speed)
                set_fan_speed(ser, front1_speed, front2_speed, back_speed, inside_back_speed, inside_front_speed)
                start_time = time.time()
                
                while time.time() - start_time < 60:  # 20 minutes
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    elapsed_minutes = int((time.time() - start_time) / 60)
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
                        temperatures.append(matched_temp if matched_temp else "NaN")
                    
                    row = f"{timestamp},{elapsed_minutes},{front1_speed},{front2_speed},{back_speed},{inside_back_speed},{inside_front_speed}," + ",".join(temperatures)
                    with open(filename, "a") as file:
                        file.write(row + "\n")
                    
                    print(f"Recorded at {timestamp}: {row}")
                    time.sleep(10)  # Log every 10 seconds

if __name__ == '__main__':
    
    
    port = "COM7"  
    baud_rate = 9600  
    BASE_DIR = "U:/20241105_microMPPT/data/20250307_test_hardware/"
    try:
        ser = serial.Serial(port, baud_rate, timeout=10)
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        exit()

    time.sleep(2)  
    print("Connected to Arduino.")
    # record_temperatures(ser, save_as_channels = True)
    
    # # Set the fan
    # set_fan1_speed(ser, 60)
    # set_fan2_speed(ser, 100)
    # # Record temperature loop
    # record_temperatures(ser, BASE_DIR, save_as_channels=True)
    
    # Grid search fan speeds
    grid_search_fan_speeds(ser, BASE_DIR, save_as_channels=True)
    
    
    # # main()
    
    # # port = input("Enter the serial port (e.g., COM3 or /dev/ttyACM0): ").strip()
    # port = "COM6"
    # baud_rate = 9600
    # try:
    #     ser = serial.Serial(port, baud_rate, timeout=5)
    # except serial.SerialException as e:
    #     print(f"Error opening serial port {port}: {e}")
    #     # return

    # # Wait for the Arduino to reset after serial connection
    # time.sleep(2)
    # print("Connected to Arduino.")
    
    # sensor_ids = list_sensor_ids(ser)

    # while True:
    #     try:
            
    #         set_fan1_speed(ser, 100)
    #         # set_fan2_speed(ser, 100)

    # # while True:
    # #     print("\nMenu:")
    # #     print("1. Count sensors")
    # #     print("2. List sensor IDs")
    # #     print("3. Get sensor temperature")
    # #     print("4. Get all sensors temperature")
    # #     print("5. Set Fan 1 speed")
    # #     print("6. Set Fan 2 speed")
    # #     print("7. Exit")
    # #     choice = input("Enter your choice: ").strip()
    # #     if choice == "1":
    # #         count_sensors(ser)
    # #     elif choice == "2":
    # #         list_sensor_ids(ser)
    # #     elif choice == "3":
    # #         sensor_id = input("Enter sensor ID (16 hex digits): ").strip()
    # #         get_sensor_temp(ser, sensor_id)
    # #     elif choice == "4":
    # #         all_sensors_temp(ser)
    # #     elif choice == "5":
    # #         speed = input("Enter Fan 1 speed (0-100): ").strip()
    # #         set_fan1_speed(ser, speed)
    # #     elif choice == "6":
    # #         speed = input("Enter Fan 2 speed (0-100): ").strip()
    # #         set_fan2_speed(ser, speed)
    # #     elif choice == "7":
    # #         break
    # #     else:
    # #         print("Invalid choice. Please try again.")

    #     except KeyboardInterrupt:
    #         ser.close()
    #         print("Connection closed to temperature sensor and fan.")
    #         break

#%%

# ser.timeout = 5  # Increase timeout to 5 seconds
# print("Sending LIST command to Arduino...")
# ser.write(b'LIST\n')  # Adjust command format if needed

# response = ser.readline().strip()
# if not response:
#     print("⚠️ No response received from Arduino. Check connection.")
# else:
#     print("Received response:", response)