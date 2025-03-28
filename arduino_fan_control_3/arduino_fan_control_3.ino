#include <OneWire.h>
#include <DallasTemperature.h>

// Pin assignments
#define ONE_WIRE_BUS 2  // Temperature sensor (DQ pin)
#define BACK_FAN 11
#define FRONT_FAN1 5
#define FRONT_FAN2 6
#define INSIDE_BACK 9
#define INSIDE_FRONT 10

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

typedef uint8_t DeviceAddress[8];

void setup() {
    Serial.begin(9600);
    while (!Serial) ;  // Wait for serial port to connect if needed
    sensors.begin();
    
    // Set up the fan control pins as outputs
    pinMode(BACK_FAN, OUTPUT);
    pinMode(FRONT_FAN1, OUTPUT);
    pinMode(FRONT_FAN2, OUTPUT);
    pinMode(INSIDE_BACK, OUTPUT);
    pinMode(INSIDE_FRONT, OUTPUT);
    
    analogWrite(BACK_FAN, 60);
    analogWrite(FRONT_FAN1, 60);
    analogWrite(FRONT_FAN2, 60);
    analogWrite(INSIDE_BACK, 60);
    analogWrite(INSIDE_FRONT, 60);
    
    Serial.println("==== FAN CONTROL SYSTEM ====");
    Serial.println("Commands:");
    Serial.println("BACK <speed>        - Set back fan speed (0-100)");
    Serial.println("FRONT1 <speed>      - Set front fan 1 speed (0-100)");
    Serial.println("FRONT2 <speed>      - Set front fan 2 speed (0-100)");
    Serial.println("INSIDE_BACK <speed> - Set inside back fan speed (0-100)");
    Serial.println("INSIDE_FRONT <speed> - Set inside front fan speed (0-100)");
    Serial.println("ALL                 - Show all temperatures");
    Serial.println("LIST                - List temperature sensor addresses");
    Serial.println("COUNT               - Get the number of sensors");
    Serial.println("GET <sensor_id>     - Get temperature from a specific sensor");
    Serial.println("==========================");
}

void loop() {
    if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();

        String upperCmd = cmd;
        upperCmd.toUpperCase();
        
        if (upperCmd.startsWith("BACK") || upperCmd.startsWith("FRONT") || upperCmd.startsWith("INSIDE")) {
            handleFanCommand(cmd);
        // } else if (upperCmd.startsWith("ALL")) {
        //     printAllTemperatures();
        // } else if (command == "LIST") {
        //     listSensors();
        // } else if (command == "COUNT") {
        //     countSensors();
        // } else if (command.startsWith("GET")) {
        //     handleGetTemperature(command);
        // }
        } if (upperCmd.startsWith("COUNT")) {
          countSensors();
        }
        else if (upperCmd.startsWith("LIST")) {
          listSensorIDs();
        }
        else if (upperCmd.startsWith("GET")) {
          int spaceIndex = upperCmd.indexOf(' ');
          if (spaceIndex == -1) {
            Serial.println("Error: GET command requires a sensor id parameter.");
            return;
          }
          String sensorIdStr = cmd.substring(spaceIndex + 1); // Use original case for sensor ID
          sensorIdStr.trim();
          getSensorTemp(sensorIdStr);
        }
        else if (upperCmd.startsWith("ALL")) {
          getAllSensorsTemp();
        }
    }
}

void handleFanCommand(String command) {
    int speed = command.substring(command.lastIndexOf(" ") + 1).toInt();
    speed = map(constrain(speed, 0, 100), 0, 100, 0, 255);

    if (command.startsWith("BACK")) {
        if (speed < 26) {
          Serial.println("BACK FAN SPEED CAN'T BE < 11%!");
          analogWrite(BACK_FAN,26);
        }
        else {
          analogWrite(BACK_FAN, speed);
          Serial.println("BACK fan set to " + String(speed));
        }
        
    } else if (command.startsWith("FRONT1")) {
        analogWrite(FRONT_FAN1, speed);
        Serial.println("FRONT1 fan set to " + String(speed));
    } else if (command.startsWith("FRONT2")) {
        analogWrite(FRONT_FAN2, speed);
        Serial.println("FRONT2 fan set to " + String(speed));
    } else if (command.startsWith("INSIDE_BACK")) {
        analogWrite(INSIDE_BACK, speed);
        Serial.println("INSIDE_BACK fan set to " + String(speed));
    } else if (command.startsWith("INSIDE_FRONT")) {
        analogWrite(INSIDE_FRONT, speed);
        Serial.println("INSIDE_FRONT fan set to " + String(speed));
    }
}

void printAllTemperatures() {
    int deviceCount = sensors.getDeviceCount();
  sensors.requestTemperatures();
  for (int i = 0; i < deviceCount; i++) {
    DeviceAddress addr;
    if (sensors.getAddress(addr, i)) {
      float tempC = sensors.getTempC(addr);
      Serial.print("Sensor ");
      Serial.print(formatAddress(addr));
      Serial.print(": ");
      if (tempC == DEVICE_DISCONNECTED_C) {
        Serial.println("Error: Could not read temperature data.");
      } else {
        Serial.print(tempC);
        Serial.println(" C");
      }
    } else {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.println(": Error reading address");
    }
  }
  // Final message indicating that the temperature of all sensor IDs have been transmitted.
  Serial.println("Transmission finished");
}
//     sensors.requestTemperatures();
//     int sensorCount = sensors.getDeviceCount();
//     for (int i = 0; i < sensorCount; i++) {
//         Serial.print("Sensor "); Serial.print(i);
//         Serial.print(": ");
//         Serial.print(sensors.getTempCByIndex(i));
//         Serial.println(" C");
//     }
//     Serial.println("Transmission finished");
// }

// void listSensors() {
//     int sensorCount = sensors.getDeviceCount();
//     for (int i = 0; i < sensorCount; i++) {
//         DeviceAddress address;
//         if (sensors.getAddress(address, i)) {
//             Serial.print("Sensor "); Serial.print(i);
//             Serial.print(": ");
//             for (uint8_t j = 0; j < 8; j++) {
//                 if (address[j] < 16) Serial.print("0");
//                 Serial.print(address[j], HEX);
//             }
//             Serial.println();
//         }
//     }
// }

// void countSensors() {
//     int deviceCount = sensors.getDeviceCount();
//     Serial.print("Number of sensors: ");
//     Serial.println(deviceCount);
// }

// void handleGetTemperature(String command) {
//     String sensorIdStr = command.substring(4);
//     sensorIdStr.trim();
    
//     if (sensorIdStr.length() != 16) {
//         Serial.println("Invalid sensor ID length. Expected 16 hex characters.");
//         return;
//     }
    
//     DeviceAddress addr;
//     for (int i = 0; i < 8; i++) {
//         String byteStr = sensorIdStr.substring(i * 2, i * 2 + 2);
//         byte b = (byte) strtol(byteStr.c_str(), NULL, 16);
//         addr[i] = b;
//     }
    
//     sensors.requestTemperatures();
//     float tempC = sensors.getTempC(addr);
    
//     if (tempC == DEVICE_DISCONNECTED_C) {
//         Serial.println("Error: Could not read temperature data.");
//     } else {
//         Serial.print("Sensor ");
//         Serial.print(sensorIdStr);
//         Serial.print(" temperature: ");
//         Serial.print(tempC);
//         Serial.println(" C");
//     }
// }

// Function to return the number of sensors
void countSensors() {
  int deviceCount = sensors.getDeviceCount();
  Serial.print("Number of sensors: ");
  Serial.println(deviceCount);
}

// Function to list all sensor IDs followed by a final transmission message
void listSensorIDs() {
  int deviceCount = sensors.getDeviceCount();
  for (int i = 0; i < deviceCount; i++) {
    DeviceAddress addr;
    if (sensors.getAddress(addr, i)) {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(formatAddress(addr));
    } else {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.println(": Error reading address");
    }
  }
  // Final message indicating that the sensor IDs have been transmitted.
  Serial.println("Transmission finished");
}

// Get the temperature from a sensor with the specified ID string (16 hex digits)
void getSensorTemp(String sensorIdStr) {
  if (sensorIdStr.length() != 16) {
    Serial.println("Invalid sensor ID length. Expected 16 hex characters.");
    return;
  }
  
  DeviceAddress addr;
  for (int i = 0; i < 8; i++) {
    String byteStr = sensorIdStr.substring(i * 2, i * 2 + 2);
    byte b = (byte) strtol(byteStr.c_str(), NULL, 16);
    addr[i] = b;
  }
  
  // Verify if the sensor exists among those detected
  int deviceCount = sensors.getDeviceCount();
  bool found = false;
  for (int i = 0; i < deviceCount; i++) {
    DeviceAddress tempAddr;
    if (sensors.getAddress(tempAddr, i)) {
      if (compareAddress(tempAddr, addr)) {
        found = true;
        break;
      }
    }
  }
  
  if (!found) {
    Serial.println("Sensor not found.");
    return;
  }
  
  // Request temperatures from all sensors (a minimal delay may be required for conversion)
  sensors.requestTemperatures();
  float tempC = sensors.getTempC(addr);
  
  if (tempC == DEVICE_DISCONNECTED_C) {
    Serial.println("Error: Could not read temperature data.");
  } else {
    Serial.print("Sensor ");
    Serial.print(sensorIdStr);
    Serial.print(" temperature: ");
    Serial.print(tempC);
    Serial.println(" C");
  }
}

// Get the temperature from all sensors
void getAllSensorsTemp() {
  int deviceCount = sensors.getDeviceCount();
  sensors.requestTemperatures();
  for (int i = 0; i < deviceCount; i++) {
    DeviceAddress addr;
    if (sensors.getAddress(addr, i)) {
      float tempC = sensors.getTempC(addr);
      Serial.print("Sensor ");
      Serial.print(formatAddress(addr));
      Serial.print(": ");
      if (tempC == DEVICE_DISCONNECTED_C) {
        Serial.println("Error: Could not read temperature data.");
      } else {
        Serial.print(tempC);
        Serial.println(" C");
      }
    } else {
      Serial.print("Sensor ");
      Serial.print(i);
      Serial.println(": Error reading address");
    }
  }
  // Final message indicating that the temperature of all sensor IDs have been transmitted.
  Serial.println("Transmission finished");
}

// Helper: Format a sensor address as a 16-digit uppercase hex string
String formatAddress(DeviceAddress deviceAddress) {
  String addrStr = "";
  for (uint8_t i = 0; i < 8; i++) {
    if (deviceAddress[i] < 16) addrStr += "0";
    addrStr += String(deviceAddress[i], HEX);
  }
  addrStr.toUpperCase();
  return addrStr;
}

// Helper: Compare two sensor addresses
bool compareAddress(DeviceAddress addr1, DeviceAddress addr2) {
  for (int i = 0; i < 8; i++) {
    if (addr1[i] != addr2[i]) return false;
  }
  return true;
}
