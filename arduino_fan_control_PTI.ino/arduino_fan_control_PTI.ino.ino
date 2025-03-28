#include <OneWire.h>
#include <DallasTemperature.h>

// Data wire for temperature sensors is connected to pin 2
#define ONE_WIRE_BUS 2 //2

// PWM output pins for the two computer fans
#define FAN_PIN1 5
#define FAN_PIN2 6

// Create oneWire instance to communicate with 1-Wire devices
OneWire oneWire(ONE_WIRE_BUS);
// Pass oneWire reference to DallasTemperature.
DallasTemperature sensors(&oneWire);

// A type definition for sensor addresses (8 bytes each)
typedef uint8_t DeviceAddress[8];

void setup() {
  Serial.begin(9600);
  while (!Serial) ;  // Wait for serial port to connect if needed
  sensors.begin();
  
  // Set up the fan control pins as outputs
  pinMode(FAN_PIN1, OUTPUT);
  pinMode(FAN_PIN2, OUTPUT);
  
  // Set default fan speed to 50% (0-100% mapped to 0-255 PWM)
  int defaultSpeed = 50;
  int pwmValue = map(defaultSpeed, 0, 100, 0, 255);
  analogWrite(FAN_PIN1, pwmValue);
  analogWrite(FAN_PIN2, pwmValue);
  
  // Print command help
  Serial.println("Temperature sensor and fan control interface ready.");
  Serial.println("Valid commands:");
  Serial.println("  COUNT             - Get the number of sensors");
  Serial.println("  LIST              - Get all sensor IDs (ends with a final message)");
  Serial.println("  GET <sensor_id>   - Get temperature from a specific sensor");
  Serial.println("  ALL               - Get temperature from all sensors");
  Serial.println("  FAN1 <speed>      - Set fan 1 speed (0-100%) on pin D5");
  Serial.println("  FAN2 <speed>      - Set fan 2 speed (0-100%) on pin D6");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.length() > 0) {
      processCommand(command);
    }
  }
}

// Process commands received via Serial
void processCommand(String cmd) {
  // Convert to uppercase to simplify command comparison
  String upperCmd = cmd;
  upperCmd.toUpperCase();

  if (upperCmd.startsWith("COUNT")) {
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
  else if (upperCmd.startsWith("FAN1")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex == -1) {
      Serial.println("Error: FAN1 command requires a speed value.");
      return;
    }
    String speedStr = cmd.substring(spaceIndex + 1);
    speedStr.trim();
    int speed = speedStr.toInt();
    if (speed < 0 || speed > 100) {
      Serial.println("Error: Fan speed must be between 0 and 100.");
      return;
    }
    int pwmValue = map(speed, 0, 100, 0, 255);
    analogWrite(FAN_PIN1, pwmValue);
    Serial.print("Fan 1 speed updated to ");
    Serial.print(speed);
    Serial.println("%.");
  }
  else if (upperCmd.startsWith("FAN2")) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex == -1) {
      Serial.println("Error: FAN2 command requires a speed value.");
      return;
    }
    String speedStr = cmd.substring(spaceIndex + 1);
    speedStr.trim();
    int speed = speedStr.toInt();
    if (speed < 0 || speed > 100) {
      Serial.println("Error: Fan speed must be between 0 and 100.");
      return;
    }
    int pwmValue = map(speed, 0, 100, 0, 255);
    analogWrite(FAN_PIN2, pwmValue);
    Serial.print("Fan 2 speed updated to ");
    Serial.print(speed);
    Serial.println("%.");
  }
  else {
    Serial.println("Unknown command. Valid commands are: COUNT, LIST, GET <sensor_id>, ALL, FAN1 <speed>, FAN2 <speed>");
  }
}

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