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

void setup() {
    Serial.begin(9600);
    sensors.begin();
    
    pinMode(BACK_FAN, OUTPUT);
    pinMode(FRONT_FAN1, OUTPUT);
    pinMode(FRONT_FAN2, OUTPUT);
    pinMode(INSIDE_BACK, OUTPUT);
    pinMode(INSIDE_FRONT, OUTPUT);
    
    analogWrite(BACK_FAN, 0);
    analogWrite(FRONT_FAN1, 0);
    analogWrite(FRONT_FAN2, 0);
    analogWrite(INSIDE_BACK, 0);
    analogWrite(INSIDE_FRONT, 0);
    
    // Display command list on startup
    Serial.println("==== FAN CONTROL SYSTEM ====");
    Serial.println("Commands:");
    Serial.println("BACK <speed>        - Set back fan speed (0-100)");
    Serial.println("FRONT1 <speed>      - Set front fan 1 speed (0-100)");
    Serial.println("FRONT2 <speed>      - Set front fan 2 speed (0-100)");
    Serial.println("INSIDE_BACK <speed> - Set inside back fan speed (0-100)");
    Serial.println("INSIDE_FRONT <speed> - Set inside front fan speed (0-100)");
    Serial.println("ALL                 - Show all temperatures");
    Serial.println("LIST                - List temperature sensor addresses");
    Serial.println("==========================");
}

void loop() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        if (command.startsWith("BACK") || command.startsWith("FRONT") || command.startsWith("INSIDE")) {
            handleFanCommand(command);
        } else if (command == "ALL") {
            printAllTemperatures();
        } else if (command == "LIST") {
            listSensors();
        }
    }
}

void handleFanCommand(String command) {
    int speed = command.substring(command.lastIndexOf(" ") + 1).toInt();
    speed = map(constrain(speed, 0, 100), 0, 100, 0, 255);  // Convert 0-100% to 0-255 PWM

    if (command.startsWith("BACK")) {
        analogWrite(BACK_FAN, speed);
        Serial.println("BACK fan set to " + String(speed));
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
    sensors.requestTemperatures();
    int sensorCount = sensors.getDeviceCount();
    for (int i = 0; i < sensorCount; i++) {
        Serial.print("Sensor "); Serial.print(i);
        Serial.print(": ");
        Serial.print(sensors.getTempCByIndex(i));
        Serial.println(" C");
    }
    Serial.println("Transmission finished");
}

void listSensors() {
    int sensorCount = sensors.getDeviceCount();
    for (int i = 0; i < sensorCount; i++) {
        DeviceAddress address;
        if (sensors.getAddress(address, i)) {
            Serial.print("Sensor "); Serial.print(i);
            Serial.print(": ");
            for (uint8_t j = 0; j < 8; j++) {
                if (address[j] < 16) Serial.print("0");
                Serial.print(address[j], HEX);
            }
            Serial.println();
        }
    }
}
