#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_BUS 5

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
    Serial.begin(9600);
    Serial.println("Starting Dallas Temperature Sensor Test...");

    sensors.begin();
    Serial.print("Sensors detected: ");
    Serial.println(sensors.getDeviceCount());

    if (sensors.getDeviceCount() == 0) {
        Serial.println("No sensors found! Check wiring.");
    }
}

void loop() {
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    if (tempC == DEVICE_DISCONNECTED_C) {
        Serial.println("Error: No sensor detected!");
    } else {
        Serial.print("Temperature: ");
        Serial.print(tempC);
        Serial.println(" C");
    }
    delay(2000);
}