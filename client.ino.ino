#include <WiFi.h>
#include "FlexLibrary.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define VCC 5.0              // Voltage supplied to the flex sensors
#define R_DIV 10000.0        // Resistance of the resistor in series with the flex sensors
#define flatResistance 32500.0  // Resistance of the flex sensors when flat
#define bendResistance 76000.0  // Resistance of the flex sensors when bent

Flex flex[5] = {Flex(36), Flex(39), Flex(34), Flex(35), Flex(32)}; // Analog pins the flex sensors are on

const char* ssid = "Gogetyourwifi";
const char* password = "Columbus@971-01";
const char* host = "192.168.1.30";  // Raspberry Pi IP address
const int port = 4444;

WiFiClient client;
Adafruit_MPU6050 mpu;

void setup() {
    Serial.begin(9600);
    while (!Serial) {
        delay(10); // Wait until the serial console opens
    }

    Serial.println("Adafruit MPU6050 test!");

    // Initialize MPU6050
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) delay(10);
    }
    Serial.println("MPU6050 Found!");

    // Configure MPU6050
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting...");
    }
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Try connecting to the server
    if (client.connect(host, port)) {
        Serial.println("Connected to server");
    } else {
        Serial.println("Failed to connect to server");
    }
}

void loop() {
    float angles[5];
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Process flex sensor data
    for (int i = 0; i < 5; i++) {
        flex[i].updateVal();
        float Vflex = flex[i].getSensorValue() * VCC / 4095.0;
        float Rflex = R_DIV * (VCC / Vflex - 1.0);
        angles[i] = constrain(map(Rflex, flatResistance, bendResistance, 0, 90), 0, 90);
    }

    // Prepare data array
    byte data[sizeof(float) * 11];
    memcpy(data, angles, sizeof(float) * 5);
    memcpy(data + sizeof(float) * 5, &a.acceleration.x, sizeof(float));
    memcpy(data + sizeof(float) * 6, &a.acceleration.y, sizeof(float));
    memcpy(data + sizeof(float) * 7, &a.acceleration.z, sizeof(float));
    memcpy(data + sizeof(float) * 8, &g.gyro.x, sizeof(float));
    memcpy(data + sizeof(float) * 9, &g.gyro.y, sizeof(float));
    memcpy(data + sizeof(float) * 10, &g.gyro.z, sizeof(float));

    // Debug print accelerometer, gyroscope, and angles
    // Print flex sensor values
    for (int i = 0; i < 5; i++) {
        Serial.print(angles[i]);
        Serial.print(",");
    }

    // Print gyroscope values
    Serial.print(g.gyro.x);
    Serial.print(",");
    Serial.print(g.gyro.y);
    Serial.print(",");
    Serial.print(g.gyro.z);
    Serial.print(",");

    // Print accelerometer values
    Serial.print(a.acceleration.x);
    Serial.print(",");
    Serial.print(a.acceleration.y);
    Serial.print(",");
    Serial.println(a.acceleration.z);  // End the line

    // Send data to server (client communication)
    send_data(data);

    delay(300);  // Adjust delay as needed
}

void send_data(byte data[]) {
    // Send data to server over WiFi
    if (client.connected()) {
        client.write(data, sizeof(data));
        Serial.println("Data sent to server.");
    } else {
        Serial.println("Connection to server lost. Reconnecting...");
        if (client.connect(host, port)) {
            Serial.println("Reconnected to server");
            client.write(data, sizeof(data));
        } else {
            Serial.println("Reconnection failed");
        }
    }
}
