#include <Arduino.h>

#define LED_BUILTIN 2

unsigned long previousMillis = 0;
const long interval = 500;
bool ledState = false;

void setup()
{
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);
    delay(3000);
    Serial.println("=== SETUP START ===");
    Serial.println("Hello from ESP32!");
    Serial.println("=== SETUP DONE ===");
}

void loop()
{
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= interval)
    {
        previousMillis = currentMillis;
        ledState = !ledState;
        digitalWrite(LED_BUILTIN, ledState);
        Serial.print("LED: ");
        Serial.println(ledState ? "ON" : "OFF");
    }
}
