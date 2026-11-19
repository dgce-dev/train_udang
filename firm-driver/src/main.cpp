#include <Arduino.h>

const int MOS_PIN = 25; // Pin untuk driver XY-MOS
const int TDS_PIN = 33; // Pin untuk sensor TDS (ADC1)

// Konfigurasi PWM ESP32 (Khusus Core v2)
const int pwmChannel = 0;    // Gunakan channel 0
const int pwmFreq = 5000;    // Frekuensi 5000 Hz
const int pwmResolution = 8; // Resolusi 8-bit (0-255)

// Variabel status
bool isAutoMode = false;
int currentPWM = 0;
unsigned long lastTDSReadTime = 0;
const unsigned long TDS_READ_INTERVAL = 1000; // Baca sensor setiap 1000ms

void setup()
{
    Serial.begin(115200);

    // Konfigurasi PWM ESP32 (Core v2)
    ledcSetup(pwmChannel, pwmFreq, pwmResolution);
    ledcAttachPin(MOS_PIN, pwmChannel);

    Serial.println("\n=== Sistem Kontrol Lampu Aktif ===");
    Serial.println("Kirim 'M<nilai 0-100>' untuk Mode Manual. Contoh: M50");
    Serial.println("Kirim 'A' untuk Mode Auto (Berdasarkan Sensor TDS)");
    Serial.println("==================================");
}

void loop()
{
    // --- LOGIKA 1: Membaca Input dari Serial Monitor ---
    if (Serial.available() > 0)
    {
        String input = Serial.readStringUntil('\n');
        input.trim();        // Hapus spasi atau karakter newline (\r\n)
        input.toUpperCase(); // Ubah ke huruf besar

        if (input.startsWith("M"))
        {
            isAutoMode = false;
            // Ambil angka persentase setelah huruf 'M'
            int percent = input.substring(1).toInt();

            // Batasi nilai agar tidak lebih dari 100
            percent = constrain(percent, 0, 100);

            // Konversi format 0-100% ke resolusi PWM 0-255
            currentPWM = map(percent, 0, 100, 0, 255);

            // Terapkan ke PWM Channel (Core v2)
            ledcWrite(pwmChannel, currentPWM);

            Serial.print("[MANUAL MODE] Lampu disetel ke: ");
            Serial.print(percent);
            Serial.println("%");
        }
        else if (input.startsWith("A"))
        {
            isAutoMode = true;
            Serial.println("[AUTO MODE] Berpindah ke kontrol sensor TDS...");
        }
    }

    // --- LOGIKA 2: Mode Otomatis (Sensor TDS) ---
    if (isAutoMode)
    {
        unsigned long currentMillis = millis();

        if (currentMillis - lastTDSReadTime >= TDS_READ_INTERVAL)
        {
            lastTDSReadTime = currentMillis;

            // Baca nilai analog TDS (0 - 4095)
            int tdsRawValue = analogRead(TDS_PIN);

            // Pemetaan nilai mentah TDS ke PWM
            currentPWM = map(tdsRawValue, 0, 4095, 0, 255);
            currentPWM = constrain(currentPWM, 0, 255);

            // Terapkan ke PWM Channel (Core v2)
            ledcWrite(pwmChannel, currentPWM);

            int displayPercent = map(currentPWM, 0, 255, 0, 100);

            Serial.print("[AUTO] Raw TDS: ");
            Serial.print(tdsRawValue);
            Serial.print(" | Output Lampu: ");
            Serial.print(displayPercent);
            Serial.println("%");
        }
    }
}