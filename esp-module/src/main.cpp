#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <IRremoteESP8266.h>
#include <IRrecv.h>
#include <IRsend.h>
#include <IRutils.h> 
#include <ArduinoJson.h>

const char* ssid = "Ne_vlezay_ubjet"; 
const char* password = "50504022"; 

const uint16_t kRecvPin = 4; 
const uint16_t kIrLed = 5;   

IRrecv irrecv(kRecvPin);
IRsend irsend(kIrLed);
decode_results results; // Хранит результаты последнего декодированного сигнала

// --- Веб-сервер ---
ESP8266WebServer server(80);

// --- Переменная для хранения последнего сигнала ---
// Используем структуру, которая может хранить сырые данные о сигнале
#define SIGNAL_BUFFER_SIZE 1024 // Размер буфера для хранения сигнала (может потребоваться увеличить)
uint16_t signalBuffer[SIGNAL_BUFFER_SIZE];
uint16_t signalLength = 0;
bool signalReceived = false;

// --- Функции обработчики эндпоинтов ---

void handleRoot() {
  String html = "<html><body>";
  html += "<h1>IR Signal Controller</h1>";
  html += "<p><a href='/signal'>/signal</a> - Send the last recorded IR signal</p>";
  html += "<p><a href='/signal-info'>/signal-info</a> - Get information about the last recorded IR signal</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleSignal() {
    if (!signalReceived) {
        server.send(404, "text/plain", "No signal recorded yet");
        return;
    }

    Serial.println("Sending recorded IR signal...");
    irsend.sendRaw(signalBuffer, signalLength, 38); // 38 kHz 
    delay(100);

    server.send(200, "text/plain", "IR signal sent");
}

void handleSignalInfo() {
    if (!signalReceived) {
        server.send(404, "text/plain", "No signal recorded yet");
        return;
    }

    const size_t capacity = JSON_OBJECT_SIZE(5) + 1024; 
    DynamicJsonDocument doc(capacity);

    doc["protocol"] = typeToString(results.decode_type);
    doc["bits"] = results.bits;
    doc["value"] = String(results.value, HEX); 
    doc["address"] = String(results.address, HEX); 
    doc["command"] = String(results.command, HEX); 

    JsonArray rawData = doc.createNestedArray("raw_data");
    for (uint16_t i = 0; i < signalLength && i < SIGNAL_BUFFER_SIZE; i++) {
        rawData.add(signalBuffer[i]);
    }

    doc["raw_length"] = signalLength;

    String jsonString;
    serializeJson(doc, jsonString);

    server.send(200, "application/json", jsonString);
}

void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n";
  for (uint8_t i = 0; i < server.args(); i++) {
    message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
  }
  server.send(404, "text/plain", message);
}

void handlePing() {
  bool isBrowserRequest = false;
  for (int i = 0; i < server.args(); i++) {
    if (server.argName(i) == "browser") {
      isBrowserRequest = true;
      break;
    }
  }

  if (isBrowserRequest || server.hasHeader("User-Agent")) {
    String html = "<html><head><title>Ping Success</title></head><body>";
    html += "<h1>Ping Successful!</h1>";
    html += "<p>The server is running and responding correctly.</p>";
    html += "<a href='/'>Back to Home</a>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  } else {
    server.send(200, "text/plain", "OK");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("\nStarting IR Signal Controller...");

  irsend.begin();

  irrecv.enableIRIn();

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());

  server.on("/", HTTP_GET, handleRoot);
  server.on("/signal", HTTP_GET, handleSignal);
  server.on("/ping", HTTP_GET, handlePing);
  server.on("/signal-info", HTTP_GET, handleSignalInfo);
  server.onNotFound(handleNotFound);

  server.begin();
}

void loop() {
  server.handleClient();

  if (irrecv.decode(&results)) {
    /*Serial.println("IR signal received:");
    Serial.print("Protocol: ");
    Serial.println(typeToString(results.decode_type));
    Serial.print("Bits: ");
    Serial.println(results.bits);
    Serial.print("Value: ");
    Serial.println(results.value, HEX);*/

    signalLength = results.rawlen - 1; 
    if (signalLength > SIGNAL_BUFFER_SIZE) {
        signalLength = SIGNAL_BUFFER_SIZE;
    }
    for (uint16_t i = 0; i < signalLength; i++) {
        signalBuffer[i] = results.rawbuf[i + 1]; 
        // Serial.print(signalBuffer[i]); Serial.print(" ");
    }
    // Serial.println();

    signalReceived = true;

    irrecv.resume(); 
  }

  yield();
}