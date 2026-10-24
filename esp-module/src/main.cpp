#include "core_esp8266_features.h"
#include "pins_arduino.h"
#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <cstdint>

const char* ssid = "JetLegacy_Bus"; 
const char* password = "50504022"; 
const char* wifi_client_name = "ESP-ServerRemoteMotherboardCtl";
constexpr int led_pin = D4;


// --- Веб-сервер ---
ESP8266WebServer server(8083);


// --- Функции обработчики эндпоинтов ---

void handleRoot() {
  String html = "<html><body>";
  html += "<h1>IR Signal Controller</h1>";
  html += "<p><a href='/signal'>/signal</a> - Send the last recorded IR signal</p>";
  html += "<p><a href='/signal-info'>/signal-info</a> - Get information about the last recorded IR signal</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
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

void handlePushPowerButton(const uint16_t& milliseconds_when_push) {
  digitalWrite(D2, HIGH);
  delay(milliseconds_when_push);
  digitalWrite(D2, LOW);
  server.send(200, "text/plain", "Done");
};


void pushShortlyPowerButton() {
  handlePushPowerButton(700);
}

void pushLongPowerButton() {
  handlePushPowerButton(6000);
}

void setup() {
  delay(800); // waiting for flash
  Serial.begin(115200);
  Serial.println("\nStarting IR Signal Controller...");
  
  WiFi.setHostname(wifi_client_name);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());


  pinMode(led_pin, OUTPUT);
  // Blink when ready
  for(int i = 0; i < 3; ++i){
    digitalWrite(led_pin, LOW);
    delay(700);
    digitalWrite(led_pin, HIGH);
    delay(400);
  };

  pinMode(D2, OUTPUT);

  server.on("/", HTTP_GET, handleRoot);
  server.on("/ping", HTTP_GET, handlePing);
  server.on("/push-power-button", HTTP_GET, pushShortlyPowerButton);
  server.on("/turn-off-pc-immidiatelly", HTTP_GET, pushLongPowerButton);
  server.onNotFound(handleNotFound);

  server.begin();
}

void loop() {
  server.handleClient();

  yield();
}