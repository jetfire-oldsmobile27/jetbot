#include "Config.hpp"
#include "Globals.hpp"
#include "Logging.hpp"
#include "VideoThread.hpp"

#include <vkbot/BotBase.hpp>
#include "OutAPI/VkBot.hpp"
#include "ObserverLoop/Recording/RecordingManager.hpp"
#include "Utility/Debug-server/DebugServer.hpp"
#include "Utility/Settings.hpp"
#include "OutAPI/OllamaServer.hpp"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <tgbot/Bot.h>
#include <tgbot/net/TgLongPoll.h>
#include <thread>

std::atomic<bool> flag_recording{true};
std::atomic<bool> flag_detection{true};
std::atomic<bool> flag_effects{true};
std::atomic<bool> flag_sound{true};
std::atomic<bool> flag_cleanup{true};
std::atomic<bool> flag_animation{true};
std::atomic<bool> flag_face{true};

std::atomic<bool> running{true};
std::atomic<bool> alert_enabled{true};
std::atomic<bool> detection_active{false};
std::atomic<bool> unstopable_mode{false};
std::atomic<int64_t> authorizedUserId{0};
std::atomic<vk::bot::BotBase*> g_vk_bot{nullptr};

cv::Mat last_frame;
cv::Mat last_raw_frame;
cv::Mat last_recognition_frame;
std::mutex frame_mutex;
std::mutex settings_mutex;

std::string video_dir;
std::string jetbot_dir;
std::string logs_dir;
std::string settings_path;
std::string resource_dir;

const int CAMERA_INDEX = []() -> int {
  const char *env = std::getenv("CAM_INDEX");
  return (env && *env) ? std::stoi(env) : 0;
}();

const std::string TG_API_TOKEN = []() -> std::string {
  const char *env = std::getenv("TG_API_TOKEN");
  return env ? env : "";
}();

const std::string VK_ACCESS_TOKEN = []() -> std::string {
  const char *env = std::getenv("VK_ACCESS_TOKEN");
  return env ? env : "";
}();

const std::string VK_GROUP_ID = []() -> std::string {
  const char *env = std::getenv("VK_GROUP_ID");
  return env ? env : "";
}();

const std::string ONNX_MODEL_PATH = []() -> std::string {
  const char *env = std::getenv("ONNX_MODEL_PATH");
  return env ? env
             : "/home/jetpclaptop/workspace/projects/jetbot/tgbotcpp/build/"
             "yolov10n.onnx";
}();

const std::string COCO_NAMES_PATH = []() -> std::string {
  const char *env = std::getenv("COCO_NAMES_PATH");
  return env ? env
             : "/home/jetpclaptop/workspace/projects/jetbot/tgbotcpp/build/"
             "coco.names";
}();

const std::string OLLAMA_IP = []() -> std::string {
  const char* env = std::getenv("OLLAMA_IP");
  return env ? env : "";
}();

const std::string LOCAL_MACHINE_IP = []() -> std::string {
  const char *env = std::getenv("LOCAL_MACHINE_IP");
  return env ? env : "";
}();

const std::string LOCAL_MACHINE_PORT = []() -> std::string {
  const char *env = std::getenv("LOCAL_MACHINE_PORT");
  return env ? env : "";
}();

void signalHandler(int) { running = false; }

int main(int argc, char *argv[]) {
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  ConfigFlags flags = parseCmdlineArgs(argc, argv);
  if (flags.help) {
    printHelp(argv[0]);
    return 0;
  }

  std::shared_ptr<Utility::DebugServer::DebugServer> debug_server;
  if (flags.debug_server) {
    debug_server =
        std::make_shared<Utility::DebugServer::DebugServer>(flags.debug_port);
    debug_server->start();
  }

  flag_recording = flags.recording;
  flag_detection = flags.detection;
  flag_effects = flags.effects;
  flag_sound = flags.sound;
  flag_cleanup = flags.cleanup;
  flag_animation = flags.animation;
  flag_face = flags.face;

  resource_dir = Utility::Settings::GetResourceDirFromExePath();

  // --- Инициализация директорий JetBot ---
  jetbot_dir = std::string(getenv("HOME")) + "/jetbot";
  video_dir = jetbot_dir + "/video";
  logs_dir = jetbot_dir + "/logs";
  settings_path = jetbot_dir + "/settings.json";
  std::filesystem::create_directories(jetbot_dir);
  std::filesystem::create_directories(video_dir);
  std::filesystem::create_directories(logs_dir);
  
  Utility::Settings settings;
  settings.load(settings_path);
  authorizedUserId = settings.authorizedUserId;
  alert_enabled = settings.alert_enabled;
  unstopable_mode = settings.unstopable_mode;

  // --- Подсистема записи видео (GStreamer: raw + processed + USB-микрофон) ---
  ObserverLoop::Recording::RecordingConfig recording_config;
  recording_config.output_dir = video_dir;
  recording_config.width = WIDTH;
  recording_config.height = HEIGHT;
  recording_config.fps = FPS;
  recording_config.segment_duration = std::chrono::seconds(15);
  recording_config.synchronous_files = true;
  // Тот же индекс камеры, что использует StartupManager (CAM_INDEX).
  recording_config.camera_device = "/dev/video" + std::to_string(CAMERA_INDEX);

  if (const char *camera_device = std::getenv("JETBOT_CAMERA")) {
    recording_config.camera_device = camera_device;
  }
  if (const char *microphone_device = std::getenv("JETBOT_MIC")) {
    recording_config.microphone_device = microphone_device;
  } else {
    logMsg("Запись: микрофон не задан, используется ALSA-устройство \"default\". "
           "Для USB-микрофона задайте JETBOT_MIC=hw:<card>,<device> "
           "(см. arecord -l).");
  }
  if (const char *encoder_chain = std::getenv("JETBOT_ENCODER_CHAIN")) {
    recording_config.video_encoder_chain = encoder_chain;
  }

  std::unique_ptr<ObserverLoop::Recording::RecordingManager> recorder_manager;
  try {
    recorder_manager =
        std::make_unique<ObserverLoop::Recording::RecordingManager>(
            recording_config);
  } catch (const std::exception &e) {
    logMsg("❌ Некорректная конфигурация записи: " + std::string(e.what()));
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }

  // Запуск потока обработки видео (он же владелец камеры и записи)
  std::thread video_thread(video_processing_thread, nullptr,
                           std::ref(*recorder_manager), std::ref(settings),
                           flags);


  auto vk_bot = std::make_unique<vk::bot::BotBase>(VK_GROUP_ID);
  if (!vk_bot->auth(VK_ACCESS_TOKEN)) {
      // Камера и запись уже могли быть запущены в видеопотоке: уходим штатно,
      // иначе std::thread уничтожится в joinable-состоянии и будет terminate.
      std::cerr << "VK Auth failed\n";
      running = false;
      video_thread.join();
      return 1;
  }

  std::thread vk_thread(vk_bot_thread, std::ref(settings), std::ref(*recorder_manager), std::ref(*vk_bot));

  while (running) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  // Останавливаем VK-цикл, затем дожидаемся видеопоток: он сам завершает
  // GStreamer-сессию записи (EOS и финализация сегментов).
  vk_bot->interrupt();
  video_thread.join();
  vk_thread.join();
  return 0;
}