#include "Globals.hpp"
#include "Config.hpp"
#include "Logging.hpp"
#include "VideoThread.hpp"

#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <tgbot/Bot.h>
#include <tgbot/net/TgLongPoll.h>

#include "ObserverLoop/VideoRecorder.hpp"
#include "Utility/Settings.hpp"
#include "Utility/Debug-server/DebugServer.hpp"

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

cv::Mat last_frame;
cv::Mat last_raw_frame;
cv::Mat last_recognition_frame;
std::mutex frame_mutex;
std::mutex settings_mutex;

std::string jetbot_dir;
std::string video_dir;
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
    debug_server = std::make_shared<Utility::DebugServer::DebugServer>(flags.debug_port);
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

  // Тесты (закомментированы)
  // bool test_result = Tests::makeTests(CAMERA_INDEX);
  // std::cout << "Tests " << (test_result ? "passed" : "failed") << std::endl;
  // return test_result ? 0 : 1;

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

  TgBot::Bot bot(TG_API_TOKEN);
  ObserverLoop::VideoRecorder recorder(video_dir);

  std::thread video_thread(video_processing_thread, &bot, std::ref(recorder),
                           std::ref(settings), flags);

  bot.getEvents().onCommand("start", [&](TgBot::Message::Ptr message) {
    auto user = message->from;
    auto uid = user->id;
    auto uname = user->username.empty() ? "<no-username>" : user->username;
    if (authorizedUserId == 0) {
      authorizedUserId = uid;
      settings.authorizedUserId = uid;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        settings.save(settings_path);
      }
      logMsg("🔓 Доступ выдан: " + uname + " (ID:" + std::to_string(uid) + ")");
      bot.getApi().sendMessage(message->chat->id,
                               "✅ Вы авторизованы!\nВаш ID: " +
                                   std::to_string(uid));
    } else if (uid == authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "ℹ️ Вы уже авторизованы.");
    } else {
      logMsg("❌ Попытка /start от неразрешённого: " + uname +
             " (ID:" + std::to_string(uid) + ")");
      bot.getApi().sendMessage(message->chat->id,
                               "⛔ Доступ имеет только первый обратившийся.");
    }
  });

  bot.getEvents().onCommand("photo", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
      return;
    }
    cv::Mat frame_copy;
    {
      std::lock_guard<std::mutex> lock(frame_mutex);
      if (last_frame.empty()) {
        bot.getApi().sendMessage(message->chat->id, "⚠️ Нет данных с камеры.");
        return;
      }
      frame_copy = last_frame.clone();
    }
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream fn;
    fn << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S") << ".jpg";
    std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / fn.str();
    if (!cv::imwrite(tmp.string(), frame_copy)) {
      bot.getApi().sendMessage(message->chat->id,
                               "❌ Ошибка сохранения кадра.");
      return;
    }
    bot.getApi().sendPhoto(message->chat->id, TgBot::InputFile::fromFile(
                                                  tmp.string(), "image/jpeg"));
    std::filesystem::remove(tmp);
  });

  bot.getEvents().onCommand("alert", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
      return;
    }
    std::string text = message->text;
    std::string arg;
    size_t pos = text.find(' ');
    if (pos != std::string::npos) {
      arg = text.substr(pos + 1);
    }
    if (arg == "on") {
      alert_enabled = true;
      settings.alert_enabled = true;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        settings.save(settings_path);
      }
      bot.getApi().sendMessage(message->chat->id, "🔔 Уведомления включены");
    } else if (arg == "off") {
      alert_enabled = false;
      settings.alert_enabled = false;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        settings.save(settings_path);
      }
      bot.getApi().sendMessage(message->chat->id, "🔕 Уведомления выключены");
    } else {
      bot.getApi().sendMessage(message->chat->id,
                               "Использование: /alert on или /alert off");
    }
  });

  bot.getEvents().onCommand("last", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
      return;
    }

    if (!flag_recording) {
      bot.getApi().sendMessage(
          message->chat->id, "⚠️ Запись видео отключена (флаг --no-recording).");
      return;
    }

    auto now = std::chrono::system_clock::now();
    time_t now_c = std::chrono::system_clock::to_time_t(now);
    tm now_tm = *std::localtime(&now_c);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
    std::string today = buf;

    std::string video_path = recorder.getVideoPathForDate(today);

    if (!std::filesystem::exists(video_path) ||
        std::filesystem::file_size(video_path) <= 48) {
      for (int i = 1; i <= 7; ++i) {
        auto date = now - std::chrono::hours(24 * i);
        time_t date_c = std::chrono::system_clock::to_time_t(date);
        tm date_tm = *std::localtime(&date_c);
        char date_buf[20];
        strftime(date_buf, sizeof(date_buf), "%Y%m%d", &date_tm);
        std::string date_str = date_buf;

        std::string path = recorder.getVideoPathForDate(date_str);
        if (std::filesystem::exists(path) &&
            std::filesystem::file_size(path) > 48) {
          video_path = path;
          break;
        }
      }
    }

    if (video_path.empty() || !std::filesystem::exists(video_path) ||
        std::filesystem::file_size(video_path) <= 48) {
      bot.getApi().sendMessage(message->chat->id, "Нет записанных видео");
    } else {
      try {
        if (recorder.isRecording()) {
          recorder.stopRecording();
          std::this_thread::sleep_for(std::chrono::seconds(1));
          recorder.startRecording();
        }

        bot.getApi().sendDocument(
            message->chat->id,
            TgBot::InputFile::fromFile(video_path, "video/mp4"));
      } catch (const std::exception &e) {
        std::cerr << "Ошибка отправки видео: " << e.what() << std::endl;
        bot.getApi().sendMessage(message->chat->id,
                                 "❌ Ошибка отправки видео: " +
                                     std::string(e.what()));
      }
    }
  });

  bot.getEvents().onCommand("unstopable", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
      return;
    }
    std::string text = message->text;
    std::string arg;
    size_t pos = text.find(' ');
    if (pos != std::string::npos) {
      arg = text.substr(pos + 1);
    }
    if (arg == "on") {
      unstopable_mode = true;
      settings.unstopable_mode = true;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        settings.save(settings_path);
      }
      bot.getApi().sendMessage(message->chat->id,
                               "🔄 Режим непрерывной записи ВКЛЮЧЕН");
    } else if (arg == "off") {
      unstopable_mode = false;
      settings.unstopable_mode = false;
      {
        std::lock_guard<std::mutex> lock(settings_mutex);
        settings.save(settings_path);
      }
      bot.getApi().sendMessage(message->chat->id,
                               "🔄 Режим непрерывной записи ВЫКЛЮЧЕН");
    } else {
      bot.getApi().sendMessage(
          message->chat->id,
          "Использование: /unstopable on или /unstopable off");
    }
  });

  bot.getEvents().onCommand("status", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
      bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
      return;
    }
    std::string status = "📊 Статус системы:\n";
    status +=
        "Запись: " +
        std::string(recorder.isRecording() ? "✅ активна" : "❌ неактивна") +
        "\n";

    if (unstopable_mode) {
      status += "Режим: 🔄 непрерывная запись\n";
    } else {
      status += "Режим: 🎯 запись при обнаружении\n";
    }

    status += "Детекция: " +
              std::string(flag_detection ? "✅ активна" : "❌ отключена") +
              "\n";
    status +=
        "Лица: " + std::string(flag_face ? "✅ активна" : "❌ отключена") +
        "\n";
    status += "Эффекты: " +
              std::string(flag_effects ? "✅ включены" : "❌ отключены") + "\n";
    status +=
        "Звук: " + std::string(flag_sound ? "✅ включён" : "❌ отключён") +
        "\n";
    status += "Очистка: " +
              std::string(flag_cleanup ? "✅ активна" : "❌ отключена") + "\n";
    status +=
        "Люди в кадре: " +
        std::string(detection_active ? "⚠️ обнаружены" : "✅ не обнаружены") +
        "\n";
    status += "Уведомления: " +
              std::string(alert_enabled ? "🔔 включены" : "🔕 выключены") +
              "\n";
    status += "Авторизован: ID " + std::to_string(authorizedUserId);
    bot.getApi().sendMessage(message->chat->id, status);
  });

  bot.getEvents().onCommand("cpuinfo", [&](TgBot::Message::Ptr message) {
    auto info = readFile("/proc/cpuinfo");
    if (info.empty()) {
      bot.getApi().sendMessage(message->chat->id,
                               "❌ Не удалось прочитать /proc/cpuinfo");
    } else {
      if (info.size() < 3500) {
        bot.getApi().sendMessage(message->chat->id, info);
      } else {
        std::string tmp = "/tmp/cpuinfo.txt";
        std::ofstream(tmp) << info;
        bot.getApi().sendDocument(
            message->chat->id, TgBot::InputFile::fromFile(tmp, "text/plain"));
        std::filesystem::remove(tmp);
      }
    }
  });

  bot.getEvents().onCommand("temp", [&](TgBot::Message::Ptr message) {
    std::ostringstream report;
    bool found = false;
    const std::filesystem::path thermalDir{"/sys/class/thermal"};
    if (std::filesystem::exists(thermalDir) &&
        std::filesystem::is_directory(thermalDir)) {
      for (auto &entry : std::filesystem::directory_iterator(thermalDir)) {
        auto name = entry.path().filename().string();
        if (name.rfind("thermal_zone", 0) != 0)
          continue;
        std::filesystem::path typeFile = entry.path() / "type";
        std::filesystem::path tempFile = entry.path() / "temp";
        if (!std::filesystem::exists(typeFile) ||
            !std::filesystem::exists(tempFile))
          continue;
        std::string typeStr, tempStr;
        std::ifstream(typeFile) >> typeStr;
        std::ifstream(tempFile) >> tempStr;
        try {
          double tc = std::stod(tempStr) / 1000.0;
          report << "[" << name << "] " << typeStr << ": " << std::fixed
                 << std::setprecision(2) << tc << " °C\n";
          found = true;
        } catch (...) {
          report << "[" << name << "] " << typeStr << ": invalid (" << tempStr
                 << ")\n";
          found = true;
        }
      }
    }
    const std::filesystem::path hwmonDir{"/sys/class/hwmon"};
    if (std::filesystem::exists(hwmonDir) &&
        std::filesystem::is_directory(hwmonDir)) {
      for (auto &entry : std::filesystem::directory_iterator(hwmonDir)) {
        std::string chip;
        std::filesystem::path nameFile = entry.path() / "name";
        if (std::filesystem::exists(nameFile)) {
          std::ifstream(nameFile) >> chip;
        } else {
          chip = entry.path().filename().string();
        }
        for (auto &f : std::filesystem::directory_iterator(entry.path())) {
          auto fname = f.path().filename().string();
          if (fname.rfind("temp", 0) != 0 ||
              fname.find("_input") == std::string::npos)
            continue;
          std::string idx = fname.substr(4, fname.find("_input") - 4);
          std::filesystem::path inFile = f.path();
          std::filesystem::path labelFile =
              entry.path() / ("temp" + idx + "_label");
          std::string tempStr, label;
          std::ifstream(inFile) >> tempStr;
          if (std::filesystem::exists(labelFile)) {
            std::getline(std::ifstream(labelFile), label);
          } else {
            label = "temp" + idx;
          }
          try {
            double tc = std::stod(tempStr) / 1000.0;
            report << "[" << entry.path().filename() << "] " << chip << " "
                   << label << ": " << std::fixed << std::setprecision(2) << tc
                   << " °C\n";
            found = true;
          } catch (...) {
            report << "[" << entry.path().filename() << "] " << chip << " "
                   << label << ": invalid (" << tempStr << ")\n";
            found = true;
          }
        }
      }
    }
    if (!found) {
      bot.getApi().sendMessage(message->chat->id,
                               "❌ Не найден ни один температурный датчик.");
    } else {
      bot.getApi().sendMessage(message->chat->id, report.str());
    }
  });

  bot.getEvents().onCommand("logs", [&](TgBot::Message::Ptr message) {
    auto logPath = getLogFilePath();
    if (!std::filesystem::exists(logPath)) {
      bot.getApi().sendMessage(message->chat->id,
                               "📄 Лог за сегодня не найден");
    } else {
      bot.getApi().sendDocument(
          message->chat->id,
          TgBot::InputFile::fromFile(logPath.string(), "text/plain"));
    }
  });

  try {
    std::cout << "Bot username: " << bot.getApi().getMe()->username
              << std::endl;
    TgBot::TgLongPoll longPoll(bot);
    while (running) {
      longPoll.start();
    }
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    logMsg("🛑 Exception: " + std::string(e.what()));
  }

  running = false;
  video_thread.join();
  return 0;
}
