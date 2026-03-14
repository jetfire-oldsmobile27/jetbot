#include "Logging.hpp"
#include "Globals.hpp"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

std::filesystem::path getLogFilePath() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  std::ostringstream oss;
  oss << "bot-" << std::put_time(std::localtime(&t), "%Y%m%d") << ".log";
  return std::filesystem::path(logs_dir) / oss.str();
}

void logMsg(const std::string &msg) {
  auto logPath = getLogFilePath();
  std::ofstream ofs(logPath, std::ios::app);
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  ofs << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << " — " << msg
      << "\n";
}

std::string readFile(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open())
    return {};
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

void cleanupOldVideos() {
  if (!flag_cleanup)
    return;
  try {
    auto now = std::chrono::system_clock::now();
    auto thirty_days_ago = now - std::chrono::hours(24 * 30);

    for (const auto &entry : std::filesystem::directory_iterator(video_dir)) {
      if (entry.path().extension() == ".mp4") {
        auto ftime = std::filesystem::last_write_time(entry);

        auto sctp =
            std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - decltype(ftime)::clock::now() +
                std::chrono::system_clock::now());

        if (sctp < thirty_days_ago) {
          std::filesystem::remove(entry.path());
          logMsg("Удален старый видеофайл: " + entry.path().string());
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Ошибка при удалении старых видео: " << e.what() << std::endl;
    logMsg("Ошибка при удалении старых видео: " + std::string(e.what()));
  }
}