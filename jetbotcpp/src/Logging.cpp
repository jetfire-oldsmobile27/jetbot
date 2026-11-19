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
  std::cout << msg << std::endl;
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

    const auto to_system_time = [](std::filesystem::file_time_type ftime) {
      return std::chrono::time_point_cast<std::chrono::system_clock::duration>(
          ftime - decltype(ftime)::clock::now() +
          std::chrono::system_clock::now());
    };

    for (const auto &entry : std::filesystem::directory_iterator(video_dir)) {
      std::error_code type_ec;
      const std::string name = entry.path().filename().string();

      // Текущий формат записи: каталог сессии session_<дата>_<время>_<pid>
      // с сегментами camera_*.mkv и processed_*.mkv внутри.
      if (entry.is_directory(type_ec) && name.rfind("session_", 0) == 0) {
        std::filesystem::file_time_type newest{};
        bool have_newest = false;
        std::error_code walk_ec;
        for (const auto &file :
             std::filesystem::directory_iterator(entry.path(), walk_ec)) {
          if (walk_ec)
            break;
          std::error_code file_ec;
          if (!file.is_regular_file(file_ec))
            continue;
          const auto write_time = file.last_write_time(file_ec);
          if (file_ec)
            continue;
          if (!have_newest || write_time > newest) {
            newest = write_time;
            have_newest = true;
          }
        }

        const auto dir_time = have_newest
                                  ? to_system_time(newest)
                                  : to_system_time(
                                        std::filesystem::last_write_time(entry));

        if (dir_time < thirty_days_ago) {
          std::error_code remove_ec;
          std::filesystem::remove_all(entry.path(), remove_ec);
          if (remove_ec) {
            logMsg("Не удалось удалить старую сессию записи " +
                   entry.path().string() + ": " + remove_ec.message());
          } else {
            logMsg("Удалена старая сессия записи: " + entry.path().string());
          }
        }
        continue;
      }

      // Прежний формат: отдельные файлы .mp4 прямо в каталоге видео.
      if (entry.is_regular_file(type_ec) &&
          entry.path().extension() == ".mp4") {
        if (to_system_time(std::filesystem::last_write_time(entry)) <
            thirty_days_ago) {
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