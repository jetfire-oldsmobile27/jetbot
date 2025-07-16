#include <tgbot/Bot.h>
#include <tgbot/net/TgLongPoll.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

static std::int64_t authorizedUserId = 0;

// Возвращает путь к лог-файлу на сегодня: bot-YYYYMMDD.log
fs::path getLogFilePath() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << "bot-" << std::put_time(std::localtime(&t), "%Y%m%d") << ".log";
    return fs::current_path() / oss.str();
}

void logMsg(const std::string &msg) {
    auto logPath = getLogFilePath();
    std::ofstream ofs(logPath, std::ios::app);
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    ofs << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
        << " — " << msg << "\n";
}

std::string readFile(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

int main() {
    const std::string BOT_TOKEN = "7639692292:AAELfPguE_-DbZq-BUfixw885VYPwHPhErs";
    TgBot::Bot bot(BOT_TOKEN);

    // --- /start и /photo (без изменений) ---
    bot.getEvents().onCommand("start", [&](TgBot::Message::Ptr message) {
        auto chatId = message->chat->id;
        auto user   = message->from;
        auto uid    = user->id;
        auto uname  = user->username.empty() ? "<no-username>" : user->username;
        if (authorizedUserId == 0) {
            authorizedUserId = uid;
            logMsg("🔓 Доступ выдан: " + uname + " (ID:" + std::to_string(uid) + ")");
            bot.getApi().sendMessage(chatId,
                "✅ Вы авторизованы!\nВаш ID: " + std::to_string(uid));
        } else if (uid == authorizedUserId) {
            bot.getApi().sendMessage(chatId, "ℹ️ Вы уже авторизованы.");
        } else {
            logMsg("❌ Попытка /start от неразрешённого: " + uname + " (ID:" + std::to_string(uid) + ")");
            bot.getApi().sendMessage(chatId, "⛔ Доступ имеет только первый обратившийся.");
        }
    });

    bot.getEvents().onCommand("photo", [&](TgBot::Message::Ptr message) {
        auto chatId = message->chat->id;
        auto user   = message->from;
        auto uid    = user->id;
        auto uname  = user->username.empty() ? "<no-username>" : user->username;
        if (uid != authorizedUserId) {
            logMsg("❌ Несанкционированный /photo от " + uname);
            bot.getApi().sendMessage(chatId, "⛔ У вас нет доступа.");
            return;
        }
        cv::VideoCapture cam(0, cv::CAP_ANY);
        if (!cam.isOpened()) {
            bot.getApi().sendMessage(chatId, "⚠️ Не удалось открыть камеру.");
            return;
        }
        cv::Mat frame;
        if (!cam.read(frame) || frame.empty()) {
            bot.getApi().sendMessage(chatId, "⚠️ Не удалось захватить изображение.");
            return;
        }
        cam.release();

        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream fn;
        fn << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S") << ".jpg";
        fs::path tmp = fs::temp_directory_path() / fn.str();

        if (!cv::imwrite(tmp.string(), frame)) {
            bot.getApi().sendMessage(chatId, "❌ Ошибка при сохранении кадра.");
            return;
        }
        logMsg("📸 Фото отправлено " + uname);
        bot.getApi().sendPhoto(chatId,
            TgBot::InputFile::fromFile(tmp.string(), "image/jpeg"));
        fs::remove(tmp);
    });

    // --- НОВЫЕ КОМАНДЫ ---

    // /cpuinfo
    bot.getEvents().onCommand("cpuinfo", [&](TgBot::Message::Ptr message) {
        auto info = readFile("/proc/cpuinfo");
        if (info.empty()) {
            bot.getApi().sendMessage(message->chat->id, "❌ Не удалось прочитать /proc/cpuinfo");
        } else {
            // Telegram ограничивает длину, можно разбить или отправить как файл
            if (info.size() < 3500) {
                bot.getApi().sendMessage(message->chat->id, info);
            } else {
                // отправим как файл
                std::string tmp = "/tmp/cpuinfo.txt";
                std::ofstream(tmp) << info;
                bot.getApi().sendDocument(message->chat->id,
                    TgBot::InputFile::fromFile(tmp, "text/plain"));
                fs::remove(tmp);
            }
        }
    });

    // /temp
    bot.getEvents().onCommand("temp", [&](TgBot::Message::Ptr message) {
        std::ostringstream report;
        bool found = false;

        // 1) thermal_zone*
        const fs::path thermalDir{"/sys/class/thermal"};
        if (fs::exists(thermalDir) && fs::is_directory(thermalDir)) {
            for (auto &entry : fs::directory_iterator(thermalDir)) {
                auto name = entry.path().filename().string();
                if (name.rfind("thermal_zone", 0) != 0) continue;

                fs::path typeFile = entry.path() / "type";
                fs::path tempFile = entry.path() / "temp";
                if (!fs::exists(typeFile) || !fs::exists(tempFile)) continue;

                std::string typeStr, tempStr;
                std::ifstream(typeFile) >> typeStr;
                std::ifstream(tempFile) >> tempStr;
                try {
                    double tc = std::stod(tempStr) / 1000.0;
                    report << "[" << name << "] " << typeStr
                           << ": " << std::fixed << std::setprecision(2)
                           << tc << " °C\n";
                    found = true;
                } catch (...) {
                    report << "[" << name << "] " << typeStr
                           << ": invalid (" << tempStr << ")\n";
                    found = true;
                }
            }
        }

        // 2) hwmon*
        const fs::path hwmonDir{"/sys/class/hwmon"};
        if (fs::exists(hwmonDir) && fs::is_directory(hwmonDir)) {
            for (auto &entry : fs::directory_iterator(hwmonDir)) {
                // читаем имя драйвера (если есть)
                std::string chip;
                fs::path nameFile = entry.path() / "name";
                if (fs::exists(nameFile)) {
                    std::ifstream(nameFile) >> chip;
                } else {
                    chip = entry.path().filename().string();
                }

                // перебираем tempN_input
                for (auto &f : fs::directory_iterator(entry.path())) {
                    auto fname = f.path().filename().string();
                    if (fname.rfind("temp", 0) != 0 || fname.find("_input") == std::string::npos)
                        continue;

                    std::string idx = fname.substr(4, fname.find("_input") - 4);
                    fs::path inFile = f.path();
                    fs::path labelFile = entry.path() / ("temp" + idx + "_label");

                    std::string tempStr, label;
                    std::ifstream(inFile) >> tempStr;
                    if (fs::exists(labelFile)) {
                        std::getline(std::ifstream(labelFile), label);
                    } else {
                        label = "temp" + idx;
                    }
                    try {
                        double tc = std::stod(tempStr) / 1000.0;
                        report << "[" << entry.path().filename() << "] "
                               << chip << " " << label
                               << ": " << std::fixed << std::setprecision(2)
                               << tc << " °C\n";
                        found = true;
                    } catch (...) {
                        report << "[" << entry.path().filename() << "] "
                               << chip << " " << label
                               << ": invalid (" << tempStr << ")\n";
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

    // /logs
    bot.getEvents().onCommand("logs", [&](TgBot::Message::Ptr message) {
        auto logPath = getLogFilePath();
        if (!fs::exists(logPath)) {
            bot.getApi().sendMessage(message->chat->id, "📄 Лог за сегодня не найден");
        } else {
            bot.getApi().sendDocument(message->chat->id,
                TgBot::InputFile::fromFile(logPath.string(), "text/plain"));
        }
    });

    // запуск
    try {
        std::cout << "Bot username: " << bot.getApi().getMe()->username << std::endl;
        TgBot::TgLongPoll longPoll(bot);
        while (true) {
            longPoll.start();
        }
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        logMsg("🛑 Exception: " + std::string(e.what()));
    }
    return 0;
}
