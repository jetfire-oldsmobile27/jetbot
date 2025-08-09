#include <tgbot/Bot.h>
#include <tgbot/net/TgLongPoll.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/objdetect.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <boost/json.hpp>
#include <boost/algorithm/string.hpp>
#include <csignal>

namespace fs = std::filesystem;
namespace json = boost::json;

// Конфигурационные параметры
const int WIDTH = 800;
const int HEIGHT = 600;
const int CAMERA_INDEX = 0;
const int FPS = 25;
const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;

// Глобальные переменные
std::atomic<bool> running(true);
std::atomic<bool> alert_enabled(true);
std::atomic<bool> recording(false);
std::atomic<bool> motion_detected(false);
std::atomic<int64_t> authorizedUserId(0);
cv::Mat last_frame;
std::mutex frame_mutex;
std::mutex settings_mutex;
std::string jetbot_dir;
std::string video_dir;
std::string logs_dir;
std::string settings_path;

// Структура настроек
struct Settings {
    int64_t authorizedUserId = 0;
    bool alert_enabled = true;
    std::string yolo_weights = "yolov3-tiny.weights";
    std::string yolo_cfg = "yolov3-tiny.cfg";
    std::string cascade_path = "haarcascade_frontalface_default.xml";

    void load(const std::string& path) {
        try {
            if (!fs::exists(path)) return;
            std::ifstream ifs(path);
            std::string content((std::istreambuf_iterator<char>(ifs)), 
                                std::istreambuf_iterator<char>());
            json::value jv = json::parse(content);
            json::object obj = jv.as_object();

            if (obj.contains("authorizedUserId"))
                authorizedUserId = obj["authorizedUserId"].as_int64();
            if (obj.contains("alert_enabled"))
                alert_enabled = obj["alert_enabled"].as_bool();
            if (obj.contains("yolo_weights"))
                yolo_weights = obj["yolo_weights"].as_string().c_str();
            if (obj.contains("yolo_cfg"))
                yolo_cfg = obj["yolo_cfg"].as_string().c_str();
            if (obj.contains("cascade_path"))
                cascade_path = obj["cascade_path"].as_string().c_str();
        } catch (...) {
            std::cerr << "Error loading settings" << std::endl;
        }
    }

    void save(const std::string& path) {
        json::object obj;
        obj["authorizedUserId"] = authorizedUserId;
        obj["alert_enabled"] = alert_enabled;
        obj["yolo_weights"] = yolo_weights;
        obj["yolo_cfg"] = yolo_cfg;
        obj["cascade_path"] = cascade_path;

        std::ofstream ofs(path);
        ofs << json::serialize(obj);
    }
};

// Возвращает путь к лог-файлу на сегодня
fs::path getLogFilePath() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << "bot-" << std::put_time(std::localtime(&t), "%Y%m%d") << ".log";
    return logs_dir + "/" + oss.str();
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

// Класс для записи видео
class VideoRecorder {
public:
    VideoRecorder(const std::string& video_dir) : video_dir_(video_dir) {}
    
    ~VideoRecorder() {
        if (isRecording()) {
            stopRecording();
        }
    }

    void startRecording() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (writer_.isOpened()) return;

        auto now = std::chrono::system_clock::now();
        time_t now_c = std::chrono::system_clock::to_time_t(now);
        tm now_tm = *std::localtime(&now_c);
        char buf[20];
        strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
        current_date_ = buf;
        std::string filename = video_dir_ + "/" + current_date_ + ".mp4";

        writer_.open(filename, cv::VideoWriter::fourcc('a','v','c','1'), 25.0, cv::Size(640, 480));
        if (!writer_.isOpened()) {
            writer_.open(filename, cv::VideoWriter::fourcc('m','p','4','v'), 25.0, cv::Size(640, 480));
        }
    }

    void stopRecording() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (writer_.isOpened()) {
            writer_.release();
        }
    }

    void writeFrame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!writer_.isOpened()) return;

        auto now = std::chrono::system_clock::now();
        time_t now_c = std::chrono::system_clock::to_time_t(now);
        tm now_tm = *std::localtime(&now_c);
        char buf[20];
        strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
        std::string today = buf;

        if (today != current_date_) {
            writer_.release();
            current_date_ = today;
            std::string filename = video_dir_ + "/" + current_date_ + ".mp4";
            writer_.open(filename, cv::VideoWriter::fourcc('a','v','c','1'), 25.0, cv::Size(640, 480));
            if (!writer_.isOpened()) {
                writer_.open(filename, cv::VideoWriter::fourcc('m','p','4','v'), 25.0, cv::Size(640, 480));
            }
        }

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 480));
        writer_ << resized;
    }

    bool isRecording() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return writer_.isOpened();
    }

private:
    std::string video_dir_;
    cv::VideoWriter writer_;
    std::string current_date_;
    mutable std::mutex mutex_;
};

// Класс для детекции движения
class MotionDetector {
public:
    MotionDetector() {
        bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);
    }

    bool detect(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat fgmask;
        bgSubtractor->apply(gray, fgmask);
        cv::threshold(fgmask, fgmask, 128, 255, cv::THRESH_BINARY);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::erode(fgmask, fgmask, kernel, cv::Point(-1,-1), 1);
        cv::dilate(fgmask, fgmask, kernel, cv::Point(-1,-1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area > 500) {
                return true;
            }
        }
        return false;
    }

private:
    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor;
};

// Функция обработки видео
void video_processing_thread(TgBot::Bot* bot, VideoRecorder& recorder, Settings& settings) {
    cv::VideoCapture cap;
    MotionDetector motion_detector;
    bool wasMotion = false;
    auto start_time = std::chrono::steady_clock::now();
    double lastMotionTime = 0.0;
    int skip_frames = 30;
    int skipped = 0;

    while (running) {
        if (!cap.isOpened()) {
            cap.open(CAMERA_INDEX);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(cv::CAP_PROP_FPS, FPS);
                skipped = 0;
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            cap.release();
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame.copyTo(last_frame);
        }

        if (skipped < skip_frames) {
            motion_detector.detect(frame);
            skipped++;
            continue;
        }

        auto loop_start = std::chrono::steady_clock::now();
        double current_time = std::chrono::duration<double>(loop_start - start_time).count();

        bool motion = motion_detector.detect(frame);
        motion_detected = motion;

        if (motion) {
            lastMotionTime = current_time;
            if (!recording) {
                recording = true;
                recorder.startRecording();
                if (alert_enabled && authorizedUserId != 0) {
                    std::string tmp_path = "/tmp/motion_alert.jpg";
                    cv::imwrite(tmp_path, frame);
                    try {
                        bot->getApi().sendPhoto(authorizedUserId, 
                            TgBot::InputFile::fromFile(tmp_path, "image/jpeg"),
                            "Обнаружено движение в кадре");
                    } catch (...) {
                        logMsg("Ошибка отправки уведомления");
                    }
                    fs::remove(tmp_path);
                }
            }
        }

        if (recording) {
            recorder.writeFrame(frame);
            if (!motion && (current_time - lastMotionTime > 5.0)) {
                recording = false;
                recorder.stopRecording();
            }
        }

        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < 40) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40 - elapsed));
        }
    }
}

// Обработчик сигналов
void signalHandler(int signal) {
    running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Инициализация путей
    jetbot_dir = std::string(getenv("HOME")) + "/jetbot";
    video_dir = jetbot_dir + "/video";
    logs_dir = jetbot_dir + "/logs";
    settings_path = jetbot_dir + "/settings.json";

    fs::create_directories(jetbot_dir);
    fs::create_directories(video_dir);
    fs::create_directories(logs_dir);

    // Загрузка настроек
    Settings settings;
    settings.load(settings_path);
    authorizedUserId = settings.authorizedUserId;
    alert_enabled = settings.alert_enabled;

    const std::string BOT_TOKEN = "7639692292:AAELfPguE_-DbZq-BUfixw885VYPwHPhErs";
    TgBot::Bot bot(BOT_TOKEN);
    VideoRecorder recorder(video_dir);

    // Запуск потока обработки видео
    std::thread video_thread(video_processing_thread, &bot, std::ref(recorder), std::ref(settings));

    // Обработчики команд
    bot.getEvents().onCommand("start", [&](TgBot::Message::Ptr message) {
        auto user = message->from;
        auto uid  = user->id;
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
                "✅ Вы авторизованы!\nВаш ID: " + std::to_string(uid));
        } else if (uid == authorizedUserId) {
            bot.getApi().sendMessage(message->chat->id, "ℹ️ Вы уже авторизованы.");
        } else {
            logMsg("❌ Попытка /start от неразрешённого: " + uname + " (ID:" + std::to_string(uid) + ")");
            bot.getApi().sendMessage(message->chat->id, "⛔ Доступ имеет только первый обратившийся.");
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
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream fn;
        fn << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S") << ".jpg";
        fs::path tmp = fs::temp_directory_path() / fn.str();

        if (!cv::imwrite(tmp.string(), frame_copy)) {
            bot.getApi().sendMessage(message->chat->id, "❌ Ошибка сохранения кадра.");
            return;
        }
        
        bot.getApi().sendPhoto(message->chat->id,
            TgBot::InputFile::fromFile(tmp.string(), "image/jpeg"));
        fs::remove(tmp);
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
            bot.getApi().sendMessage(message->chat->id, "Использование: /alert on или /alert off");
        }
    });

    bot.getEvents().onCommand("last", [&](TgBot::Message::Ptr message) {
    auto uid = message->from->id;
    if (uid != authorizedUserId) {
        bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
        return;
    }

    std::string last_video;
    fs::file_time_type last_time = fs::file_time_type::min();

    for (const auto& entry : fs::directory_iterator(video_dir)) {
        if (entry.path().extension() != ".mp4") continue;
        try {
            auto ftime = fs::last_write_time(entry);
            if (ftime > last_time) {
                last_time = ftime;
                last_video = entry.path().string();
            }
        } catch (const std::exception& e) {
            // на случай проблем с правами / недоступными файлами
            logMsg(std::string("warning: failed to stat ") + entry.path().string() + ": " + e.what());
            continue;
        }
    }

    if (last_video.empty()) {
        bot.getApi().sendMessage(message->chat->id, "📭 Нет записанных видео");
    } else {
        bot.getApi().sendDocument(message->chat->id,
            TgBot::InputFile::fromFile(last_video, "video/mp4"));
    }
    });


    bot.getEvents().onCommand("status", [&](TgBot::Message::Ptr message) {
        auto uid = message->from->id;
        if (uid != authorizedUserId) {
            bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
            return;
        }
        
        std::string status = "📊 Статус системы:\n";
        status += "Запись: " + std::string(recording ? "✅ активна" : "❌ неактивна") + "\n";
        status += "Движение: " + std::string(motion_detected ? "⚠️ обнаружено" : "⛔ не обнаружено") + "\n";
        status += "Уведомления: " + std::string(alert_enabled ? "🔔 включены" : "🔕 выключены") + "\n";
        status += "Авторизован: ID " + std::to_string(authorizedUserId);
        
        bot.getApi().sendMessage(message->chat->id, status);
    });

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

    try {
        std::cout << "Bot username: " << bot.getApi().getMe()->username << std::endl;
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