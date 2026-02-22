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
#include <boost/json.hpp>
#include <boost/algorithm/string.hpp>
#include <csignal>
#include <random>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>             // для парсинга аргументов

#include "Utility/Settings.hpp"
#include "Utility/Effects.hpp"
#include "ObserverLoop/VideoRecorder.hpp"
#include "Startup/StartupManager.hpp"
#include "Tests/Test.hpp"

// ==================== Структура флагов конфигурации ====================
struct ConfigFlags {
    bool recording   = true;   // запись видео
    bool detection   = true;   // детекция людей (YOLO)
    bool effects     = true;   // визуальные эффекты (красный оттенок, кинескоп, дамп)
    bool sound       = true;   // звук при обнаружении
    bool cleanup     = true;   // очистка старых видео
    bool animation   = true;   // анимация при старте
    bool face        = true;   // детекция лиц

    bool help        = false;  // показать справку
};

// Глобальные атомарные флаги (для простоты доступа из потока)
std::atomic<bool> flag_recording{true};
std::atomic<bool> flag_detection{true};
std::atomic<bool> flag_effects{true};
std::atomic<bool> flag_sound{true};
std::atomic<bool> flag_cleanup{true};
std::atomic<bool> flag_animation{true};
std::atomic<bool> flag_face{true};

// ==================== Парсинг аргументов командной строки ====================
void printHelp(const char* progname) {
    std::cout << "Использование: " << progname << " [ОПЦИИ]\n"
              << "Опции:\n"
              << "  --no-recording      Отключить запись видео на диск\n"
              << "  --no-detection       Отключить детекцию людей (YOLO)\n"
              << "  --no-effects         Отключить визуальные эффекты (красный оттенок, кинескоп, дамп памяти)\n"
              << "  --no-sound           Отключить звуковой сигнал при обнаружении\n"
              << "  --no-cleanup         Отключить автоматическую очистку старых видео\n"
              << "  --no-animation       Отключить анимацию при инициализации\n"
              << "  --no-face            Отключить детекцию лиц\n"
              << "  --help, -h           Показать эту справку\n";
}

ConfigFlags parseArgs(int argc, char* argv[]) {
    ConfigFlags flags;
    const struct option long_options[] = {
        {"no-recording", no_argument, nullptr, 'r'},
        {"no-detection",  no_argument, nullptr, 'd'},
        {"no-effects",    no_argument, nullptr, 'e'},
        {"no-sound",      no_argument, nullptr, 's'},
        {"no-cleanup",    no_argument, nullptr, 'c'},
        {"no-animation",  no_argument, nullptr, 'a'},
        {"no-face",       no_argument, nullptr, 'f'},
        {"help",          no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'r': flags.recording = false; break;
            case 'd': flags.detection = false; break;
            case 'e': flags.effects   = false; break;
            case 's': flags.sound     = false; break;
            case 'c': flags.cleanup   = false; break;
            case 'a': flags.animation = false; break;
            case 'f': flags.face      = false; break;
            case 'h': flags.help      = true;  break;
            default: break;
        }
    }
    return flags;
}

// ==================== Остальные глобальные переменные ====================
const int WIDTH = 640;
const int HEIGHT = 480;
const int CAMERA_INDEX = []() -> int {
    const char* env = std::getenv("CAM_INDEX");
    if (env && std::strlen(env) > 0) {
        try {
            return std::stoi(env);
        } catch (const std::exception& e) {
            std::cerr << "Invalid CAM_INDEX value: " << env 
                      << " (using default: 0). Error: " << e.what() << std::endl;
        }
    }
    return 0;
}();
const int FPS = 25;
const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int RECORDING_DURATION = 60;  // 60 секунд записи после обнаружения
const int INITIAL_RECORDING_DURATION = 60;  // 60 секунд записи при старте

std::atomic<bool> running(true);
std::atomic<bool> alert_enabled(true);
std::atomic<bool> detection_active(false);
std::atomic<bool> unstopable_mode(false);
std::atomic<int64_t> authorizedUserId(0);
cv::Mat last_frame;
std::mutex frame_mutex;
std::mutex settings_mutex;

std::string jetbot_dir;
std::string video_dir;
std::string logs_dir;
std::string settings_path;
std::string resource_dir;

// ==================== Логирование и вспомогательные функции ====================
std::filesystem::path getLogFilePath() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << "bot-" << std::put_time(std::localtime(&t), "%Y%m%d") << ".log";
    return std::filesystem::path(logs_dir) / oss.str();
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

void cleanupOldVideos() {
    if (!flag_cleanup) return;   // проверка флага
    try {
        auto now = std::chrono::system_clock::now();
        auto thirty_days_ago = now - std::chrono::hours(24 * 30);

        for (const auto& entry : std::filesystem::directory_iterator(video_dir)) {
            if (entry.path().extension() == ".mp4") {
                auto ftime = std::filesystem::last_write_time(entry);

                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - decltype(ftime)::clock::now() + std::chrono::system_clock::now()
                );

                if (sctp < thirty_days_ago) {
                    std::filesystem::remove(entry.path());
                    logMsg("Удален старый видеофайл: " + entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при удалении старых видео: " << e.what() << std::endl;
        logMsg("Ошибка при удалении старых видео: " + std::string(e.what()));
    }
}

// ==================== Класс трекера (без изменений) ====================
class ObjectTracker {
public:
    ObjectTracker(int max_disappeared = 5) : next_id_(0), max_disappeared_(max_disappeared) {}
    
    int registerObject(const cv::Point& centroid, const cv::Rect& bbox) {
        objects_[next_id_] = std::make_pair(centroid, bbox);
        disappeared_[next_id_] = 0;
        return next_id_++;
    }
    
    void deregister(int object_id) {
        objects_.erase(object_id);
        disappeared_.erase(object_id);
    }
    
    std::map<int, std::pair<cv::Point, cv::Rect>> update(const std::vector<std::pair<cv::Point, cv::Rect>>& detections) {
        if (detections.empty()) {
            for (auto it = disappeared_.begin(); it != disappeared_.end();) {
                it->second++;
                if (it->second > max_disappeared_) {
                    deregister(it->first);
                    it = disappeared_.begin();
                } else {
                    ++it;
                }
            }
            return objects_;
        }
        
        std::vector<bool> used_detections(detections.size(), false);
        
        for (auto& object : objects_) {
            int object_id = object.first;
            cv::Point centroid = object.second.first;
            cv::Rect bbox = object.second.second;
            
            double min_dist = std::numeric_limits<double>::max();
            int min_idx = -1;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                if (used_detections[i]) continue;
                
                cv::Point det_centroid = detections[i].first;
                double dist = cv::norm(centroid - det_centroid);
                
                if (dist < min_dist && dist < 100) {
                    min_dist = dist;
                    min_idx = i;
                }
            }
            
            if (min_idx != -1) {
                objects_[object_id] = detections[min_idx];
                disappeared_[object_id] = 0;
                used_detections[min_idx] = true;
            } else {
                disappeared_[object_id]++;
                if (disappeared_[object_id] > max_disappeared_) {
                    deregister(object_id);
                }
            }
        }
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!used_detections[i]) {
                registerObject(detections[i].first, detections[i].second);
            }
        }
        
        return objects_;
    }
    
private:
    int next_id_;
    int max_disappeared_;
    std::map<int, std::pair<cv::Point, cv::Rect>> objects_;
    std::map<int, int> disappeared_;
};

// ==================== Оптимизированная детекция YOLO ====================
std::vector<std::pair<cv::Point, cv::Rect>> detectPeopleYolo(cv::Mat& frame, cv::dnn::Net& net, 
                                                           const std::vector<cv::String>& output_layers,
                                                           const std::vector<std::string>& classes,
                                                           float conf_threshold = 0.5, 
                                                           float nms_threshold = 0.4) {
    // если детекция отключена глобально, возвращаем пустой вектор
    if (!flag_detection) return {};

    int height = frame.rows;
    int width = frame.cols;
    std::vector<std::pair<cv::Point, cv::Rect>> people;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(320, 320), cv::Scalar(), true, false);
    net.setInput(blob);
    
    std::vector<cv::Mat> outs;
    net.forward(outs, output_layers);
    
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classId;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            
            if (confidence > conf_threshold && classes[classId.x] == "person") {
                int centerX = (int)(data[0] * width);
                int centerY = (int)(data[1] * height);
                int w = (int)(data[2] * width);
                int h = (int)(data[3] * height);
                
                int x = centerX - w / 2;
                int y = centerY - h / 2;
                
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(x, y, w, h));
            }
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        int x = boxes[idx].x;
        int y = boxes[idx].y;
        int w = boxes[idx].width;
        int h = boxes[idx].height;
        
        cv::Point centroid(x + w/2, y + h/2);
        people.push_back(std::make_pair(centroid, boxes[idx]));
    }
    
    return people;
}

// ==================== Применение эффектов ====================
cv::Mat applyEffects(cv::Mat& frame, cv::Mat& memory_dump) {
    // если эффекты отключены, просто возвращаем исходный кадр
    if (!flag_effects) {
        return frame.clone();
    }

    // 1. Применение красных оттенков
    cv::Mat processed;
    
    if (frame.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        
        std::vector<cv::Mat> red_channels(3);
        red_channels[0] = cv::Mat::zeros(frame.size(), CV_8UC1);  // B
        red_channels[1] = cv::Mat::zeros(frame.size(), CV_8UC1);  // G
        red_channels[2] = channels[2];                            // R
        
        cv::Mat red_channel;
        cv::merge(red_channels, red_channel);
        
        cv::addWeighted(frame, 0.3, red_channel, 0.7, 0, processed);
    } else {
        frame.copyTo(processed);
    }
    
    // 2. Эффект кинескопа
    for (int i = 1; i < processed.rows; i += 2) {
        processed.row(i) = cv::Scalar(0, 0, 0);
    }
    
    // 3. Конвертация в BGRA
    cv::Mat processed_bgra;
    cv::cvtColor(processed, processed_bgra, cv::COLOR_BGR2BGRA);
    
    // 4. Наложение дампа памяти
    if (!memory_dump.empty()) {
        int dump_width = memory_dump.cols;
        int dump_height = std::min(processed_bgra.rows, memory_dump.rows);
        
        for (int i = 0; i < dump_height; ++i) {
            for (int j = 0; j < dump_width; ++j) {
                cv::Vec4b dump_pixel = memory_dump.at<cv::Vec4b>(i, j);
                float alpha = dump_pixel[3] / 255.0f;
                cv::Vec4b& frame_pixel = processed_bgra.at<cv::Vec4b>(i, j);
                
                for (int c = 0; c < 3; ++c) {
                    frame_pixel[c] = static_cast<uchar>(
                        dump_pixel[c] * alpha + frame_pixel[c] * (1 - alpha)
                    );
                }
            }
        }
    }
    
    // 5. Обратно в BGR
    cv::Mat result;
    cv::cvtColor(processed_bgra, result, cv::COLOR_BGRA2BGR);
    
    return result;
}

// ==================== Обработчик сигналов ====================
void signalHandler(int signal) {
    running = false;
}

// ==================== Функция обработки видео (с флагами) ====================
void video_processing_thread(TgBot::Bot* bot, ObserverLoop::VideoRecorder& recorder, 
                              Utility::Settings& settings, const ConfigFlags& flags) {
    // Первоначальная очистка старых видео (с учётом флага)
    cleanupOldVideos();
    
    Startup::StartupManager startup_manager(WIDTH, HEIGHT);
    
    // Анимация инициализации
    if (flags.animation) {
        std::thread init_thread([&]() {
            startup_manager.initialize(resource_dir, CAMERA_INDEX, FPS);
        });
        init_thread.detach();

        while (!startup_manager.isInitializationComplete() || startup_manager.getAnimationPhase() < 1) {
            cv::Mat anim_frame = startup_manager.updateAnimation();
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                anim_frame.copyTo(last_frame);
            }
            
            // Запись видео, если разрешена
            if (flags.recording && !recorder.isRecording()) {
                recorder.startRecording();
            }
            
            cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);
            cv::Mat processed_anim = flags.effects ? applyEffects(anim_frame, memory_dump) : anim_frame.clone();
            if (flags.recording) {
                recorder.writeFrame(processed_anim);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    } else {
        // Если анимация отключена, просто инициализируем синхронно
        startup_manager.initialize(resource_dir, CAMERA_INDEX, FPS);
    }
    
    // Получаем компоненты
    cv::VideoCapture& cap = startup_manager.getCapture();
    cv::dnn::Net net = startup_manager.getNet();
    std::vector<std::string> classes = startup_manager.getClasses();
    std::vector<cv::String> output_layers = startup_manager.getOutputLayers();
    cv::CascadeClassifier face_cascade = startup_manager.getFaceCascade();
    
    ObjectTracker tracker(3);
    int frame_counter = 0;
    double last_dump_time = 0.0;
    cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);
    
    auto start_time = std::chrono::steady_clock::now();
    double last_detection_time = 0.0;
    auto system_start_time = start_time;
    
    while (running) {
        if (!cap.isOpened()) {
            cap.open(CAMERA_INDEX);
            if (cap.isOpened()) {
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                cap.set(cv::CAP_PROP_FPS, FPS);
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
        
        auto loop_start = std::chrono::steady_clock::now();
        double current_time = std::chrono::duration<double>(loop_start - start_time).count();
        double system_uptime = std::chrono::duration<double>(loop_start - system_start_time).count();
        
        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(640, 480));
        cv::Mat display_frame;
        cv::resize(small_frame, display_frame, cv::Size(WIDTH, HEIGHT));
        
        // Эффекты (если включены)
        cv::Mat processed;
        if (flags.effects) {
            if (current_time - last_dump_time > 1.0) {
                memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);
                last_dump_time = current_time;
            }
            processed = applyEffects(display_frame, memory_dump);
        } else {
            processed = display_frame.clone();
        }
        
        // Детекция людей
        std::vector<std::pair<cv::Point, cv::Rect>> detections;
        if (flags.detection && frame_counter % 3 == 0 && !net.empty()) {
            try {
                detections = detectPeopleYolo(small_frame, net, output_layers, classes,
                                               CONF_THRESHOLD, NMS_THRESHOLD);
            } catch (const cv::Exception& e) {
                std::cerr << "Ошибка детекции людей: " << e.what() << std::endl;
            }
        }
        
        std::map<int, std::pair<cv::Point, cv::Rect>> tracked_objects = tracker.update(detections);
        
        // Детекция лиц (если включена)
        std::vector<cv::Rect> detected_faces;
        if (flags.face && !face_cascade.empty() && !tracked_objects.empty()) {
            cv::Mat gray;
            cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
            
            for (const auto& obj : tracked_objects) {
                int object_id = obj.first;
                cv::Point centroid = obj.second.first;
                cv::Rect bbox = obj.second.second;
                
                int roi_y = std::max(0, bbox.y);
                int roi_h = std::min(static_cast<int>(bbox.height * 0.7), 480 - roi_y);
                int roi_x = std::max(0, bbox.x);
                int roi_w = std::min(bbox.width, 640 - roi_x);
                
                if (roi_h > 30 && roi_w > 30) {
                    cv::Mat roi = gray(cv::Rect(roi_x, roi_y, roi_w, roi_h));
                    std::vector<cv::Rect> faces;
                    face_cascade.detectMultiScale(roi, faces, 1.05, 5, 0, cv::Size(40, 40));
                    
                    for (const auto& face : faces) {
                        detected_faces.push_back(cv::Rect(roi_x + face.x, roi_y + face.y, face.width, face.height));
                    }
                }
            }
        }
        
        // Отрисовка результатов
        for (const auto& obj : tracked_objects) {
            int object_id = obj.first;
            cv::Point centroid = obj.second.first;
            cv::Rect bbox = obj.second.second;
            
            int x = static_cast<int>(bbox.x * WIDTH / 640.0);
            int y = static_cast<int>(bbox.y * HEIGHT / 480.0);
            int w = static_cast<int>(bbox.width * WIDTH / 640.0);
            int h = static_cast<int>(bbox.height * HEIGHT / 480.0);
            centroid = cv::Point(
                static_cast<int>(centroid.x * WIDTH / 640.0), 
                static_cast<int>(centroid.y * HEIGHT / 480.0)
            );
            
            cv::rectangle(processed, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 2);
            
            if (w > 60 && h > 100) {
                cv::rectangle(processed, cv::Point(x, y - 20), cv::Point(x + 100, y), cv::Scalar(0, 0, 0), -1);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distr(85, 99);
                cv::putText(processed, "Human: " + std::to_string(distr(gen)) + "%", 
                           cv::Point(x, y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);
            }
            
            cv::line(processed, cv::Point(centroid.x - 15, centroid.y), cv::Point(centroid.x + 15, centroid.y), 
                    cv::Scalar(0, 255, 0), 1);
            cv::line(processed, cv::Point(centroid.x, centroid.y - 15), cv::Point(centroid.x, centroid.y + 15), 
                    cv::Scalar(0, 255, 0), 1);
        }
        
        for (const auto& face : detected_faces) {
            int x = static_cast<int>(face.x * WIDTH / 640.0);
            int y = static_cast<int>(face.y * HEIGHT / 480.0);
            int w = static_cast<int>(face.width * WIDTH / 640.0);
            int h = static_cast<int>(face.height * HEIGHT / 480.0);
            
            cv::Point center(x + w / 2, y + h / 2);
            cv::Size axes(w / 2, h / 2);
            cv::ellipse(processed, center, axes, 0, 0, 360, cv::Scalar(0, 255, 0), 1);
            cv::line(processed, cv::Point(center.x - 10, center.y), cv::Point(center.x + 10, center.y), 
                    cv::Scalar(0, 255, 0), 1);
            cv::line(processed, cv::Point(center.x, center.y - 10), cv::Point(center.x, center.y + 10), 
                    cv::Scalar(0, 255, 0), 1);
        }
        
        // Звук обнаружения (если включён)
        if (flags.sound && !tracked_objects.empty() && frame_counter % 5 == 0) {
            Utility::Effects::playDetectSound();
        }
        
        // Информационный HUD
        cv::rectangle(processed, cv::Point(WIDTH - 180, 10), cv::Point(WIDTH - 10, 70), cv::Scalar(0, 0, 0, 200), -1);
        double fps = 1.0 / (std::chrono::duration<double>(loop_start - start_time).count() + 0.001);
        cv::putText(processed, "TARGETS: " + std::to_string(tracked_objects.size()), 
                   cv::Point(WIDTH - 170, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
        cv::putText(processed, "FPS: " + std::to_string(static_cast<int>(fps)), 
                   cv::Point(WIDTH - 170, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
        
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            processed.copyTo(last_frame);
        }
        
        // Логика записи видео
        if (flags.recording) {
            if (!tracked_objects.empty()) {
                last_detection_time = current_time;
                detection_active = true;
                
                if (!recorder.isRecording()) {
                    recorder.startRecording();
                    if (alert_enabled && authorizedUserId != 0) {
                        std::string tmp_path = "/tmp/detection_alert.jpg";
                        cv::imwrite(tmp_path, frame);
                        try {
                            bot->getApi().sendPhoto(authorizedUserId, 
                                TgBot::InputFile::fromFile(tmp_path, "image/jpeg"),
                                "Обнаружены люди в кадре");
                        } catch (...) {
                            logMsg("Ошибка отправки уведомления");
                        }
                        std::filesystem::remove(tmp_path);
                    }
                }
            }
            
            if (unstopable_mode) {
                if (!recorder.isRecording()) {
                    recorder.startRecording();
                }
                recorder.writeFrame(processed);
            }
            else if (recorder.isRecording()) {
                recorder.writeFrame(processed);
                if (tracked_objects.empty() && (current_time - last_detection_time > RECORDING_DURATION) && 
                    system_uptime > INITIAL_RECORDING_DURATION) {
                    recorder.stopRecording();
                }
            }
            
            if (system_uptime < INITIAL_RECORDING_DURATION && !recorder.isRecording()) {
                recorder.startRecording();
            }
        }
        
        frame_counter++;
        
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < 40) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40 - elapsed));
        }
        
        // Периодическая очистка (раз в час)
        static auto last_cleanup = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(loop_end - last_cleanup).count() > 3600) {
            cleanupOldVideos();   // внутри проверяется флаг
            last_cleanup = loop_end;
        }
    }
}

// ==================== main ====================
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Парсим аргументы командной строки
    ConfigFlags flags = parseArgs(argc, argv);
    if (flags.help) {
        printHelp(argv[0]);
        return 0;
    }

    // Копируем флаги в атомарные глобальные переменные для доступа из функций
    flag_recording = flags.recording;
    flag_detection = flags.detection;
    flag_effects   = flags.effects;
    flag_sound     = flags.sound;
    flag_cleanup   = flags.cleanup;
    flag_animation = flags.animation;
    flag_face      = flags.face;

    // Определяем путь к ресурсам
    resource_dir = Utility::Settings::GetResourceDirFromExePath();

    // Тесты (закомментированы)
    // bool test_result = Tests::makeTests(CAMERA_INDEX);
    // std::cout << "Tests " << (test_result ? "passed" : "failed") << std::endl;
    // return test_result ? 0 : 1;
    
    // Инициализация путей
    jetbot_dir = std::string(getenv("HOME")) + "/jetbot";
    video_dir = jetbot_dir + "/video";
    logs_dir = jetbot_dir + "/logs";
    settings_path = jetbot_dir + "/settings.json";
    
    std::filesystem::create_directories(jetbot_dir);
    std::filesystem::create_directories(video_dir);
    std::filesystem::create_directories(logs_dir);
    
    // Загрузка настроек
    Utility::Settings settings;
    settings.load(settings_path);
    authorizedUserId = settings.authorizedUserId;
    alert_enabled = settings.alert_enabled;
    unstopable_mode = settings.unstopable_mode;
    
    TgBot::Bot bot(settings.kBot_token);
    ObserverLoop::VideoRecorder recorder(video_dir);
    
    // Запуск потока обработки видео с передачей флагов
    std::thread video_thread(video_processing_thread, &bot, std::ref(recorder), std::ref(settings), flags);
    
    // Обработчики команд (без изменений, но в /last можно добавить проверку записи)
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
        std::filesystem::path tmp = std::filesystem::temp_directory_path() / fn.str();
        if (!cv::imwrite(tmp.string(), frame_copy)) {
            bot.getApi().sendMessage(message->chat->id, "❌ Ошибка сохранения кадра.");
            return;
        }
        bot.getApi().sendPhoto(message->chat->id,
            TgBot::InputFile::fromFile(tmp.string(), "image/jpeg"));
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
            bot.getApi().sendMessage(message->chat->id, "Использование: /alert on или /alert off");
        }
    });
    
    bot.getEvents().onCommand("last", [&](TgBot::Message::Ptr message) {
        auto uid = message->from->id;
        if (uid != authorizedUserId) {
            bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
            return;
        }

        // Если запись отключена, сообщаем об этом
        if (!flag_recording) {
            bot.getApi().sendMessage(message->chat->id, "⚠️ Запись видео отключена (флаг --no-recording).");
            return;
        }
        
        auto now = std::chrono::system_clock::now();
        time_t now_c = std::chrono::system_clock::to_time_t(now);
        tm now_tm = *std::localtime(&now_c);
        char buf[20];
        strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
        std::string today = buf;
        
        std::string video_path = recorder.getVideoPathForDate(today);
        
        if (!std::filesystem::exists(video_path) || std::filesystem::file_size(video_path) <= 48) {
            for (int i = 1; i <= 7; ++i) {
                auto date = now - std::chrono::hours(24 * i);
                time_t date_c = std::chrono::system_clock::to_time_t(date);
                tm date_tm = *std::localtime(&date_c);
                char date_buf[20];
                strftime(date_buf, sizeof(date_buf), "%Y%m%d", &date_tm);
                std::string date_str = date_buf;
                
                std::string path = recorder.getVideoPathForDate(date_str);
                if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 48) {
                    video_path = path;
                    break;
                }
            }
        }
        
        if (video_path.empty() || !std::filesystem::exists(video_path) || std::filesystem::file_size(video_path) <= 48) {
            bot.getApi().sendMessage(message->chat->id, "Нет записанных видео");
        } else {
            try {
                if (recorder.isRecording()) {
                    recorder.stopRecording();
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    recorder.startRecording();
                }
                
                bot.getApi().sendDocument(message->chat->id,
                    TgBot::InputFile::fromFile(video_path, "video/mp4"));
            } catch (const std::exception& e) {
                std::cerr << "Ошибка отправки видео: " << e.what() << std::endl;
                bot.getApi().sendMessage(message->chat->id, "❌ Ошибка отправки видео: " + std::string(e.what()));
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
            bot.getApi().sendMessage(message->chat->id, "🔄 Режим непрерывной записи ВКЛЮЧЕН");
        } else if (arg == "off") {
            unstopable_mode = false;
            settings.unstopable_mode = false;
            {
                std::lock_guard<std::mutex> lock(settings_mutex);
                settings.save(settings_path);
            }
            bot.getApi().sendMessage(message->chat->id, "🔄 Режим непрерывной записи ВЫКЛЮЧЕН");
        } else {
            bot.getApi().sendMessage(message->chat->id, "Использование: /unstopable on или /unstopable off");
        }
    });
    
    bot.getEvents().onCommand("status", [&](TgBot::Message::Ptr message) {
        auto uid = message->from->id;
        if (uid != authorizedUserId) {
            bot.getApi().sendMessage(message->chat->id, "⛔ У вас нет доступа.");
            return;
        }
        std::string status = "📊 Статус системы:\n";
        status += "Запись: " + std::string(recorder.isRecording() ? "✅ активна" : "❌ неактивна") + "\n";
        
        if (unstopable_mode) {
            status += "Режим: 🔄 непрерывная запись\n";
        } else {
            status += "Режим: 🎯 запись при обнаружении\n";
        }
        
        status += "Детекция: " + std::string(flag_detection ? "✅ активна" : "❌ отключена") + "\n";
        status += "Лица: " + std::string(flag_face ? "✅ активна" : "❌ отключена") + "\n";
        status += "Эффекты: " + std::string(flag_effects ? "✅ включены" : "❌ отключены") + "\n";
        status += "Звук: " + std::string(flag_sound ? "✅ включён" : "❌ отключён") + "\n";
        status += "Очистка: " + std::string(flag_cleanup ? "✅ активна" : "❌ отключена") + "\n";
        status += "Люди в кадре: " + std::string(detection_active ? "⚠️ обнаружены" : "✅ не обнаружены") + "\n";
        status += "Уведомления: " + std::string(alert_enabled ? "🔔 включены" : "🔕 выключены") + "\n";
        status += "Авторизован: ID " + std::to_string(authorizedUserId);
        bot.getApi().sendMessage(message->chat->id, status);
    });
    
    bot.getEvents().onCommand("cpuinfo", [&](TgBot::Message::Ptr message) {
        auto info = readFile("/proc/cpuinfo");
        if (info.empty()) {
            bot.getApi().sendMessage(message->chat->id, "❌ Не удалось прочитать /proc/cpuinfo");
        } else {
            if (info.size() < 3500) {
                bot.getApi().sendMessage(message->chat->id, info);
            } else {
                std::string tmp = "/tmp/cpuinfo.txt";
                std::ofstream(tmp) << info;
                bot.getApi().sendDocument(message->chat->id,
                    TgBot::InputFile::fromFile(tmp, "text/plain"));
                std::filesystem::remove(tmp);
            }
        }
    });
    
    bot.getEvents().onCommand("temp", [&](TgBot::Message::Ptr message) {
        std::ostringstream report;
        bool found = false;
        const std::filesystem::path thermalDir{"/sys/class/thermal"};
        if (std::filesystem::exists(thermalDir) && std::filesystem::is_directory(thermalDir)) {
            for (auto &entry : std::filesystem::directory_iterator(thermalDir)) {
                auto name = entry.path().filename().string();
                if (name.rfind("thermal_zone", 0) != 0) continue;
                std::filesystem::path typeFile = entry.path() / "type";
                std::filesystem::path tempFile = entry.path() / "temp";
                if (!std::filesystem::exists(typeFile) || !std::filesystem::exists(tempFile)) continue;
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
        const std::filesystem::path hwmonDir{"/sys/class/hwmon"};
        if (std::filesystem::exists(hwmonDir) && std::filesystem::is_directory(hwmonDir)) {
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
                    if (fname.rfind("temp", 0) != 0 || fname.find("_input") == std::string::npos)
                        continue;
                    std::string idx = fname.substr(4, fname.find("_input") - 4);
                    std::filesystem::path inFile = f.path();
                    std::filesystem::path labelFile = entry.path() / ("temp" + idx + "_label");
                    std::string tempStr, label;
                    std::ifstream(inFile) >> tempStr;
                    if (std::filesystem::exists(labelFile)) {
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
    
    bot.getEvents().onCommand("logs", [&](TgBot::Message::Ptr message) {
        auto logPath = getLogFilePath();
        if (!std::filesystem::exists(logPath)) {
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
