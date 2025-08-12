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
#include <random>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace json = boost::json;

// Конфигурационные параметры
const int WIDTH = 800;
const int HEIGHT = 600;
const int CAMERA_INDEX = 0;
const int FPS = 25;
const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const int RECORDING_DURATION = 60;  // 60 секунд записи после обнаружения
const int INITIAL_RECORDING_DURATION = 60;  // 60 секунд записи при старте

// Глобальные переменные
std::atomic<bool> running(true);
std::atomic<bool> alert_enabled(true);
std::atomic<bool> detection_active(false);
std::atomic<bool> unstopable_mode(false);  // Новый режим непрерывной записи
std::atomic<int64_t> authorizedUserId(0);
cv::Mat last_frame;
std::mutex frame_mutex;
std::mutex settings_mutex;
std::mutex video_mutex;
std::string jetbot_dir;
std::string video_dir;
std::string logs_dir;
std::string settings_path;
std::string resource_dir;  // Добавлено: путь к ресурсам

// Структура настроек
struct Settings {
    int64_t authorizedUserId = 0;
    bool alert_enabled = true;
    bool unstopable_mode = false;  // Новый параметр для режима непрерывной записи
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
            if (obj.contains("unstopable_mode"))
                unstopable_mode = obj["unstopable_mode"].as_bool();
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
        obj["unstopable_mode"] = unstopable_mode;
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
    return fs::path(logs_dir) / oss.str();
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

// Удаление файлов старше 30 дней
void cleanupOldVideos() {
    try {
        auto now = std::chrono::system_clock::now();
        auto thirty_days_ago = now - std::chrono::hours(24 * 30);

        for (const auto& entry : fs::directory_iterator(video_dir)) {
            if (entry.path().extension() == ".mp4") {
                auto ftime = fs::last_write_time(entry);

                // Преобразуем file_time_type → system_clock::time_point
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - decltype(ftime)::clock::now() + std::chrono::system_clock::now()
                );

                if (sctp < thirty_days_ago) {
                    fs::remove(entry.path());
                    logMsg("Удален старый видеофайл: " + entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при удалении старых видео: " << e.what() << std::endl;
        logMsg("Ошибка при удалении старых видео: " + std::string(e.what()));
    }
}

// Генерация псевдо-дампа памяти
cv::Mat generateMemoryDump(int width, int height) {
    int dump_width = width / 6;
    cv::Mat dump = cv::Mat::zeros(height, dump_width, CV_8UC4);  // BGRA
    
    // Полупрозрачный фон
    for (int i = 0; i < dump.rows; ++i) {
        for (int j = 0; j < dump.cols; ++j) {
            dump.at<cv::Vec4b>(i, j) = cv::Vec4b(20, 0, 20, 128);
        }
    }
    
    // Заголовок
    cv::putText(dump, "MEM DUMP:", cv::Point(10, 20), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 200, 255, 255), 1);
    
    // Генерация случайных hex-данных
    int line_height = 20;
    int num_lines = std::min(12, (height - 60) / line_height);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> addr_dist(0x1000, 0xFFFF);
    std::uniform_int_distribution<> byte_dist(0, 255);
    
    for (int i = 0; i < num_lines; ++i) {
        int y = 40 + i * line_height;
        
        // Генерация адреса
        std::stringstream ss;
        ss << std::hex << std::uppercase << std::setw(4) << std::setfill('0') 
           << addr_dist(gen);
        std::string address = ss.str();
        
        // Генерация значений
        std::string values;
        for (int j = 0; j < 4; ++j) {
            ss.str("");
            ss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') 
               << byte_dist(gen);
            values += ss.str() + " ";
        }
        
        cv::putText(dump, address + ": " + values, cv::Point(10, y), 
                   cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0, 200, 255, 255), 1);
    }
    
    return dump;
}

// Генерация звука обнаружения (сохранение в WAV-файл)
void generateDetectSound(const std::string& filename) {
    // Параметры звука
    int sample_rate = 22050;
    float duration = 0.3f;
    int num_samples = static_cast<int>(sample_rate * duration);
    
    // Создаем директорию, если её нет
    fs::create_directories(fs::path(filename).parent_path());
    
    // WAV-заголовок (простой 16-битный моно)
    char header[44];
    memset(header, 0, 44);
    
    // RIFF chunk descriptor
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    int32_t chunk_size = 36 + num_samples * 2; // 36 + данные
    memcpy(header + 4, &chunk_size, 4);
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    
    // Format subchunk
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    int32_t subchunk1_size = 16; // 16 для PCM
    memcpy(header + 16, &subchunk1_size, 4);
    int16_t audio_format = 1; // PCM
    memcpy(header + 20, &audio_format, 2);
    int16_t num_channels = 1; // Моно
    memcpy(header + 22, &num_channels, 2);
    memcpy(header + 24, &sample_rate, 4);
    int32_t byte_rate = sample_rate * num_channels * 2; // 16 бит
    memcpy(header + 28, &byte_rate, 4);
    int16_t block_align = num_channels * 2;
    memcpy(header + 32, &block_align, 2);
    int16_t bits_per_sample = 16;
    memcpy(header + 34, &bits_per_sample, 2);
    
    // Data subchunk
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    int32_t data_size = num_samples * 2;
    memcpy(header + 40, &data_size, 4);
    
    // Генерируем звуковые данные
    std::vector<int16_t> sound_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float wave1 = 0.2f * sinf(2 * M_PI * 800 * t);
        float wave2 = 0.2f * sinf(2 * M_PI * 1200 * t);
        float wave = wave1 + wave2;
        float envelope = expf(-4 * t / duration);
        sound_data[i] = static_cast<int16_t>(wave * envelope * 10000.0f); // Громкость 10000
    }
    
    // Записываем в файл
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(header, 44);
    ofs.write(reinterpret_cast<const char*>(sound_data.data()), num_samples * 2);
    ofs.close();
}

// Воспроизведение звука
void playDetectSound() {
    static bool sound_initialized = false;
    static std::string sound_file = "/tmp/detect_sound.wav";
    
    // Инициализируем звук только один раз
    if (!sound_initialized) {
        generateDetectSound(sound_file);
        sound_initialized = true;
    }
    
    // Воспроизводим звук через системную команду
    // Используем nohup и &, чтобы звук играл в фоновом режиме
    system(("nohup aplay -q " + sound_file + " >/dev/null 2>&1 &").c_str());
}

// Класс для управления анимацией и инициализацией
class StartupManager {
public:
    StartupManager(int width, int height) : 
        width_(width), 
        height_(height),
        center_x_(width / 2), 
        center_y_(height / 2),
        circle_radius_(30),
        circle_distance_(120),
        animation_phase_(0),
        start_time_(std::chrono::steady_clock::now()),
        initialization_complete_(false),
        initialization_progress_(0.0f),
        cam_index_(-1),
        loading_text_("INITIALIZING SYSTEM"),
        pulse_value_(0.0f),
        camera_error_count_(0) {
        
        // Создаем 3 круга под углом 120 градусов
        for (int i = 0; i < 3; ++i) {
            float angle = 2 * M_PI * i / 3;
            int x = center_x_ + static_cast<int>(circle_distance_ * cos(angle));
            int y = center_y_ + static_cast<int>(circle_distance_ * sin(angle));
            circles_.push_back({x, y, 0, 1});
        }
    }
    
    // ДОБАВЛЕННЫЙ МЕТОД
    int getAnimationPhase() const {
        return animation_phase_;
    }
    
    cv::Mat updateAnimation() {
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time_).count();
        cv::Mat anim_frame = cv::Mat::zeros(height_, width_, CV_8UC3);
        float pulse_speed = 0.5f;  // Скорость пульсации
        
        // Фаза 0: Пульсация кругов во время инициализации
        if (animation_phase_ == 0) {
            // Пульсация кругов
            pulse_value_ = sin(elapsed * pulse_speed * 2 * M_PI) * 0.2f + 0.8f;
            int pulse_size = static_cast<int>(circle_radius_ * pulse_value_);
            
            for (auto& circle : circles_) {
                // Отрисовка круга
                cv::circle(anim_frame, cv::Point(circle.x, circle.y), pulse_size, cv::Scalar(255, 255, 255), -1);
            }
            
            // Отображение прогресса инициализации
            std::string progress_text = loading_text_ + ": " + std::to_string(static_cast<int>(initialization_progress_ * 100)) + "%";
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 0.7;
            int thickness = 2;
            int baseline;
            cv::Size text_size = cv::getTextSize(progress_text, font_face, font_scale, thickness, &baseline);
            int text_x = (width_ - text_size.width) / 2;
            int text_y = height_ - 50;
            cv::putText(anim_frame, progress_text, cv::Point(text_x, text_y), 
                        font_face, font_scale, cv::Scalar(200, 200, 200), thickness);
            
            // Переход к следующей фазе после завершения инициализации
            if (initialization_progress_ >= 1.0f) {
                animation_phase_ = 1;
                start_time_ = std::chrono::steady_clock::now();
            }
        }
        // Фаза 1: Основная анимация (схлопывание кругов)
        else if (animation_phase_ == 1) {
            elapsed = elapsed - 0.5f;  // Задержка перед началом анимации
            // Параметры анимации
            float text_alpha = 0.0f;
            int bar_width_left = 0;
            int bar_width_right = 0;
            float video_alpha = 0.0f;
            int max_bar_width = width_ / 2 - circle_distance_;
            
            // Фаза 1.1: Вращение кругов и появление полос (0-1.2 сек)
            if (elapsed < 1.2f) {
                float rotation = elapsed * 1.5f;  // Быстрое вращение
                for (size_t i = 0; i < circles_.size(); ++i) {
                    float angle = 2 * M_PI * i / 3 + rotation;
                    circles_[i].x = center_x_ + static_cast<int>(circle_distance_ * cos(angle));
                    circles_[i].y = center_y_ + static_cast<int>(circle_distance_ * sin(angle));
                }
                // Появление текста
                text_alpha = std::min(1.0f, elapsed / 1.2f);
                // Появление полос
                float bar_progress = std::min(1.0f, elapsed / 1.2f);
                bar_width_left = static_cast<int>(max_bar_width * bar_progress);
                bar_width_right = static_cast<int>(max_bar_width * bar_progress);
            }
            // Фаза 1.2: Схлопывание кругов и появление видео (1.2-2.0 сек)
            else if (elapsed < 2.0f) {
                float progress = (elapsed - 1.2f) / 0.8f;
                for (auto& circle : circles_) {
                    // Плавное перемещение к центру
                    circle.x = static_cast<int>(circle.x + (center_x_ - circle.x) * progress);
                    circle.y = static_cast<int>(circle.y + (center_y_ - circle.y) * progress);
                }
                // Появление видео
                video_alpha = std::min(1.0f, (elapsed - 1.5f) / 0.5f);
            }
            
            // Отрисовка кругов
            float progress = std::min(1.0f, (elapsed - 1.2f) / 0.8f);
            for (auto& circle : circles_) {
                int radius = static_cast<int>(circle_radius_ * (1 - progress) * pulse_value_);
                cv::circle(anim_frame, cv::Point(circle.x, circle.y), radius, cv::Scalar(255, 255, 255), -1);
            }
            
            // Отрисовка полос
            if (bar_width_left > 0) {
                // Левая полоса
                cv::rectangle(anim_frame, cv::Point(0, center_y_ - 2), 
                             cv::Point(bar_width_left, center_y_ + 2), cv::Scalar(255, 255, 255), -1);
                // Правая полоса
                cv::rectangle(anim_frame, cv::Point(width_ - bar_width_right, center_y_ - 2), 
                             cv::Point(width_, center_y_ + 2), cv::Scalar(255, 255, 255), -1);
            }
            
            // Отрисовка текста
            if (text_alpha > 0) {
                std::string text = "JetVision Systems";
                int font_face = cv::FONT_HERSHEY_SIMPLEX;
                double font_scale = 1.0;
                int thickness = 2;
                int baseline;
                cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
                int text_x = (width_ - text_size.width) / 2;
                int text_y = height_ - 50;
                cv::Scalar color(255 * text_alpha, 255 * text_alpha, 255 * text_alpha);
                cv::putText(anim_frame, text, cv::Point(text_x, text_y), 
                            font_face, font_scale, color, thickness);
            }
            
            // Если есть кадр с камеры, смешиваем с анимацией
            if (!frame_.empty() && video_alpha > 0) {
                // Применяем эффект кинескопа к видео
                cv::Mat video_frame = frame_.clone();
                for (int i = 1; i < video_frame.rows; i += 2) {
                    video_frame.row(i) = cv::Scalar(0, 0, 0);
                }
                
                // Смешиваем анимацию и видео
                cv::Mat weighted_video, weighted_anim;
                cv::addWeighted(video_frame, video_alpha, cv::Mat::zeros(height_, width_, CV_8UC3), 1 - video_alpha, 0, weighted_video);
                cv::addWeighted(anim_frame, 1 - video_alpha, cv::Mat::zeros(height_, width_, CV_8UC3), video_alpha, 0, weighted_anim);
                anim_frame = weighted_video + weighted_anim;
            }
        }
        
        return anim_frame;
    }
    
    void initialize() {
        // Шаг 1: Загрузка классов
        loading_text_ = "LOADING CLASSES";
        std::string classes_path = resource_dir + "/coco.names";
        std::ifstream class_file(classes_path);
        if (class_file.is_open()) {
            std::string line;
            while (std::getline(class_file, line)) {
                classes_.push_back(line);
            }
            class_file.close();
            initialization_progress_ = 0.1f;
            std::cout << "Классы загружены из: " << classes_path << std::endl;
        } else {
            std::cerr << "Не удалось загрузить классы из: " << classes_path << std::endl;
            classes_ = {"person"}; // Минимальный набор
        }
        
        // Шаг 2: Инициализация YOLO
        loading_text_ = "LOADING OBJECT DETECTOR";
        try {
            // Пробуем загрузить YOLO-tiny
            std::string weights_path = resource_dir + "/yolov3-tiny.weights";
            std::string cfg_path = resource_dir + "/yolov3-tiny.cfg";
            
            std::cout << "Попытка загрузить YOLO из: " << weights_path << " и " << cfg_path << std::endl;
            
            // Проверяем существование файлов
            if (!fs::exists(weights_path)) {
                std::cerr << "Файл весов не найден: " << weights_path << std::endl;
                throw std::runtime_error("Weights file not found");
            }
            if (!fs::exists(cfg_path)) {
                std::cerr << "Конфигурационный файл не найден: " << cfg_path << std::endl;
                throw std::runtime_error("Config file not found");
            }
            
            net_ = cv::dnn::readNet(weights_path, cfg_path);
            
            // ВСЕГДА ИСПОЛЬЗУЕМ CPU
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "Используется CPU (CUDA отключена)" << std::endl;
            
            // Получаем имена выходных слоев
            std::vector<cv::String> layer_names = net_.getLayerNames();
            for (int i : net_.getUnconnectedOutLayers()) {
                output_layers_.push_back(layer_names[i-1]);
            }
            
            initialization_progress_ = 0.6f;
        } catch (const cv::Exception& e) {
            std::cerr << "Не удалось загрузить YOLO: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Не удалось загрузить YOLO: неизвестная ошибка" << std::endl;
        }
        
        // Шаг 3: Загрузка каскада для лиц
        loading_text_ = "LOADING FACE DETECTOR";
        // Попробуем стандартные пути к файлу каскада
        std::vector<std::string> cascade_paths = {
            resource_dir + "/haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
        };
        
        bool cascade_loaded = false;
        for (const auto& path : cascade_paths) {
            if (fs::exists(path) && face_cascade_.load(path)) {
                cascade_loaded = true;
                std::cout << "Детектор лиц загружен из: " << path << std::endl;
                break;
            } else if (fs::exists(path)) {
                std::cout << "Файл каскада найден, но не загружен: " << path << std::endl;
            }
        }
        
        if (!cascade_loaded) {
            std::cerr << "Не удалось загрузить детектор лиц ни из одного пути" << std::endl;
        } else {
            initialization_progress_ = 0.8f;
        }
        
        // Шаг 4: Открытие камеры
        loading_text_ = "CONNECTING CAMERA";
        cap_.open(CAMERA_INDEX);
        if (cap_.isOpened()) {
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
            cap_.set(cv::CAP_PROP_FPS, FPS);
            
            // Получаем первый кадр
            cv::Mat frame;
            if (cap_.read(frame)) {
                cv::resize(frame, frame_, cv::Size(width_, height_));
            }
        }
        
        initialization_progress_ = 1.0f;
        initialization_complete_ = true;
    }
    
    bool isInitializationComplete() const {
        return initialization_complete_;
    }
    
    cv::Mat getFrame() const {
        return frame_;
    }
    
    cv::dnn::Net getNet() const {
        return net_;
    }
    
    std::vector<std::string> getClasses() const {
        return classes_;
    }
    
    std::vector<cv::String> getOutputLayers() const {
        return output_layers_;
    }
    
    cv::CascadeClassifier getFaceCascade() const {
        return face_cascade_;
    }
    
    cv::VideoCapture& getCapture() {
        return cap_;
    }

private:
    struct Circle {
        int x;
        int y;
        int radius;
        int pulse_direction;
    };
    
    int width_;
    int height_;
    int center_x_;
    int center_y_;
    int circle_radius_;
    int circle_distance_;
    int animation_phase_;
    std::chrono::steady_clock::time_point start_time_;
    bool initialization_complete_;
    float initialization_progress_;
    int cam_index_;
    std::string loading_text_;
    float pulse_value_;
    int camera_error_count_;
    
    std::vector<Circle> circles_;
    cv::Mat frame_;
    cv::VideoCapture cap_;
    cv::dnn::Net net_;
    std::vector<std::string> classes_;
    std::vector<cv::String> output_layers_;
    cv::CascadeClassifier face_cascade_;
};

// Класс для трекинга объектов
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
        
        // Обновление существующих объектов
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
        
        // Регистрация новых объектов
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

// Оптимизированная детекция с помощью YOLO
std::vector<std::pair<cv::Point, cv::Rect>> detectPeopleYolo(cv::Mat& frame, cv::dnn::Net& net, 
                                                           const std::vector<cv::String>& output_layers,
                                                           const std::vector<std::string>& classes,
                                                           float conf_threshold = 0.5, 
                                                           float nms_threshold = 0.4) {
    int height = frame.rows;
    int width = frame.cols;
    std::vector<std::pair<cv::Point, cv::Rect>> people;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Создаем блоб с уменьшенным размером для ускорения
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

// Применение эффектов к кадру
cv::Mat applyEffects(cv::Mat& frame, cv::Mat& memory_dump) {
    // 1. Применение красных оттенков
    cv::Mat processed;
    
    if (frame.channels() == 3) {
        // Разделяем каналы
        std::vector<cv::Mat> channels;
        cv::split(frame, channels);
        
        // Создаем матрицу только с красным каналом
        std::vector<cv::Mat> red_channels(3);
        red_channels[0] = cv::Mat::zeros(frame.size(), CV_8UC1);  // Синий канал - нули
        red_channels[1] = cv::Mat::zeros(frame.size(), CV_8UC1);  // Зеленый канал - нули
        red_channels[2] = channels[2];  // Красный канал - из оригинала
        
        cv::Mat red_channel;
        cv::merge(red_channels, red_channel);
        
        // Смешиваем с оригиналом
        cv::addWeighted(frame, 0.3, red_channel, 0.7, 0, processed);
    } else {
        // Если изображение не трехканальное, просто копируем
        frame.copyTo(processed);
    }
    
    // 2. Эффект кинескопа (только к видео)
    for (int i = 1; i < processed.rows; i += 2) {
        processed.row(i) = cv::Scalar(0, 0, 0);
    }
    
    // 3. Конвертируем в BGRA для наложения прозрачности
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
    
    // 5. Конвертируем обратно в BGR
    cv::Mat result;
    cv::cvtColor(processed_bgra, result, cv::COLOR_BGRA2BGR);
    
    return result;
}

// Класс для записи видео
class VideoRecorder {
public:
    VideoRecorder(const std::string& video_dir) : video_dir_(video_dir), current_date_("") {}
    ~VideoRecorder() {
        stopRecording();
    }
    
    void startRecording() {
        std::lock_guard<std::mutex> lock(video_mutex);
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
        
        if (!writer_.isOpened()) {
            std::cerr << "Не удалось открыть VideoWriter для записи: " << filename << std::endl;
        }
    }
    
    void stopRecording() {
        std::lock_guard<std::mutex> lock(video_mutex);
        if (writer_.isOpened()) {
            writer_.release();
        }
    }
    
    void writeFrame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(video_mutex);
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
        std::lock_guard<std::mutex> lock(video_mutex);
        return writer_.isOpened();
    }
    
    std::string getCurrentVideoPath() const {
        std::lock_guard<std::mutex> lock(video_mutex);
        if (current_date_.empty()) return "";
        return video_dir_ + "/" + current_date_ + ".mp4";
    }
    
    std::string getVideoPathForDate(const std::string& date) const {
        return video_dir_ + "/" + date + ".mp4";
    }

private:
    std::string video_dir_;
    cv::VideoWriter writer_;
    std::string current_date_;
};

// Обработчик сигналов
void signalHandler(int signal) {
    running = false;
}

// Функция обработки видео
void video_processing_thread(TgBot::Bot* bot, VideoRecorder& recorder, Settings& settings) {
    // Очистка старых видеофайлов при старте
    cleanupOldVideos();
    
    // Инициализация StartupManager
    StartupManager startup_manager(WIDTH, HEIGHT);
    
    // Запускаем инициализацию в отдельном потоке
    std::thread init_thread([&]() {
        startup_manager.initialize();
    });
    init_thread.detach();
    
    // Отображение анимации инициализации
    auto system_start_time = std::chrono::steady_clock::now();
    
    // ПЕРВАЯ ВАЖНАЯ ИЗМЕНЕНИЕ: продолжаем анимацию, пока инициализация не завершена
    while (!startup_manager.isInitializationComplete() || startup_manager.getAnimationPhase() < 1) {
        cv::Mat anim_frame = startup_manager.updateAnimation();
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            anim_frame.copyTo(last_frame);
        }
        
        // Записываем анимацию инициализации
        if (!recorder.isRecording()) {
            recorder.startRecording();
        }
        
        // Применяем эффекты к анимации инициализации перед записью
        cv::Mat memory_dump = generateMemoryDump(WIDTH, HEIGHT);
        cv::Mat processed_anim = applyEffects(anim_frame, memory_dump);
        recorder.writeFrame(processed_anim);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    
    // Получаем компоненты для обработки
    cv::VideoCapture& cap = startup_manager.getCapture();
    cv::dnn::Net net = startup_manager.getNet();
    std::vector<std::string> classes = startup_manager.getClasses();
    std::vector<cv::String> output_layers = startup_manager.getOutputLayers();
    cv::CascadeClassifier face_cascade = startup_manager.getFaceCascade();
    
    // Инициализация трекера
    ObjectTracker tracker(3);
    int frame_counter = 0;
    double last_dump_time = 0.0;
    cv::Mat memory_dump = generateMemoryDump(WIDTH, HEIGHT);
    
    auto start_time = std::chrono::steady_clock::now();
    double last_detection_time = 0.0;
    
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
        
        // Подготовка кадра для отображения
        cv::Mat small_frame;
        cv::resize(frame, small_frame, cv::Size(640, 480));
        cv::Mat display_frame;
        cv::resize(small_frame, display_frame, cv::Size(WIDTH, HEIGHT));
        
        // 1. Применение эффектов
        cv::Mat processed = applyEffects(display_frame, memory_dump);
        
        // 2. Генерация дампа памяти раз в секунду
        if (current_time - last_dump_time > 1.0) {
            memory_dump = generateMemoryDump(WIDTH, HEIGHT);
            last_dump_time = current_time;
        }
        
        // 3. Детекция людей (используем YOLO только каждый 3-й кадр)
        std::vector<std::pair<cv::Point, cv::Rect>> detections;
        if (frame_counter % 3 == 0 && !net.empty()) {
            try {
                detections = detectPeopleYolo(
                    small_frame, 
                    net, 
                    output_layers, 
                    classes,
                    CONF_THRESHOLD,
                    NMS_THRESHOLD
                );
            } catch (const cv::Exception& e) {
                std::cerr << "Ошибка детекции людей: " << e.what() << std::endl;
            }
        }
        
        // 4. Обновляем трекер
        std::map<int, std::pair<cv::Point, cv::Rect>> tracked_objects = tracker.update(detections);
        
        // 5. Детекция лиц
        std::vector<cv::Rect> detected_faces;
        if (!face_cascade.empty() && !tracked_objects.empty()) {
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
                    face_cascade.detectMultiScale(
                        roi, 
                        faces, 
                        1.05,
                        5,
                        0,
                        cv::Size(40, 40)
                    );
                    
                    for (const auto& face : faces) {
                        detected_faces.push_back(cv::Rect(roi_x + face.x, roi_y + face.y, face.width, face.height));
                    }
                }
            }
        }
        
        // 6. Отрисовка результатов
        for (const auto& obj : tracked_objects) {
            int object_id = obj.first;
            cv::Point centroid = obj.second.first;
            cv::Rect bbox = obj.second.second;
            
            // Масштабирование координат
            int x = static_cast<int>(bbox.x * WIDTH / 640.0);
            int y = static_cast<int>(bbox.y * HEIGHT / 480.0);
            int w = static_cast<int>(bbox.width * WIDTH / 640.0);
            int h = static_cast<int>(bbox.height * HEIGHT / 480.0);
            centroid = cv::Point(
                static_cast<int>(centroid.x * WIDTH / 640.0), 
                static_cast<int>(centroid.y * HEIGHT / 480.0)
            );
            
            // Рисуем контур человека
            cv::rectangle(processed, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(0, 0, 255), 2);
            
            // Информационная панель
            if (w > 60 && h > 100) {
                cv::rectangle(processed, cv::Point(x, y - 20), cv::Point(x + 100, y), cv::Scalar(0, 0, 0), -1);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distr(85, 99);
                cv::putText(processed, "Human: " + std::to_string(distr(gen)) + "%", 
                           cv::Point(x, y - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);
            }
            
            // Перекрестие
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
        
        // 7. Звук обнаружения
        // ВТОРАЯ ВАЖНАЯ ИЗМЕНЕНИЕ: добавляем звук обнаружения
        if (!tracked_objects.empty() && frame_counter % 5 == 0) {
            playDetectSound();
        }
        
        // 8. Информационный HUD
        cv::rectangle(processed, cv::Point(WIDTH - 180, 10), cv::Point(WIDTH - 10, 70), cv::Scalar(0, 0, 0, 200), -1);
        double fps = 1.0 / (std::chrono::duration<double>(loop_start - start_time).count() + 0.001);
        cv::putText(processed, "TARGETS: " + std::to_string(tracked_objects.size()), 
                   cv::Point(WIDTH - 170, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
        cv::putText(processed, "FPS: " + std::to_string(static_cast<int>(fps)), 
                   cv::Point(WIDTH - 170, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1);
        
        // Обновляем last_frame
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            processed.copyTo(last_frame);
        }
        
        // Запись видео при обнаружении людей
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
                    fs::remove(tmp_path);
                }
            }
        }
        
        // Запись видео в режиме unstopable
        if (unstopable_mode) {
            if (!recorder.isRecording()) {
                recorder.startRecording();
            }
            recorder.writeFrame(processed);  // ЗАПИСЫВАЕМ ОБРАБОТАННЫЙ КАДР
        }
        // Обычная запись: при обнаружении людей и в течение 60 секунд после
        else if (recorder.isRecording()) {
            recorder.writeFrame(processed);  // ЗАПИСЫВАЕМ ОБРАБОТАННЫЙ КАДР
            if (tracked_objects.empty() && (current_time - last_detection_time > RECORDING_DURATION) && 
                system_uptime > INITIAL_RECORDING_DURATION) {
                recorder.stopRecording();
            }
        }
        
        // Всегда записываем первые 60 секунд после старта
        if (system_uptime < INITIAL_RECORDING_DURATION && !recorder.isRecording()) {
            recorder.startRecording();
        }
        
        frame_counter++;
        
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start).count();
        if (elapsed < 40) {
            std::this_thread::sleep_for(std::chrono::milliseconds(40 - elapsed));
        }
        
        // Периодическая очистка старых видеофайлов (раз в час)
        static auto last_cleanup = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(loop_end - last_cleanup).count() > 3600) {
            cleanupOldVideos();
            last_cleanup = loop_end;
        }
    }
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Определяем путь к исполняемому файлу
    char path[1024];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
        path[len] = '\0';
        resource_dir = fs::path(path).parent_path().string();
        std::cout << "Ресурсы ищем в: " << resource_dir << std::endl;
    } else {
        // Если не удалось определить путь, используем текущую директорию
        resource_dir = ".";
        std::cout << "Не удалось определить путь к исполняемому файлу, используем текущую директорию" << std::endl;
    }
    
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
    unstopable_mode = settings.unstopable_mode;  // Загружаем состояние режима unstopable
    
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
        
        // Сначала проверяем текущий день
        auto now = std::chrono::system_clock::now();
        time_t now_c = std::chrono::system_clock::to_time_t(now);
        tm now_tm = *std::localtime(&now_c);
        char buf[20];
        strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
        std::string today = buf;
        
        std::string video_path = recorder.getVideoPathForDate(today);
        
        // Если текущий день не найден, ищем последний существующий файл
        if (!fs::exists(video_path) || fs::file_size(video_path) <= 48) {
            for (int i = 1; i <= 7; ++i) {
                auto date = now - std::chrono::hours(24 * i);
                time_t date_c = std::chrono::system_clock::to_time_t(date);
                tm date_tm = *std::localtime(&date_c);
                char date_buf[20];
                strftime(date_buf, sizeof(date_buf), "%Y%m%d", &date_tm);
                std::string date_str = date_buf;
                
                std::string path = recorder.getVideoPathForDate(date_str);
                if (fs::exists(path) && fs::file_size(path) > 48) {
                    video_path = path;
                    break;
                }
            }
        }
        
        if (video_path.empty() || !fs::exists(video_path) || fs::file_size(video_path) <= 48) {
            bot.getApi().sendMessage(message->chat->id, "ostringstream Нет записанных видео");
        } else {
            try {
                // Убедимся, что файл закрыт перед отправкой
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
        
        // Определяем тип записи
        if (unstopable_mode) {
            status += "Режим: 🔄 непрерывная запись\n";
        } else {
            if (recorder.isRecording()) {
                status += "Режим: 🎯 запись при обнаружении\n";
            } else {
                status += "Режим: 🎯 запись при обнаружении\n";
            }
        }
        
        status += "Детекция: " + std::string("✅ активна (всегда)") + "\n";
        status += "Люди в кадре: " + std::string(detection_active ? "⚠️ обнаружены" : "✅ не обнаружены") + "\n";
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