#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>


namespace Startup {

class StartupManager {
public:

   // @todo Maybe width&height in init?
  StartupManager(int width, int height)
      : width_(width), height_(height), center_x_(width / 2),
        center_y_(height / 2), circle_radius_(30), circle_distance_(120),
        animation_phase_(0), start_time_(std::chrono::steady_clock::now()),
        initialization_complete_(false), initialization_progress_(0.0f),
        cam_index_(-1), loading_text_("INITIALIZING SYSTEM"),
        pulse_value_(0.0f), camera_error_count_(0) {

    // Создаем 3 круга под углом 120 градусов
    for (int i = 0; i < 3; ++i) {
      float angle = 2 * M_PI * i / 3;
      int x = center_x_ + static_cast<int>(circle_distance_ * cos(angle));
      int y = center_y_ + static_cast<int>(circle_distance_ * sin(angle));
      circles_.push_back({x, y, 0, 1});
    }
  }

  int getAnimationPhase() const { return animation_phase_; }

  cv::Mat updateAnimation() {
    auto elapsed = std::chrono::duration<float>(
                       std::chrono::steady_clock::now() - start_time_)
                       .count();
    cv::Mat anim_frame = cv::Mat::zeros(height_, width_, CV_8UC3);
    float pulse_speed = 0.5f; // Скорость пульсации

    // Фаза 0: Пульсация кругов во время инициализации
    if (animation_phase_ == 0) {
      // Пульсация кругов
      pulse_value_ = sin(elapsed * pulse_speed * 2 * M_PI) * 0.2f + 0.8f;
      int pulse_size = static_cast<int>(circle_radius_ * pulse_value_);

      for (auto &circle : circles_) {
        // Отрисовка круга
        cv::circle(anim_frame, cv::Point(circle.x, circle.y), pulse_size,
                   cv::Scalar(255, 255, 255), -1);
      }

      // Отображение прогресса инициализации
      std::string progress_text =
          loading_text_ + ": " +
          std::to_string(static_cast<int>(initialization_progress_ * 100)) +
          "%";
      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.7;
      int thickness = 2;
      int baseline;
      cv::Size text_size = cv::getTextSize(progress_text, font_face, font_scale,
                                           thickness, &baseline);
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
      elapsed = elapsed - 0.5f; // Задержка перед началом анимации
      // Параметры анимации
      float text_alpha = 0.0f;
      int bar_width_left = 0;
      int bar_width_right = 0;
      float video_alpha = 0.0f;
      int max_bar_width = width_ / 2 - circle_distance_;

      // Фаза 1.1: Вращение кругов и появление полос (0-1.2 сек)
      if (elapsed < 1.2f) {
        float rotation = elapsed * 1.5f; // Быстрое вращение
        for (size_t i = 0; i < circles_.size(); ++i) {
          float angle = 2 * M_PI * i / 3 + rotation;
          circles_[i].x =
              center_x_ + static_cast<int>(circle_distance_ * cos(angle));
          circles_[i].y =
              center_y_ + static_cast<int>(circle_distance_ * sin(angle));
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
        for (auto &circle : circles_) {
          // Плавное перемещение к центру
          circle.x =
              static_cast<int>(circle.x + (center_x_ - circle.x) * progress);
          circle.y =
              static_cast<int>(circle.y + (center_y_ - circle.y) * progress);
        }
        // Появление видео
        video_alpha = std::min(1.0f, (elapsed - 1.5f) / 0.5f);
      }

      // Отрисовка кругов
      float progress = std::min(1.0f, (elapsed - 1.2f) / 0.8f);
      for (auto &circle : circles_) {
        int radius =
            static_cast<int>(circle_radius_ * (1 - progress) * pulse_value_);
        cv::circle(anim_frame, cv::Point(circle.x, circle.y), radius,
                   cv::Scalar(255, 255, 255), -1);
      }

      // Отрисовка полос
      if (bar_width_left > 0) {
        // Левая полоса
        cv::rectangle(anim_frame, cv::Point(0, center_y_ - 2),
                      cv::Point(bar_width_left, center_y_ + 2),
                      cv::Scalar(255, 255, 255), -1);
        // Правая полоса
        cv::rectangle(
            anim_frame, cv::Point(width_ - bar_width_right, center_y_ - 2),
            cv::Point(width_, center_y_ + 2), cv::Scalar(255, 255, 255), -1);
      }

      // Отрисовка текста
      if (text_alpha > 0) {
        std::string text = "JetVision Systems";
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        int thickness = 2;
        int baseline;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        int text_x = (width_ - text_size.width) / 2;
        int text_y = height_ - 50;
        cv::Scalar color(255 * text_alpha, 255 * text_alpha, 255 * text_alpha);
        cv::putText(anim_frame, text, cv::Point(text_x, text_y), font_face,
                    font_scale, color, thickness);
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
        cv::addWeighted(video_frame, video_alpha,
                        cv::Mat::zeros(height_, width_, CV_8UC3),
                        1 - video_alpha, 0, weighted_video);
        cv::addWeighted(anim_frame, 1 - video_alpha,
                        cv::Mat::zeros(height_, width_, CV_8UC3), video_alpha,
                        0, weighted_anim);
        anim_frame = weighted_video + weighted_anim;
      }
    }

    return anim_frame;
  }

  void initialize(const std::string& resource_dir, const int camera_index, const int fps) {
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
      std::cerr << "Не удалось загрузить классы из: " << classes_path
                << std::endl;
      classes_ = {"person"}; // Минимальный набор
    }

    // Шаг 2: Инициализация YOLO
    loading_text_ = "LOADING OBJECT DETECTOR";
    try {
      // Пробуем загрузить YOLO-tiny
      std::string weights_path = resource_dir + "/yolov3-tiny.weights";
      std::string cfg_path = resource_dir + "/yolov3-tiny.cfg";

      std::cout << "Попытка загрузить YOLO из: " << weights_path << " и "
                << cfg_path << std::endl;

      // Проверяем существование файлов
      if (!std::filesystem::exists(weights_path)) {
        std::cerr << "Файл весов не найден: " << weights_path << std::endl;
        throw std::runtime_error("Weights file not found");
      }
      if (!std::filesystem::exists(cfg_path)) {
        std::cerr << "Конфигурационный файл не найден: " << cfg_path
                  << std::endl;
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
        output_layers_.push_back(layer_names[i - 1]);
      }

      initialization_progress_ = 0.6f;
    } catch (const cv::Exception &e) {
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
        "/usr/local/share/opencv4/haarcascades/"
        "haarcascade_frontalface_default.xml",
        "/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml"};

    bool cascade_loaded = false;
    for (const auto &path : cascade_paths) {
      if (std::filesystem::exists(path) && face_cascade_.load(path)) {
        cascade_loaded = true;
        std::cout << "Детектор лиц загружен из: " << path << std::endl;
        break;
      } else if (std::filesystem::exists(path)) {
        std::cout << "Файл каскада найден, но не загружен: " << path
                  << std::endl;
      }
    }

    if (!cascade_loaded) {
      std::cerr << "Не удалось загрузить детектор лиц ни из одного пути"
                << std::endl;
    } else {
      initialization_progress_ = 0.8f;
    }

    // Шаг 4: Открытие камеры
    loading_text_ = "CONNECTING CAMERA";
    cap_.open(camera_index, cv::CAP_V4L);
    if (cap_.isOpened()) {
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
      cap_.set(cv::CAP_PROP_FPS, fps);

      // Получаем первый кадр
      cv::Mat frame;
      if (cap_.read(frame)) {
        cv::resize(frame, frame_, cv::Size(width_, height_));
      }
    }

    initialization_progress_ = 1.0f;
    initialization_complete_ = true;
  }

  bool isInitializationComplete() const { return initialization_complete_; }

  cv::Mat getFrame() const { return frame_; }

  cv::dnn::Net getNet() const { return net_; }

  std::vector<std::string> getClasses() const { return classes_; }

  std::vector<cv::String> getOutputLayers() const { return output_layers_; }

  cv::CascadeClassifier getFaceCascade() const { return face_cascade_; }

  cv::VideoCapture &getCapture() { return cap_; }

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

}; // namespace Startup