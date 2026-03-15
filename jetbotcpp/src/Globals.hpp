#pragma once
#include <atomic>
#include <string>
#include <opencv2/core.hpp>

// Флаги конфигурации (доступны из командной строки)
extern std::atomic<bool> flag_recording;
extern std::atomic<bool> flag_detection;
extern std::atomic<bool> flag_effects;
extern std::atomic<bool> flag_sound;
extern std::atomic<bool> flag_cleanup;
extern std::atomic<bool> flag_animation;
extern std::atomic<bool> flag_face;

// Состояние системы
extern std::atomic<bool> running;
extern std::atomic<bool> alert_enabled;
extern std::atomic<bool> detection_active;
extern std::atomic<bool> unstopable_mode;
extern std::atomic<int64_t> authorizedUserId;

// Последние кадры (захваченные, обработанные, с распознаванием)
extern cv::Mat last_frame;
extern cv::Mat last_raw_frame;
extern cv::Mat last_recognition_frame;
extern std::mutex frame_mutex;
extern std::mutex settings_mutex;

// Директории и пути
extern std::string jetbot_dir;
extern std::string video_dir;
extern std::string logs_dir;
extern std::string settings_path;
extern std::string resource_dir;

extern const std::string VK_ACCESS_TOKEN;
extern const std::string VK_GROUP_ID;

// Параметры камеры и детекции (константы)
const int WIDTH = 640;
const int HEIGHT = 480;
extern const int CAMERA_INDEX;
extern const std::string TG_API_TOKEN;
extern const std::string ONNX_MODEL_PATH;
extern const std::string COCO_NAMES_PATH;
const int FPS = 25;
const float CONF_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.4f;
const int RECORDING_DURATION = 60;
const int INITIAL_RECORDING_DURATION = 60;