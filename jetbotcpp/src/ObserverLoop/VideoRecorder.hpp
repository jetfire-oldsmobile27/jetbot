#pragma once
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include "Logging.hpp"

namespace ObserverLoop {

class VideoRecorder {
private:
    mutable std::mutex video_mutex;
    std::string video_dir_;
    cv::VideoWriter writer_;
    std::string current_video_path_;
    std::chrono::steady_clock::time_point segment_start_time_;
    static constexpr int SEGMENT_DURATION_SEC = 300; // 5 минут
    
    cv::Size frame_size_;
    bool use_mjpg_;

    std::string generateFilename(const std::string& prefix = "segment") {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_c);
        
        std::ostringstream oss;
        oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S");
        
        // Создаем подпапку по дате
        std::string date_folder;
        std::ostringstream date_oss;
        date_oss << std::put_time(&now_tm, "%Y%m%d");
        date_folder = date_oss.str();
        
        std::filesystem::path full_dir = std::filesystem::path(video_dir_) / date_folder;
        std::filesystem::create_directories(full_dir);
        
        return (full_dir / (prefix + "_" + oss.str() + ".avi")).string();
    }

    bool openWriter(const cv::Size& frame_size) {
        // Пробуем разные кодеки в порядке надежности
        std::vector<std::pair<std::string, int>> codecs_to_try = {
            {"MJPG", cv::VideoWriter::fourcc('M', 'J', 'P', 'G')},
            {"XVID", cv::VideoWriter::fourcc('X', 'V', 'I', 'D')},
            {"DIVX", cv::VideoWriter::fourcc('D', 'I', 'V', 'X')},
            {"I420", cv::VideoWriter::fourcc('I', '4', '2', '0')} // YUV, самый надежный
        };
        
        for (const auto& [codec_name, fourcc_val] : codecs_to_try) {
            writer_.open(current_video_path_, fourcc_val, 25.0, frame_size);
            if (writer_.isOpened()) {
                logMsg(std::format("✅ Кодек {} успешно открыт для записи:  {}", codec_name, current_video_path_));
                use_mjpg_ = (codec_name == "MJPG");
                return true;
            }
        }
        
        // Если ни один кодек не сработал, пробуем без указания кодека
        writer_.open(current_video_path_, cv::CAP_ANY, 25.0, frame_size);
        if (writer_.isOpened()) {
            logMsg(std::format("⚠️  Запись открыта с кодеком по умолчанию:  {}", current_video_path_));
            return true;
        }
        logMsg("❌ Не удалось открыть VideoWriter ни с одним кодеком!");
        return false;
    }

public:
    VideoRecorder(const std::string &video_dir) 
        : video_dir_(video_dir), 
          frame_size_(640, 480),
          use_mjpg_(false),
          segment_start_time_(std::chrono::steady_clock::now()) {
        
        // Создаем основную директорию
        std::filesystem::create_directories(video_dir_);
    }
    
    ~VideoRecorder() { 
        stopRecording(); 
    }

    void startRecording(const cv::Size& frame_size = cv::Size(640, 480)) {
        std::lock_guard<std::mutex> lock(video_mutex);
        
        if (writer_.isOpened()) {
            return;
        }
        
        frame_size_ = frame_size;
        current_video_path_ = generateFilename();
        
        if (!openWriter(frame_size)) {
            std::cerr << "ОШИБКА: Не удалось начать запись!" << std::endl;
            return;
        }
        
        segment_start_time_ = std::chrono::steady_clock::now();
        logMsg(std::format("🎥 Начата запись: {}", current_video_path_));
    }

    void writeFrame(const cv::Mat &frame) {
        std::lock_guard<std::mutex> lock(video_mutex);
        
        if (!writer_.isOpened()) {
            return;
        }
        
        // Проверяем размер кадра
        if (frame.size() != frame_size_) {
            cv::Mat resized;
            cv::resize(frame, resized, frame_size_);
            writer_.write(resized);
        } else {
            writer_.write(frame);
        }
        
        // Проверяем, не пора ли начать новый сегмент
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - segment_start_time_).count();
        
        if (elapsed >= SEGMENT_DURATION_SEC) {
            // Закрываем текущий файл и начинаем новый
            std::string old_path = current_video_path_;
            writer_.release();
            logMsg(std::format("🔄 Завершен сегмент: {} (длительность: {}сек)", old_path, elapsed));
            
            startRecording(frame_size_);
        }
    }

    void stopRecording() {
        std::lock_guard<std::mutex> lock(video_mutex);
        if (writer_.isOpened()) {
            writer_.release();
            logMsg(std::format("⏹️  Запись завершена: {}", current_video_path_));
        }
    }

    bool isRecording() const {
        std::lock_guard<std::mutex> lock(video_mutex);
        return writer_.isOpened();
    }

    std::string getCurrentVideoPath() const {
        std::lock_guard<std::mutex> lock(video_mutex);
        return current_video_path_;
    }

    std::string getVideoPathForDate(const std::string &date) const {
        // Ищем последний файл в папке с указанной датой
        std::filesystem::path date_dir = std::filesystem::path(video_dir_) / date;
        
        if (!std::filesystem::exists(date_dir)) {
            return "";
        }
        
        // Находим самый свежий файл
        std::string latest_file;
        std::filesystem::file_time_type latest_time;
        
        for (const auto& entry : std::filesystem::directory_iterator(date_dir)) {
            if (entry.path().extension() == ".avi" || entry.path().extension() == ".mp4") {
                auto write_time = std::filesystem::last_write_time(entry);
                if (latest_file.empty() || write_time > latest_time) {
                    latest_file = entry.path().string();
                    latest_time = write_time;
                }
            }
        }
        
        return latest_file;
    }
};
} // namespace ObserverLoop