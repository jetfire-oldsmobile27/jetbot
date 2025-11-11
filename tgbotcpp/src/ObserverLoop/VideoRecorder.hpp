#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace ObserverLoop {

class VideoRecorder {
private:
    mutable std::mutex video_mutex;
public:
  VideoRecorder(const std::string &video_dir)
      : video_dir_(video_dir), current_date_("") {}
  ~VideoRecorder() { stopRecording(); }

  void startRecording() {
    std::lock_guard<std::mutex> lock(video_mutex);
    if (writer_.isOpened())
      return;

    std::chrono::time_point now = std::chrono::system_clock::now();
    time_t now_c = std::chrono::system_clock::to_time_t(now);
    tm now_tm = *std::localtime(&now_c);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
    current_date_ = buf;

    std::string filename = video_dir_ + "/" + current_date_ + ".mp4";
    writer_.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 25.0,
                 cv::Size(640, 480));

    if (!writer_.isOpened()) {
      writer_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25.0,
                   cv::Size(640, 480));
    }

    if (!writer_.isOpened()) {
      std::cerr << "Не удалось открыть VideoWriter для записи: " << filename
                << std::endl;
    }
  }

  void stopRecording() {
    std::lock_guard<std::mutex> lock(video_mutex);
    if (writer_.isOpened()) {
      writer_.release();
    }
  }

  void writeFrame(const cv::Mat &frame) {
    std::lock_guard<std::mutex> lock(video_mutex);
    if (!writer_.isOpened())
      return;

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
      writer_.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 25.0,
                   cv::Size(640, 480));

      if (!writer_.isOpened()) {
        writer_.open(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                     25.0, cv::Size(640, 480));
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
    if (current_date_.empty())
      return "";
    return video_dir_ + "/" + current_date_ + ".mp4";
  }

  std::string getVideoPathForDate(const std::string &date) const {
    return video_dir_ + "/" + date + ".mp4";
  }

private:
  std::string video_dir_;
  cv::VideoWriter writer_;
  std::string current_date_;
};
}; // namespace ObserverLoop