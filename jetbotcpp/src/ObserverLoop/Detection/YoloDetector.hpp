#pragma once
#include "IDetector.hpp"
#include <expected>
#include <memory>

namespace ObserverLoop::Detection {

class YoloDetector : public IDetector {
public:
  static auto create(const std::string &modelPath,
                     const std::string &labelsPath = "")
      -> std::expected<std::unique_ptr<YoloDetector>, GenericDetectorError>;

  YoloDetector(const YoloDetector &) = delete;
  YoloDetector &operator=(const YoloDetector &) = delete;
  YoloDetector(YoloDetector &&) = delete;
  YoloDetector &operator=(YoloDetector &&) = delete;

  ~YoloDetector() override;

  auto initModel(const std::string &modelPath,
                 const std::string &labelsPath = "")
      -> std::expected<bool, GenericDetectorError> override;
  auto isModelLoaded() const
      -> std::expected<bool, GenericDetectorError> override;

  auto detect(cv::Mat &frame) -> std::expected<std::vector<DetectionResult>,
                                               GenericDetectorError> override;
  auto drawDetections(cv::Mat &frame,
                      const std::vector<DetectionResult> &detections)
      -> std::expected<void, GenericDetectorError> override;

private:
  YoloDetector(const std::string &modelPath, const std::string &labelsPath);
  struct Impl;
  std::unique_ptr<Impl> pImpl_;
};
} // namespace ObserverLoop::Detection

