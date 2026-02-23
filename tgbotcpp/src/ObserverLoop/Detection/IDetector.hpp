#pragma once
#include <expected>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ObserverLoop::Detection {
struct DetectionResult {

  float confidence_;
  cv::Rect bbox_;
  std::string label_;

  DetectionResult(const std::string &lbl = "", float conf = 0.0f,
                  const cv::Rect &box = cv::Rect())
      : label_(lbl), confidence_(conf), bbox_(box) {}
};

enum class GenericDetectorError {
  SUCCESS = 0,
  INIT_FAILED = -999,
  BAD_PATH,
  BAD_ONNX,
  NOT_INITIALIZED,
  DETECTION_FAILED,
  DRAWING_FAILED,
  EMPTY_FRAME,
  MODEL_NOT_LOADED,
  INFERENCE_ERROR
};

class IDetector {
public:
  virtual ~IDetector() = default;

  virtual auto initModel(const std::string &modelPath,
                         const std::string &labelsPath = "")
      -> std::expected<bool, GenericDetectorError> = 0;
  virtual auto isModelLoaded() const
      -> std::expected<bool, GenericDetectorError> = 0;

  virtual auto detect(cv::Mat &frame)
      -> std::expected<std::vector<DetectionResult>, GenericDetectorError> = 0;
  virtual auto drawDetections(cv::Mat &frame,
                              const std::vector<DetectionResult> &detections)
      -> std::expected<void, GenericDetectorError> = 0;
};
} // namespace ObserverLoop::Detection