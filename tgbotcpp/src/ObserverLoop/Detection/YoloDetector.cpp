// cpp:
#include "YoloDetector.hpp"
#include "Utility/thirdparty/yolo-cpp/yolos/tasks/detection.hpp"

namespace ObserverLoop::Detection {
struct YoloDetector::Impl {
  std::unique_ptr<yolos::det::YOLODetector> detector;
  std::vector<std::string> classNames;
  cv::Mat frame_with_bbox_;

  Impl(const std::string &modelPath, const std::string &labelsPath, bool useGPU)
      : detector(
            std::make_unique<yolos::det::YOLODetector>(modelPath, labelsPath, useGPU)) {
    if (!labelsPath.empty()) {
      loadClassNames(labelsPath);
    }
  }

  void loadClassNames(const std::string &labelsPath) {
    std::ifstream file(labelsPath);
    if (!file.is_open()) {
      printf("Failed to open labels file: %s", labelsPath.c_str());
      return;
    }

    classNames.clear();
    std::string line;
    while (std::getline(file, line)) {
      line.erase(std::remove_if(line.begin(), line.end(), ::isspace),
                 line.end());
      if (!line.empty()) {
        classNames.push_back(line);
      }
    }
    printf("Loaded %zu class names from %s", classNames.size(),
         labelsPath.c_str());
  }

  void drawOptimizedBoundingBoxes(cv::Mat &frame,
                                  const std::vector<yolos::det::Detection> &detections) {
    if (frame.empty() || detections.empty())
      return;

    const double minFontScale = 0.4;
    const double maxFontScale = 0.7;
    const int minThickness = 1;
    const int maxThickness = 2;
    double fontScale = std::clamp(std::min(frame.rows, frame.cols) * 0.001,
                                  minFontScale, maxFontScale);
    int thickness =
        std::clamp(static_cast<int>(std::min(frame.rows, frame.cols) * 0.0015),
                   minThickness, maxThickness);

    for (const auto &det : detections) {
      std::string label = "unknown";
      if (det.classId >= 0 &&
          det.classId < static_cast<int>(classNames.size())) {
        label = classNames[det.classId];
      } else if (det.classId >= 0) {
        label = "class_" + std::to_string(det.classId);
      }

      std::string display_text =
          label + " " + std::to_string(static_cast<int>(det.conf * 100)) + "%";

      cv::Scalar color;
      if (classNames.empty() || det.classId < 0 ||
          det.classId >= static_cast<int>(classNames.size())) {
        color = cv::Scalar(0, 255, 255); // желтый для неизвестных
      } else {
        int r = (det.classId * 37) % 255;
        int g = (det.classId * 59) % 255;
        int b = (det.classId * 73) % 255;
        color = cv::Scalar(b, g, r);
      }

      cv::Rect bbox(det.box.x, det.box.y, det.box.width, det.box.height);

      bbox.x = std::max(0, bbox.x);
      bbox.y = std::max(0, bbox.y);
      bbox.width = std::min(bbox.width, frame.cols - bbox.x);
      bbox.height = std::min(bbox.height, frame.rows - bbox.y);

      if (bbox.width <= 0 || bbox.height <= 0)
        continue;

      cv::rectangle(frame, bbox, color, thickness + 1, cv::LINE_AA);

      int baseline = 0;
      cv::Size text_size =
          cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                          thickness, &baseline);

      cv::Point text_org(bbox.x, bbox.y - 5);
      if (text_org.y < text_size.height + 5) {
        text_org.y = bbox.y + bbox.height + text_size.height + 5;
        if (text_org.y > frame.rows - 5)
          text_org.y = frame.rows - 5;
      }

      cv::Point top_left(text_org.x, text_org.y - text_size.height - 5);
      cv::Point bottom_right(text_org.x + text_size.width + 5, text_org.y + 2);

      top_left.x = std::max(0, top_left.x);
      top_left.y = std::max(0, top_left.y);
      bottom_right.x = std::min(frame.cols - 1, bottom_right.x);
      bottom_right.y = std::min(frame.rows - 1, bottom_right.y);

      if (top_left.x < bottom_right.x && top_left.y < bottom_right.y) {
        cv::rectangle(frame, top_left, bottom_right, cv::Scalar(0, 0, 0, 180),
                      -1);
        cv::putText(frame, display_text, cv::Point(text_org.x, text_org.y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
      }
    }
  };

}; // YoloDetector::Impl

YoloDetector::YoloDetector(const std::string &modelPath, const std::string &labelsPath)
    : pImpl_(std::make_unique<Impl>(modelPath, labelsPath, /*useGPU*/ false)) {}

YoloDetector::~YoloDetector() = default;

auto YoloDetector::create(const std::string &modelPath, const std::string &labelsPath)
    -> std::expected<std::unique_ptr<YoloDetector>, GenericDetectorError>
{
    printf("==> Creating YOLO detector with model: %s\n", modelPath.c_str());
    
    if (modelPath.empty()) {
        printf("ERROR: Empty model path\n");
        return std::unexpected(GenericDetectorError::BAD_PATH);
    }
    
    printf("Will use GPU: true\n");
    
    try {
        return std::unique_ptr<YoloDetector>(new YoloDetector(modelPath, labelsPath));
    } catch (const std::exception &e) {
        printf("Failed to create YOLO detector: %s\n", e.what());
        return std::unexpected(GenericDetectorError::INIT_FAILED);
    }
}

auto YoloDetector::initModel(const std::string &modelPath, const std::string &labelsPath)
    -> std::expected<bool, GenericDetectorError>
{
    if (!pImpl_ || !pImpl_->detector) {
        return std::unexpected(GenericDetectorError::NOT_INITIALIZED);
    }
    
    if (pImpl_ && pImpl_->detector) {
        return true;
    }
    
    return std::unexpected(GenericDetectorError::MODEL_NOT_LOADED);
}

auto YoloDetector::isModelLoaded() const
    -> std::expected<bool, GenericDetectorError>
{
    if (!pImpl_) {
        return std::unexpected(GenericDetectorError::NOT_INITIALIZED);
    }
    
    if (!pImpl_->detector) {
        return std::unexpected(GenericDetectorError::MODEL_NOT_LOADED);
    }
    
    return true;
}

auto YoloDetector::detect(cv::Mat &frame)
    -> std::expected<std::vector<DetectionResult>, GenericDetectorError>
{
    auto loaded = isModelLoaded();
    if (!loaded) {
        return std::unexpected(loaded.error());
    }
    
    if (frame.empty()) {
        return std::unexpected(GenericDetectorError::EMPTY_FRAME);
    }
    
    printf("-> Processing frame: %d x %d\n", frame.cols, frame.rows);
    
    try {
        auto yoloDetections = pImpl_->detector->detect(frame);
        std::vector<DetectionResult> results;
        results.reserve(yoloDetections.size());
        
        for (const auto &det : yoloDetections) {
            cv::Rect bbox(det.box.x, det.box.y, det.box.width, det.box.height);
            std::string label;
            
            if (det.classId >= 0 && 
                det.classId < static_cast<int>(pImpl_->classNames.size())) {
                label = pImpl_->classNames[det.classId];
            } else {
                label = "class_" + std::to_string(det.classId);
            }
            
            results.emplace_back(label, det.conf, bbox);
        }
        
        
        printf("Detected %zu objects\n", results.size());
        
        return results;
        
    } catch (const std::exception &e) {
        printf("Detection failed: %s\n", e.what());
        return std::unexpected(GenericDetectorError::DETECTION_FAILED);
    }
}

auto YoloDetector::drawDetections(cv::Mat &frame,
                                  const std::vector<DetectionResult> &detections)
    -> std::expected<void, GenericDetectorError>
{
    auto loaded = isModelLoaded();
    if (!loaded) {
        return std::unexpected(loaded.error());
    }
    
    if (frame.empty()) {
        return std::unexpected(GenericDetectorError::EMPTY_FRAME);
    }
    
    try {
        if (detections.empty()) {
            return {}; // void (OK)
        }
        
        std::vector<yolos::det::Detection> yoloDetections;
        yoloDetections.reserve(detections.size());
        
        for (const auto &det : detections) {
            yolos::BoundingBox box(
                det.bbox_.x, det.bbox_.y, 
                det.bbox_.width, det.bbox_.height
            );
            
            int classId = 0;
            if (det.label_.find("class_") == 0) {
                try {
                    classId = std::stoi(det.label_.substr(6));
                } catch (...) {
                    auto it = std::find(pImpl_->classNames.begin(), 
                                       pImpl_->classNames.end(), det.label_);
                    if (it != pImpl_->classNames.end()) {
                        classId = std::distance(pImpl_->classNames.begin(), it);
                    }
                }
            } else {
                auto it = std::find(pImpl_->classNames.begin(), 
                                   pImpl_->classNames.end(), det.label_);
                if (it != pImpl_->classNames.end()) {
                    classId = std::distance(pImpl_->classNames.begin(), it);
                }
            }
            
            yoloDetections.emplace_back(box, det.confidence_, classId);
        }
        
        pImpl_->drawOptimizedBoundingBoxes(frame, yoloDetections);
        
        return {}; // void (OK)
        
    } catch (const std::exception &e) {
        printf("Drawing failed: %s\n", e.what());
        return std::unexpected(GenericDetectorError::DRAWING_FAILED);
    }
}

} // namespace ObserverLoop::Detection