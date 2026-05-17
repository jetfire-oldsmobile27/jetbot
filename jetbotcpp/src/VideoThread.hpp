#pragma once
#include <tgbot/Bot.h>

#include <chrono>
#include <expected>
#include <memory>
#include <opencv2/opencv.hpp>
#include <thread>

#include "Config.hpp"
#include "Effects.hpp"
#include "Globals.hpp"
#include "Logging.hpp"
#include "ObserverLoop/Detection/YoloDetector.hpp"
#include "ObserverLoop/VideoRecorder.hpp"
#include "Startup/StartupManager.hpp"
#include "Utility/Effects.hpp"
#include "Utility/Settings.hpp"

void video_processing_thread(TgBot::Bot* bot,
                             ObserverLoop::VideoRecorder& recorder,
                             Utility::Settings& settings,
                             const ConfigFlags& flags) {
  cleanupOldVideos();

  Startup::StartupManager startup_manager(WIDTH, HEIGHT);

  if (flags.animation) {
    std::thread init_thread(
        [&]() { startup_manager.initialize(resource_dir, CAMERA_INDEX, FPS); });
    init_thread.detach();

    while (!startup_manager.isInitializationComplete() ||
           startup_manager.getAnimationPhase() < 1) {
      cv::Mat anim_frame = startup_manager.updateAnimation();
      {
        std::lock_guard<std::mutex> lock(frame_mutex);
        anim_frame.copyTo(last_frame);
      }

      if (flags.recording && !recorder.isRecording()) {
        recorder.startRecording();
      }

      cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);
      cv::Mat processed_anim = flags.effects
                                   ? applyEffects(anim_frame, memory_dump)
                                   : anim_frame.clone();
      if (flags.recording) {
        recorder.writeFrame(processed_anim);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
  } else {
    startup_manager.initialize(resource_dir, CAMERA_INDEX, FPS);
  }

  cv::VideoCapture& cap = startup_manager.getCapture();
  cv::dnn::Net net = startup_manager.getNet();
  std::vector<std::string> classes = startup_manager.getClasses();
  std::vector<cv::String> output_layers = startup_manager.getOutputLayers();
  cv::CascadeClassifier face_cascade = startup_manager.getFaceCascade();

  std::unique_ptr<ObserverLoop::Detection::YoloDetector> det_ptr_;
  std::expected<std::unique_ptr<ObserverLoop::Detection::YoloDetector>,
                ObserverLoop::Detection::GenericDetectorError>
      yolo_det;
  if (flags.detection) {
    yolo_det = ObserverLoop::Detection::YoloDetector::create(ONNX_MODEL_PATH,
                                                             COCO_NAMES_PATH);
    if (yolo_det.has_value()) {
      det_ptr_ = std::move(*yolo_det);
      std::cout << "Detector created and moved successfully\n";
    } else {
      std::cerr << "Failed to create YOLO detector. Detection disabled.\n";
    }
  }
  // int frame_counter{0};
  double last_dump_time{0.0};
  cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);

  auto start_time = std::chrono::steady_clock::now();
  double last_detection_time{0.0};
  auto system_start_time = start_time;

  auto last_frame_time = std::chrono::steady_clock::now();
  while (running) {
    auto now = std::chrono::steady_clock::now();

    // Траблы камеры
    if (!cap.isOpened()) {
      cap.open(CAMERA_INDEX);
      if (cap.isOpened()) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, FPS);
      } else {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (now - last_frame_time > std::chrono::seconds(10)) {
          std::lock_guard<std::mutex> lk(frame_mutex);
          last_raw_frame.release();
        }
        continue;
      }
    }

    // Захват кадра 
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
      cap.release();
      continue;
    }
    last_frame_time = now;

    {
      std::lock_guard<std::mutex> lk(frame_mutex);
      frame.copyTo(last_raw_frame);
    }

    // Детекция YOLO 
    if (det_ptr_) {
      auto detection_result = det_ptr_->detect(frame);
      if (detection_result) {
        const auto& detections = *detection_result;
        if (!det_ptr_->drawDetections(frame, detections)) {
          std::cerr << "Drawing failed" << std::endl;
        }
        {
          std::lock_guard<std::mutex> lk(frame_mutex);
          frame.copyTo(last_recognition_frame);
        }
      } else {
        auto error = detection_result.error();
        std::cerr << (error == ObserverLoop::Detection::GenericDetectorError::
                                   EMPTY_FRAME
                          ? "Empty frame"
                          : "Detection error: " +
                                std::to_string(static_cast<int>(error)))
                  << std::endl;
      }
    }

    // --- Видеоэффекты
    auto loop_start = std::chrono::steady_clock::now();
    double current_time =
        std::chrono::duration<double>(loop_start - start_time).count();
    double system_uptime =
        std::chrono::duration<double>(loop_start - system_start_time).count();

    cv::Mat small_frame, display_frame;
    cv::resize(frame, small_frame, cv::Size(640, 480));
    cv::resize(small_frame, display_frame, cv::Size(WIDTH, HEIGHT));

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

    // Ззвук, FPS, отрисовка
    // if (flags.sound && frame_counter % 5 == 0) {
    //   Utility::Effects::playDetectSound();
    // }
    // cv::rectangle(processed, cv::Point(WIDTH - 180, 10),
    //               cv::Point(WIDTH - 10, 70), cv::Scalar(0, 0, 0, 200), -1);
    // double fps =
    //     1.0 / (std::chrono::duration<double>(loop_start - start_time).count()
    //     +
    //            0.001);
    // cv::putText(processed, "TARGETS: " +
    // std::to_string(tracked_objects.size()),
    //             cv::Point(WIDTH - 170, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
    //             cv::Scalar(0, 0, 255), 1);
    // cv::putText(processed, "FPS: " + std::to_string(static_cast<int>(fps)),
    //             cv::Point(WIDTH - 170, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    //             cv::Scalar(0, 200, 255), 1);

    {
      std::lock_guard<std::mutex> lock(frame_mutex);
      processed.copyTo(last_frame);
    }

    // Запись видео 
    if (flags.recording) {
      //   if (!tracked_objects.empty()) {
      //     last_detection_time = current_time;
      //     detection_active = true;
      //     if (!recorder.isRecording()) {
      //       recorder.startRecording();
      //       if (alert_enabled && authorizedUserId != 0) {
      //         std::string tmp_path = "/tmp/detection_alert.jpg";
      //         cv::imwrite(tmp_path, frame);
      //         try {
      //           bot->getApi().sendPhoto(
      //               authorizedUserId,
      //               TgBot::InputFile::fromFile(tmp_path, "image/jpeg"),
      //               "Обнаружены люди в кадре");
      //         } catch (...) {
      //           logMsg("Ошибка отправки уведомления");
      //         }
      //         std::filesystem::remove(tmp_path);
      //       }
      //     }
      //   }

      if (unstopable_mode) {
        if (!recorder.isRecording()) recorder.startRecording();
        recorder.writeFrame(processed);
      } else if (recorder.isRecording()) {
        recorder.writeFrame(processed);
        if ((current_time - last_detection_time > RECORDING_DURATION) &&
            system_uptime > INITIAL_RECORDING_DURATION) {
          recorder.stopRecording();
        }
      }

      if (system_uptime < INITIAL_RECORDING_DURATION &&
          !recorder.isRecording()) {
        recorder.startRecording();
      }
    }

    // Ограничение FPS и фоновая очистка 
    auto loop_end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       loop_end - loop_start)
                       .count();
    if (elapsed < 40) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40 - elapsed));
    }

    static auto last_cleanup = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(loop_end - last_cleanup).count() >
        3600) {            // Раз в час
      cleanupOldVideos();  // внутри проверяется флаг
      last_cleanup = loop_end;
    }
  }
}
