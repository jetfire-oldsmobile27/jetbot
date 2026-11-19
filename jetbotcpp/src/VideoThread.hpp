#pragma once
#include <tgbot/Bot.h>

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <utility>

#include "Config.hpp"
#include "Effects.hpp"
#include "Globals.hpp"
#include "Logging.hpp"
#include "ObserverLoop/Detection/YoloDetector.hpp"
#include "ObserverLoop/Recording/RecordingManager.hpp"
#include "Startup/StartupManager.hpp"
#include "Utility/Effects.hpp"
#include "Utility/Settings.hpp"

void video_processing_thread(TgBot::Bot* bot,
                             ObserverLoop::Recording::RecordingManager& recorder,
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

      cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);
      cv::Mat processed_anim =
          flags.effects ? applyEffects(anim_frame, memory_dump) : anim_frame;

      {
        std::lock_guard<std::mutex> lock(frame_mutex);
        processed_anim.copyTo(last_frame);
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

  // После перезагрузки камера может получить другой номер, поэтому OpenCV-путь
  // тоже перебирает индексы, а не полагается только на CAM_INDEX.
  auto open_cv_camera = [&cap](int preferred) {
    if (cap.open(preferred)) {
      return preferred;
    }
    for (int idx = 0; idx < 10; ++idx) {
      if (idx == preferred) {
        continue;
      }
      if (cap.open(idx)) {
        return idx;
      }
    }
    return -1;
  };

  // Пути к модели: сначала env, затем файлы рядом с бинарником. Значения по
  // умолчанию в main.cpp указывают на другую машину.
  std::string model_path = ONNX_MODEL_PATH;
  std::string names_path = COCO_NAMES_PATH;
  if (!std::filesystem::exists(model_path)) {
    const std::string local = resource_dir + "/yolov10n.onnx";
    if (std::filesystem::exists(local)) {
      model_path = local;
    }
  }
  if (!std::filesystem::exists(names_path)) {
    const std::string local = resource_dir + "/coco.names";
    if (std::filesystem::exists(local)) {
      names_path = local;
    }
  }
  if (flags.detection && !std::filesystem::exists(model_path)) {
    logMsg("⚠️ Модель детектора не найдена: " + model_path +
           " (задайте ONNX_MODEL_PATH или положите yolov10n.onnx рядом с бинарником).");
  }

  std::unique_ptr<ObserverLoop::Detection::YoloDetector> det_ptr_;
  std::expected<std::unique_ptr<ObserverLoop::Detection::YoloDetector>,
                ObserverLoop::Detection::GenericDetectorError>
      yolo_det;
  if (flags.detection) {
    yolo_det = ObserverLoop::Detection::YoloDetector::create(model_path,
                                                             names_path);
    if (yolo_det.has_value()) {
      det_ptr_ = std::move(*yolo_det);
      std::cout << "Detector created and moved successfully\n";
    } else {
      std::cerr << "Failed to create YOLO detector (model: " << model_path
                << "). Detection disabled.\n";
    }
  }

  // --- Захват камеры и запись ---
  // GStreamer-пайплайн должен быть единственным владельцем /dev/video0,
  // поэтому OpenCV-захват, открытый StartupManager-ом, освобождается.
  // Если запись отключена или не поднялась, остаётся прежний путь через
  // cv::VideoCapture (фото/статус продолжают работать).
  bool pipeline_capture = false;
  if (flags.recording) {
    cap.release();
    if (recorder.start()) {
      pipeline_capture = true;
      std::cout << "Recording session: " << recorder.currentSessionDirectory()
                << std::endl;
    } else {
      logMsg("❌ Не удалось запустить запись видео: " + recorder.lastError() +
             ". Работаем без записи, захват через OpenCV.");
      const int opened = open_cv_camera(CAMERA_INDEX);
      if (opened < 0) {
        logMsg("❌ Камера не открылась после отката на OpenCV (индексы 0..9).");
      } else if (opened != CAMERA_INDEX) {
        logMsg("OpenCV: камера открыта по индексу " + std::to_string(opened) +
               " вместо CAM_INDEX=" + std::to_string(CAMERA_INDEX));
      }
    }
  }

  double last_dump_time{0.0};
  cv::Mat memory_dump = Utility::Effects::generateMemoryDump(WIDTH, HEIGHT);

  auto start_time = std::chrono::steady_clock::now();
  auto system_start_time = start_time;
  double last_detection_time{0.0};

  // Состояние событийной задвижки: true - сегменты пишутся. Начинаем открытой,
  // чтобы записать стартовое окно INITIAL_RECORDING_DURATION.
  bool segment_gate_open = true;
  bool warned_no_detection = false;

  auto last_frame_time = std::chrono::steady_clock::now();

  FrameBuffer buffer;

  while (running) {
    auto now = std::chrono::steady_clock::now();

    cv::Mat frame;
    ObserverLoop::Recording::TimestampNs frame_pts = 0;

    if (pipeline_capture) {
      // Кадр приходит из GStreamer вместе с исходным PTS камеры.
      ObserverLoop::Recording::FramePacket packet;
      if (!recorder.readCameraFrame(packet, std::chrono::milliseconds(1000))) {
        if (!recorder.isRunning() || !recorder.lastError().empty()) {
          logMsg("❌ Ошибка захвата/записи: " + recorder.lastError());
          break;
        }
        continue; // таймаут ожидания кадра, пробуем снова
      }
      frame = std::move(packet.image);
      frame_pts = packet.pts_ns;
      if (frame.empty()) {
        continue;
      }
    } else {
      // Прежний путь: OpenCV-захват (режим без записи).
      if (!cap.isOpened()) {
        if (open_cv_camera(CAMERA_INDEX) >= 0) {
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

      if (!cap.read(frame) || frame.empty()) {
        cap.release();
        continue;
      }
    }

    last_frame_time = now;

    {
      std::lock_guard<std::mutex> lk(frame_mutex);
      frame.copyTo(last_raw_frame);
    }

    auto loop_start = std::chrono::steady_clock::now();
    double current_time =
        std::chrono::duration<double>(loop_start - start_time).count();
    double system_uptime =
        std::chrono::duration<double>(loop_start - system_start_time).count();

    // Детекция YOLO
    if (det_ptr_) {
      auto detection_result = det_ptr_->detect(frame);
      if (detection_result) {
        const auto& detections = *detection_result;
        detection_active = !detections.empty();
        if (!detections.empty()) {
          last_detection_time = current_time;
        }
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
    } else {
      detection_active = false;
    }

    // --- Политика записи: непрерывная (/unstopable on) или по обнаружению ---
    // Камера и пайплайн работают всегда; задвижки valve решают, попадают ли
    // кадры в muxer-ы. Так переключение мгновенное и не рвёт сегменты.
    if (pipeline_capture) {
      // Без детектора событий не будет, и запись по обнаружению превратилась бы
      // в "60 секунд после старта и тишина". В этом случае пишем непрерывно.
      const bool detection_available = static_cast<bool>(det_ptr_);
      if (!detection_available && !unstopable_mode.load() &&
          !warned_no_detection) {
        warned_no_detection = true;
        logMsg("⚠️ Детекция выключена или детектор не создан: запись по "
               "событию невозможна, пишем непрерывно.");
      }

      const bool always_record = unstopable_mode.load() || !detection_available;
      const bool should_record =
          always_record || system_uptime <= INITIAL_RECORDING_DURATION ||
          (current_time - last_detection_time) <= RECORDING_DURATION;

      if (should_record != segment_gate_open) {
        segment_gate_open = should_record;
        if (!recorder.setStreamsEnabled(should_record)) {
          logMsg("⚠️ Не удалось переключить задвижки записи");
        }
        logMsg(should_record
                   ? "▶️ Запись: событие активно, сегменты пишутся"
                   : "⏸️ Запись: событий нет, сегменты на паузе");
      }
    }

    // --- Видеоэффекты
    buffer.push(frame, current_time);

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

    {
      std::lock_guard<std::mutex> lock(frame_mutex);
      processed.copyTo(last_frame);
    }

    // Отправка обработанного кадра в ветку processed с исходным PTS камеры:
    // YOLO может быть медленнее камеры, тогда в файле будут пропуски по PTS,
    // но не ускоренное воспроизведение.
    if (pipeline_capture && recorder.isRunning() &&
        recorder.streamsEnabled() && !processed.empty()) {
      if (!recorder.pushProcessedFrame(processed, frame_pts)) {
        logMsg("❌ Не удалось записать обработанный кадр: " +
               recorder.lastError());
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

  if (pipeline_capture && recorder.isRunning()) {
    if (recorder.stop()) {
      logMsg("⏹️ Запись остановлена, файлы: " +
             recorder.currentSessionDirectory());
    } else {
      logMsg("⚠️ Остановка записи завершилась с предупреждением: " +
             recorder.lastError());
    }
  }
}
