#pragma once
#include <opencv2/core.hpp>
#include "Globals.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>

inline cv::Mat applyEffects(cv::Mat &frame, cv::Mat &memory_dump) {
  if (!flag_effects) {
    return frame.clone();
  }

  // 1. Применение красных оттенков
  cv::Mat processed;

  if (frame.channels() == 3) {
    std::vector<cv::Mat> channels;
    cv::split(frame, channels);

    std::vector<cv::Mat> red_channels(3);
    red_channels[0] = cv::Mat::zeros(frame.size(), CV_8UC1); // B
    red_channels[1] = cv::Mat::zeros(frame.size(), CV_8UC1); // G
    red_channels[2] = channels[2];                           // R

    cv::Mat red_channel;
    cv::merge(red_channels, red_channel);

    cv::addWeighted(frame, 0.3, red_channel, 0.7, 0, processed);
  } else {
    frame.copyTo(processed);
  }

  // 2. Эффект кинескопа
  for (int i = 1; i < processed.rows; i += 2) {
    processed.row(i) = cv::Scalar(0, 0, 0);
  }

  // 3. Конвертация в BGRA
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
        cv::Vec4b &frame_pixel = processed_bgra.at<cv::Vec4b>(i, j);

        for (int c = 0; c < 3; ++c) {
          frame_pixel[c] = static_cast<uchar>(dump_pixel[c] * alpha +
                                              frame_pixel[c] * (1 - alpha));
        }
      }
    }
  }

  // 5. Обратно в BGR
  cv::Mat result;
  cv::cvtColor(processed_bgra, result, cv::COLOR_BGRA2BGR);

  return result;
}