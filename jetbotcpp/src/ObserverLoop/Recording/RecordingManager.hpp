#pragma once

#include "RecordingConfig.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace ObserverLoop::Recording {

using TimestampNs = std::uint64_t;

/// Какой из двух записываемых потоков нужен вызывающему.
enum class VideoStream {
    Camera,   // camera_%06d.mkv — исходный поток с камеры, без YOLO и эффектов
    Processed // processed_%06d.mkv — кадры после YOLO и эффектов
};

struct FramePacket {
    cv::Mat image;
    TimestampNs pts_ns{};
    TimestampNs duration_ns{};
};

class RecordingManager {
public:
    explicit RecordingManager(RecordingConfig config);
    ~RecordingManager();

    RecordingManager(const RecordingManager&) = delete;
    RecordingManager& operator=(const RecordingManager&) = delete;
    RecordingManager(RecordingManager&&) = delete;
    RecordingManager& operator=(RecordingManager&&) = delete;

    // Starts one GStreamer pipeline: V4L2 camera -> raw splitmuxsink + appsink,
    // processed appsrc -> second splitmuxsink, one ALSA mic -> both muxers.
    // If the audio branch cannot be opened, the pipeline is started once more
    // without audio: the absence of a microphone never costs video recording.
    [[nodiscard]] bool start();
    [[nodiscard]] bool stop();
    [[nodiscard]] bool isRunning() const noexcept;

    // Pulls the newest available camera frame. Appsink is bounded and drops
    // stale processing frames if YOLO is slower than the camera. The raw
    // recording branch is independent and continues at camera rate.
    // Does not hold internal locks while waiting for a frame.
    [[nodiscard]] bool readCameraFrame(
        FramePacket& packet,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});

    // Push the processed form of the same camera frame using packet.pts_ns.
    // Processed frames are allowed to be dropped when the encoder is behind;
    // the camera (raw) branch is never affected.
    [[nodiscard]] bool pushProcessedFrame(const cv::Mat& frame, TimestampNs pts_ns);

    // Event gate for the recording streams (see RecordingConfig.event_driven).
    // enabled == true  -> buffers reach the muxers (event is being recorded)
    // enabled == false -> valves drop data, files stop growing; the raw stream
    //                     stays open if RecordingConfig::raw_always_recording.
    // The camera keeps running either way, so switching back is instant.
    [[nodiscard]] bool setStreamsEnabled(bool enabled);
    [[nodiscard]] bool streamsEnabled() const noexcept;

    // True when the running pipeline really contains the audio branch.
    [[nodiscard]] bool audioActive() const noexcept;

    // Pipeline running-time, useful for synthetic frames (not camera frames).
    [[nodiscard]] TimestampNs timestampNow() const noexcept;

    [[nodiscard]] std::string lastError() const;
    [[nodiscard]] std::string currentSessionDirectory() const;

    // Newest *finalized* segment of the requested stream for the newest session
    // whose name starts with "session_<date>" (date is YYYYMMDD). The segment
    // currently being written is never returned, so callers can send the file
    // without stopping the recording.
    [[nodiscard]] std::string latestFinalizedVideo(VideoStream stream,
                                                   const std::string& date) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ObserverLoop::Recording
