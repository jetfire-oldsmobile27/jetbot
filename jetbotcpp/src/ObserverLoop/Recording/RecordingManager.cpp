#include "ObserverLoop/Recording/RecordingManager.hpp"

#include "Logging.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace ObserverLoop::Recording {
namespace {

std::once_flag gst_init_flag;

struct GstUnref {
    void operator()(GstObject* object) const noexcept {
        if (object) gst_object_unref(object);
    }
};
using GstHandle = std::unique_ptr<GstObject, GstUnref>;

GstHandle gstRef(GstObject* object) { return GstHandle(object); }

std::string gstQuote(const std::string& value) {
    std::string out{"\""};
    out.reserve(value.size() + 2);
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') out.push_back('\\');
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string makeSessionName() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    std::ostringstream out;
    out << "session_" << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << "_" << static_cast<long long>(::getpid());
    return out.str();
}

std::string gstErrorMessage(GstMessage* message) {
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);

    std::string result = error ? error->message : "unknown GStreamer error";
    if (debug && *debug) {
        result += " [";
        result += debug;
        result += "]";
    }
    if (error) g_error_free(error);
    g_free(debug);
    return result;
}

bool haveElement(const char* name) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (!factory) {
        return false;
    }
    gst_object_unref(factory);
    return true;
}

// Проверяет, что ALSA-устройство реально отдаёт данные: «открылось» и
// «работает» - разные вещи. Без этой пробы зависший alsasrc блокирует
// preroll всего пайплайна, и файлы остаются нулевого размера.
// Требует уже выполненного gst_init().
bool probeAudioDevice(const std::string& device, int timeout_ms) {
    const std::string description =
        "alsasrc device=" + gstQuote(device) +
        " num-buffers=4 ! audioconvert ! audioresample ! "
        "audio/x-raw,rate=48000,channels=1 ! fakesink sync=false";

    GError* parse_error = nullptr;
    GstElement* pipeline = gst_parse_launch(description.c_str(), &parse_error);
    if (parse_error != nullptr) {
        g_error_free(parse_error);
        if (pipeline != nullptr) {
            gst_object_unref(pipeline);
        }
        return false;
    }
    if (pipeline == nullptr) {
        return false;
    }

    bool ok = false;
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) !=
        GST_STATE_CHANGE_FAILURE) {
        GstBus* bus = gst_element_get_bus(pipeline);
        GstMessage* message = gst_bus_timed_pop_filtered(
            bus, static_cast<GstClockTime>(timeout_ms) * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
        ok = (message != nullptr) && GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS;
        if (message != nullptr) {
            gst_message_unref(message);
        }
        gst_object_unref(bus);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return ok;
}

const char* streamPrefix(VideoStream stream) {
    return stream == VideoStream::Camera ? "camera" : "processed";
}

// Имя файла, который splitmuxsink пишет прямо сейчас (пусто между сегментами).
std::string currentMuxFile(GstElement* mux) {
    if (!mux) {
        return {};
    }
    gchar* path = nullptr;
    g_object_get(mux, "current-file", &path, nullptr);
    std::string result = path ? path : "";
    g_free(path);
    return result;
}

// Ищет V4L2-устройство камеры через GstDeviceMonitor, а не по номеру: после
// перезагрузки камера может оказаться /dev/video2 вместо /dev/video0.
// Формат больше не навязывается (масштаб/частота приводятся в пайплайне),
// поэтому достаточно любого устройства, отдающего video/x-raw.
// preferred используется, только если этот узел существует как источник видео.
// Требует уже выполненного gst_init().
std::string findAutoCameraDevice(const std::string& preferred,
                                 std::string& diagnostics) {
    GstDeviceMonitor* monitor = gst_device_monitor_new();

    GstCaps* filter_caps = gst_caps_new_empty_simple("video/x-raw");
    const gboolean filter_ok =
        gst_device_monitor_add_filter(monitor, "Video/Source", filter_caps);
    gst_caps_unref(filter_caps);
    if (!filter_ok) {
        gst_object_unref(monitor);
        diagnostics = "не удалось добавить фильтр Video/Source";
        return {};
    }

    if (!gst_device_monitor_start(monitor)) {
        gst_object_unref(monitor);
        diagnostics = "GstDeviceMonitor не запустился";
        return {};
    }

    GstCaps* wanted = gst_caps_new_empty_simple("video/x-raw");

    std::string chosen;
    std::string fallback;
    std::string preferred_matching;
    std::string preferred_any;

    GList* devices = gst_device_monitor_get_devices(monitor);
    for (GList* item = devices; item != nullptr; item = item->next) {
        GstDevice* device = GST_DEVICE(item->data);
        GstStructure* properties = gst_device_get_properties(device);

        const gchar* api = gst_structure_get_string(properties, "device.api");
        const gchar* path = gst_structure_get_string(properties, "device.path");
        const bool is_v4l2 = (api == nullptr || g_strcmp0(api, "v4l2") == 0);

        if (is_v4l2 && path != nullptr) {
            const std::string node = path;
            if (fallback.empty()) {
                fallback = node;
            }
            if (!preferred.empty() && node == preferred) {
                preferred_any = node;
            }

            GstCaps* device_caps = gst_device_get_caps(device);
            if (device_caps != nullptr) {
                const bool intersects = gst_caps_can_intersect(device_caps, wanted);
                if (intersects && chosen.empty()) {
                    chosen = node;
                }
                if (intersects && !preferred.empty() && node == preferred) {
                    preferred_matching = node;
                }
                gst_caps_unref(device_caps);
            }
        }

        gst_structure_free(properties);
    }

    for (GList* item = devices; item != nullptr; item = item->next) {
        gst_object_unref(item->data);
    }
    g_list_free(devices);
    gst_caps_unref(wanted);
    gst_device_monitor_stop(monitor);
    gst_object_unref(monitor);

    if (!preferred_matching.empty()) {
        diagnostics = "предпочтение CAM_INDEX подтверждено: " + preferred_matching;
        return preferred_matching;
    }
    if (!preferred.empty() && preferred_any.empty()) {
        diagnostics = "устройство " + preferred +
                      " не найдено, ищем камеру по возможностям";
    } else if (!preferred.empty() && preferred_matching.empty()) {
        diagnostics = "устройство " + preferred +
                      " не поддерживает запрошенный формат, ищем другую камеру";
    }
    if (!chosen.empty()) {
        if (diagnostics.empty()) {
            diagnostics = "выбрано по caps: " + chosen;
        } else {
            diagnostics += "; выбрано по caps: " + chosen;
        }
        return chosen;
    }
    if (!fallback.empty()) {
        const std::string note = "caps не совпали, взято первое V4L2-устройство: " +
                                 fallback;
        diagnostics = diagnostics.empty() ? note : diagnostics + "; " + note;
        return fallback;
    }

    diagnostics += "; не найдено ни одного V4L2-устройства (/dev/video*)";
    return {};
}

} // namespace

class RecordingManager::Impl {
public:
    explicit Impl(RecordingConfig config) : config_(std::move(config)) {
        if (config_.width <= 0 || config_.height <= 0 || config_.fps <= 0)
            throw std::invalid_argument("Invalid video dimensions or FPS");
        if (config_.segment_duration.count() <= 0)
            throw std::invalid_argument("segment_duration must be positive");
        if (config_.processed_queue_buffers <= 0)
            throw std::invalid_argument("processed_queue_buffers must be positive");

        audio_requested_ = !config_.microphone_device.empty();

        if (config_.video_encoder_chain.empty()) {
            config_.video_encoder_chain =
                "videoconvert ! video/x-raw,format=I420 ! "
                "x264enc tune=zerolatency speed-preset=ultrafast "
                "bitrate=" + std::to_string(config_.video_bitrate_kbps) +
                " key-int-max=" + std::to_string(config_.fps * 2) +
                " bframes=0 ! h264parse config-interval=-1 ! "
                "video/x-h264,stream-format=avc,alignment=au";
        }
    }

    ~Impl() { (void)stop(); }

    bool start() {
        std::lock_guard control(control_mutex_);
        if (running_) {
            return true;
        }

        std::call_once(gst_init_flag, [] { gst_init(nullptr, nullptr); });

        if (!checkVideoElements()) {
            return false;
        }

        bool with_audio = audio_requested_;
        if (with_audio && !checkAudioElements()) {
            logMsg("⚠️ GStreamer: нет элементов для звука (alsasrc/opusenc). "
                   "Запись пойдёт без аудиодорожки.");
            with_audio = false;
        }
        if (with_audio && !probeAudioDevice(config_.microphone_device, 3000)) {
            logMsg("⚠️ ALSA-устройство \"" + config_.microphone_device +
                   "\" не отдаёт данные (проверьте arecord -l и JETBOT_MIC). "
                   "Запись пойдёт без звука.");
            with_audio = false;
        }

        if (startInternal(with_audio)) {
            audio_active_ = with_audio;
            if (!with_audio && audio_requested_) {
                logMsg("⚠️ Запись видео без звука: микрофон не используется.");
            }
            return true;
        }

        const std::string video_error = lastError();
        if (!with_audio) {
            logMsg("❌ Запуск записи не удался: " + video_error);
            return false;
        }

        // Микрофон не должен лишать пользователя видеозаписи: пробуем ещё раз
        // без аудиоветки.
        logMsg("⚠️ Не удалось поднять звук (" + video_error +
               "). Повторный запуск записи без аудио.");
        if (startInternal(false)) {
            audio_active_ = false;
            logMsg("⚠️ Запись запущена БЕЗ звука. Проверьте JETBOT_MIC "
                   "(arecord -l).");
            return true;
        }

        logMsg("❌ Запуск записи не удался и без звука: " + lastError());
        return false;
    }

    bool stop() {
        std::lock_guard control(control_mutex_);
        return stopInternal(true);
    }

    bool isRunning() const noexcept { return running_.load(); }

    bool readCameraFrame(FramePacket& packet, std::chrono::milliseconds timeout) {
        GstHandle sink;
        GstHandle bus;
        {
            std::lock_guard lock(state_mutex_);
            if (!running_ || failed_ || !camera_sink_) {
                return false;
            }
            sink = gstRef(GST_OBJECT(gst_object_ref(camera_sink_)));
            if (bus_) {
                bus = gstRef(GST_OBJECT(gst_object_ref(bus_)));
            }
        }

        const auto timeout_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count();
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink.get()),
            static_cast<GstClockTime>(std::max<std::int64_t>(0, timeout_ns)));
        if (!sample) {
            pollBus(bus.get());
            return false; // таймаут, EOS или ошибка пайплайна
        }

        GstCaps* caps = gst_sample_get_caps(sample);
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        GstVideoInfo info;
        gst_video_info_init(&info);
        if (!caps || !buffer || !gst_video_info_from_caps(&info, caps) ||
            GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_BGR) {
            gst_sample_unref(sample);
            setError("appsink delivered invalid caps/buffer (expected BGR)");
            return false;
        }

        GstVideoFrame mapped{};
        if (!gst_video_frame_map(&mapped, &info, buffer, GST_MAP_READ)) {
            gst_sample_unref(sample);
            setError("Could not map camera GstVideoFrame");
            return false;
        }

        const int width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
        const int height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
        bool announce = false;
        {
            std::lock_guard lock(state_mutex_);
            if (!logged_first_frame_) {
                logged_first_frame_ = true;
                announce = true;
            }
        }
        if (announce) {
            logMsg("Запись: первый кадр камеры " + std::to_string(width) + "x" +
                   std::to_string(height));
        }
        const auto* data =
            static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&mapped, 0));
        const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&mapped, 0);
        cv::Mat view(height, width, CV_8UC3,
                     const_cast<std::uint8_t*>(data),
                     static_cast<std::size_t>(stride));
        packet.image = view.clone();

        const GstClockTime pts = GST_BUFFER_PTS(buffer);
        const GstClockTime duration = GST_BUFFER_DURATION(buffer);
        packet.pts_ns = GST_CLOCK_TIME_IS_VALID(pts)
                            ? static_cast<TimestampNs>(pts)
                            : timestampNow();
        packet.duration_ns =
            GST_CLOCK_TIME_IS_VALID(duration)
                ? static_cast<TimestampNs>(duration)
                : static_cast<TimestampNs>(GST_SECOND /
                                           static_cast<GstClockTime>(config_.fps));

        gst_video_frame_unmap(&mapped);
        gst_sample_unref(sample);

        return pollBus(bus.get());
    }

    bool pushProcessedFrame(const cv::Mat& input, TimestampNs pts) {
        GstHandle src;
        GstHandle bus;
        {
            std::lock_guard lock(state_mutex_);
            if (!running_ || failed_ || !processed_src_) {
                return false;
            }
            src = gstRef(GST_OBJECT(gst_object_ref(processed_src_)));
            if (bus_) {
                bus = gstRef(GST_OBJECT(gst_object_ref(bus_)));
            }
        }

        if (input.empty()) {
            return false;
        }

        cv::Mat frame;
        if (input.depth() != CV_8U) {
            setError("Processed frame must have 8-bit depth");
            return false;
        }
        if (input.type() == CV_8UC3) {
            frame = input;
        } else if (input.channels() == 1) {
            cv::cvtColor(input, frame, cv::COLOR_GRAY2BGR);
        } else if (input.channels() == 4) {
            cv::cvtColor(input, frame, cv::COLOR_BGRA2BGR);
        } else {
            setError("Unsupported processed frame format; expected BGR, gray or BGRA");
            return false;
        }

        if (frame.cols != config_.width || frame.rows != config_.height) {
            cv::resize(frame, frame, cv::Size(config_.width, config_.height));
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }

        const GstClockTime duration =
            GST_SECOND / static_cast<GstClockTime>(config_.fps);
        GstClockTime safe_pts = static_cast<GstClockTime>(pts);
        if (!GST_CLOCK_TIME_IS_VALID(safe_pts)) {
            safe_pts = timestampNow();
        }
        {
            std::lock_guard lock(state_mutex_);
            // PTS должны строго возрастать, иначе muxer ругается.
            if (GST_CLOCK_TIME_IS_VALID(last_processed_pts_) &&
                safe_pts <= last_processed_pts_) {
                safe_pts = last_processed_pts_ + duration;
            }
            last_processed_pts_ = safe_pts;
        }

        const gsize bytes =
            static_cast<gsize>(frame.total() * frame.elemSize());
        GstBuffer* out = gst_buffer_new_allocate(nullptr, bytes, nullptr);
        if (!out) {
            setError("Could not allocate processed video GstBuffer");
            return false;
        }
        gst_buffer_fill(out, 0, frame.data, bytes);
        GST_BUFFER_PTS(out) = safe_pts;
        GST_BUFFER_DTS(out) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(out) = duration;

        const GstFlowReturn flow =
            gst_app_src_push_buffer(GST_APP_SRC(src.get()), out);
        if (flow == GST_FLOW_FLUSHING) {
            return false; // пайплайн уже останавливается, это не ошибка
        }
        if (flow != GST_FLOW_OK) {
            setError(std::string("processed appsrc push failed: ") +
                     gst_flow_get_name(flow));
            return false;
        }

        return pollBus(bus.get());
    }

    bool setStreamsEnabled(bool enabled) {
        GstHandle raw_valve;
        GstHandle processed_valve;
        {
            std::lock_guard lock(state_mutex_);
            streams_enabled_ = enabled;
            if (raw_valve_) {
                raw_valve = gstRef(GST_OBJECT(gst_object_ref(raw_valve_)));
            }
            if (processed_valve_) {
                processed_valve =
                    gstRef(GST_OBJECT(gst_object_ref(processed_valve_)));
            }
        }

        if (raw_valve) {
            // raw можно писать непрерывно, тогда по событию открывается только
            // очередь обработанного потока.
            const bool raw_enabled = enabled || config_.raw_always_recording;
            g_object_set(raw_valve.get(), "drop", !raw_enabled, nullptr);
        }
        if (processed_valve) {
            g_object_set(processed_valve.get(), "drop", !enabled, nullptr);
        }
        return true;
    }

    bool streamsEnabled() const noexcept { return streams_enabled_.load(); }

    bool audioActive() const noexcept { return audio_active_.load(); }

    TimestampNs timestampNow() const noexcept {
        GstHandle pipeline;
        {
            std::lock_guard lock(state_mutex_);
            if (!pipeline_ || !running_) {
                return 0;
            }
            pipeline = gstRef(GST_OBJECT(gst_object_ref(pipeline_)));
        }

        GstClock* clock = gst_element_get_clock(GST_ELEMENT(pipeline.get()));
        if (!clock) {
            return 0;
        }
        const GstClockTime now = gst_clock_get_time(clock);
        gst_object_unref(clock);

        const GstClockTime base = gst_element_get_base_time(GST_ELEMENT(pipeline.get()));
        if (!GST_CLOCK_TIME_IS_VALID(now) || !GST_CLOCK_TIME_IS_VALID(base) ||
            now < base) {
            return 0;
        }
        return static_cast<TimestampNs>(now - base);
    }

    std::string lastError() const {
        std::lock_guard lock(state_mutex_);
        return last_error_;
    }

    std::string currentSessionDirectory() const {
        std::lock_guard lock(state_mutex_);
        return session_dir_.string();
    }

    std::string latestFinalizedVideo(VideoStream stream,
                                     const std::string& date) const {
        std::filesystem::path output_dir;
        std::string in_progress;
        {
            std::lock_guard lock(state_mutex_);
            output_dir = config_.output_dir;
            in_progress = currentMuxFile(const_cast<GstElement*>(
                stream == VideoStream::Camera ? raw_mux_ : processed_mux_));
        }

        std::error_code ec;
        if (!std::filesystem::is_directory(output_dir, ec)) {
            return {};
        }

        // Сессии называются session_<YYYYMMDD>_<HHMMSS>_<pid>, поэтому самая
        // свежая — лексикографически максимальная с нужным префиксом.
        const std::string prefix = "session_" + date;
        std::filesystem::path session;
        for (const auto& entry :
             std::filesystem::directory_iterator(output_dir, ec)) {
            if (ec) {
                break;
            }
            std::error_code entry_ec;
            if (!entry.is_directory(entry_ec)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0) {
                continue;
            }
            if (session.empty() || name > session.filename().string()) {
                session = entry.path();
            }
        }

        if (session.empty()) {
            return {};
        }

        ec.clear();

        // Берём завершённый сегмент с максимальным номером и пропускаем файл,
        // который splitmuxsink пишет прямо сейчас.
        const std::string file_prefix = std::string(streamPrefix(stream)) + "_";
        int best_index = -1;
        std::filesystem::path best_path;

        for (const auto& entry : std::filesystem::directory_iterator(session, ec)) {
            if (ec) {
                break;
            }
            std::error_code entry_ec;
            if (!entry.is_regular_file(entry_ec)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind(file_prefix, 0) != 0 ||
                entry.path().extension() != ".mkv") {
                continue;
            }
            if (!in_progress.empty() &&
                entry.path().string() == in_progress) {
                continue; // ещё пишется
            }

            const std::string digits =
                name.substr(file_prefix.size(),
                            name.size() - file_prefix.size() - 4);
            int index = -1;
            try {
                index = std::stoi(digits);
            } catch (const std::exception&) {
                continue;
            }

            if (index > best_index) {
                best_index = index;
                best_path = entry.path();
            }
        }

        if (best_index < 0) {
            return {};
        }

        const auto size = std::filesystem::file_size(best_path, ec);
        if (ec || size <= 48) {
            return {};
        }

        return best_path.string();
    }

private:
    bool checkVideoElements() {
        // Элемент -> пакет, в котором он лежит (Debian/Armbian).
        std::vector<std::pair<std::string, std::string>> required = {
            {"v4l2src", "gstreamer1.0-plugins-good"},
            {"tee", "gstreamer1.0-plugins-base"},
            {"queue", "gstreamer1.0-plugins-base"},
            {"valve", "gstreamer1.0-plugins-base"},
            {"videoconvert", "gstreamer1.0-plugins-base"},
            {"videoscale", "gstreamer1.0-plugins-base"},
            {"videorate", "gstreamer1.0-plugins-base"},
            {"appsink", "gstreamer1.0-plugins-base"},
            {"appsrc", "gstreamer1.0-plugins-base"},
            {"filesink", "gstreamer1.0-plugins-base"},
            {"splitmuxsink", "gstreamer1.0-plugins-good"},
            {"matroskamux", "gstreamer1.0-plugins-good"},
            {"h264parse", "gstreamer1.0-plugins-bad"},
        };
        // x264enc обязателен только для цепочки по умолчанию: при своём
        // JETBOT_ENCODER_CHAIN (например, mpph264enc) он не нужен.
        if (config_.video_encoder_chain.find("x264enc") != std::string::npos) {
            required.emplace_back("x264enc", "gstreamer1.0-plugins-ugly");
        }

        std::string missing;
        std::string packages;
        for (const auto& [element, package] : required) {
            if (haveElement(element.c_str())) {
                continue;
            }
            missing += (missing.empty() ? "" : ", ") + element +
                       " (" + package + ")";
            if (packages.find(package) == std::string::npos) {
                packages += (packages.empty() ? "" : " ") + package;
            }
        }

        if (!missing.empty()) {
            setError("Отсутствуют плагины GStreamer: " + missing +
                     ". Установите: sudo apt install " + packages +
                     ", либо задайте JETBOT_ENCODER_CHAIN с имеющимся "
                     "энкодером.");
            return false;
        }
        return true;
    }

    bool checkAudioElements() {
        const std::vector<std::pair<std::string, std::string>> required = {
            {"alsasrc", "gstreamer1.0-alsa"},
            {"audioconvert", "gstreamer1.0-plugins-base"},
            {"audioresample", "gstreamer1.0-plugins-base"},
            {"opusenc", "gstreamer1.0-plugins-base"},
        };
        for (const auto& [element, package] : required) {
            if (!haveElement(element.c_str())) {
                logMsg("⚠️ Нет элемента GStreamer " + element + " (пакет " +
                       package + "): запись пойдёт без звука.");
                return false;
            }
        }
        return true;
    }

    // Логирует элементы, которые не дошли до PLAYING: сразу видно, чей preroll
    // не завершился (muxer, sink или src).
    void logStalledElements(GstElement* pipeline) {
        if (pipeline == nullptr) {
            return;
        }
        std::string stalled;
        int shown = 0;
        GstIterator* iterator = gst_bin_iterate_recurse(GST_BIN(pipeline));
        GValue item = G_VALUE_INIT;
        while (shown < 20 &&
               gst_iterator_next(iterator, &item) == GST_ITERATOR_OK) {
            GstElement* element = GST_ELEMENT(g_value_get_object(&item));
            GstState state = GST_STATE_VOID_PENDING;
            GstState pending = GST_STATE_VOID_PENDING;
            gst_element_get_state(element, &state, &pending, 0);
            if (state != GST_STATE_PLAYING ||
                pending != GST_STATE_VOID_PENDING) {
                stalled += " ";
                stalled += GST_ELEMENT_NAME(element);
                stalled += "=";
                stalled += gst_element_state_get_name(state);
                if (pending != GST_STATE_VOID_PENDING) {
                    stalled += "->";
                    stalled += gst_element_state_get_name(pending);
                }
                ++shown;
            }
            g_value_unset(&item);
        }
        gst_iterator_free(iterator);
        logMsg("Запись: элементы не в PLAYING:" +
               (stalled.empty() ? std::string(" нет") : stalled));
    }

    bool startInternal(bool with_audio) {
        {
            std::lock_guard lock(state_mutex_);
            if (pipeline_) {
                teardownLocked();
            }
            failed_ = false;
            last_error_.clear();
            streams_enabled_ = true;
            last_processed_pts_ = GST_CLOCK_TIME_NONE;
            logged_first_frame_ = false;
        }

        try {
            std::filesystem::create_directories(config_.output_dir);
            const auto session = config_.output_dir / makeSessionName();
            std::filesystem::create_directories(session);
            std::lock_guard lock(state_mutex_);
            session_dir_ = session;
        } catch (const std::exception& e) {
            setError(std::string("Cannot create recording directory: ") +
                     e.what());
            return false;
        }

        std::filesystem::path session_dir;
        {
            std::lock_guard lock(state_mutex_);
            session_dir = session_dir_;
        }

        const auto raw_location = (session_dir / "camera_%06d.mkv").string();
        const auto processed_location =
            (session_dir / "processed_%06d.mkv").string();
        const auto segment_ns =
            static_cast<std::uint64_t>(config_.segment_duration.count()) *
            GST_SECOND;
        // async=false у файловых sink-ов: живой рекордер не должен ждать
        // preroll. Иначе splitmuxsink processed-ветки ждёт первый буфер от
        // appsrc, а приложение ждёт кадр из appsink - оба стоят, пайплайн не
        // выходит в PLAYING, файлы остаются нулевого размера.
        const std::string sink_properties =
            config_.synchronous_files
                ? " sink-properties=\"properties,o-sync=true,async=false\""
                : " sink-properties=\"properties,async=false\"";
        const std::string mux_common =
            " max-size-time=" + std::to_string(segment_ns) +
            " max-size-bytes=0 send-keyframe-requests=true async-finalize=true "
            "muxer-factory=matroskamux "
            "muxer-properties=\"properties,streamable=true\" "
            "sink-factory=filesink" + sink_properties;

        const std::string appsrc_caps =
            "video/x-raw,format=BGR,width=" + std::to_string(config_.width) +
            ",height=" + std::to_string(config_.height) +
            ",framerate=" + std::to_string(config_.fps) + "/1";

        // Политика перегрузки обработанной ветки:
        //  * appsrc не блокирует основной цикл (block=false), кадры сверх
        //    max-bytes отбрасываются;
        //  * очередь перед энкодером ограничена processed_queue_buffers и
        //    помечена leaky=downstream: при отставании энкодера теряются
        //    самые старые кадры, а не накапливается задержка;
        //  * raw-ветка не ограничена и пишется с частотой камеры независимо
        //    от скорости YOLO/кодировщика processed.
        const std::string processed_queue =
            "queue max-size-buffers=" +
            std::to_string(config_.processed_queue_buffers) +
            " max-size-bytes=0 max-size-time=0 leaky=downstream";

        // Устройство камеры: явный путь или автопоиск по возможностям.
        std::string camera_device = config_.camera_device;
        if (camera_device.empty() || camera_device == "auto") {
            std::string diagnostics;
            camera_device = findAutoCameraDevice(
                config_.camera_preferred_device, diagnostics);
            if (camera_device.empty()) {
                setError("Камера не найдена: " + diagnostics +
                         ". Задайте JETBOT_CAMERA=/dev/videoN или CAM_INDEX.");
                return false;
            }
            logMsg("Запись: " + diagnostics);
        } else {
            logMsg("Запись: камера задана явно: " + camera_device);
        }
        {
            std::lock_guard lock(state_mutex_);
            active_camera_device_ = camera_device;
        }

        // Valve перед энкодером: закрытая задвижка не тратит CPU на кодирование
        // и не даёт данным дойти до muxer, пока событие не записывается.
        std::ostringstream launch;
        launch
            << "v4l2src device=" << gstQuote(camera_device)
            << " do-timestamp=true ! "
            // Камера не обязана поддерживать ровно width/height/fps: принимаем
            // её нативные caps и приводим формат/размер/частоту здесь, иначе
            // v4l2src падает с not-negotiated.
            << "videoconvert ! videoscale ! videorate ! "
            << "video/x-raw,format=I420,width=" << config_.width
            << ",height=" << config_.height
            << ",framerate=" << config_.fps << "/1 ! "
            << "tee name=camera_tee "

            << "camera_tee. ! queue max-size-time=1000000000 max-size-bytes=0 max-size-buffers=0 ! "
            << "valve name=raw_valve drop=false ! "
            << config_.video_encoder_chain << " ! "
            << "splitmuxsink name=raw_mux location=" << gstQuote(raw_location)
            << mux_common << " "

            << "camera_tee. ! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            << "videoconvert ! video/x-raw,format=BGR ! "
            << "appsink name=camera_sink emit-signals=false sync=false async=false max-buffers=2 drop=true wait-on-eos=false "

            << "appsrc name=processed_src is-live=true format=time do-timestamp=false "
            << "block=false max-bytes=" << config_.processed_appsrc_max_bytes
            << " caps=" << gstQuote(appsrc_caps) << " ! "
            << processed_queue << " ! "
            << "valve name=processed_valve drop=false ! "
            << config_.video_encoder_chain << " ! "
            << "splitmuxsink name=processed_mux location=" << gstQuote(processed_location)
            << mux_common << " ";

        if (with_audio) {
            launch
                << "alsasrc device=" << gstQuote(config_.microphone_device)
                << " do-timestamp=true ! "
                << "queue max-size-time=2000000000 max-size-bytes=0 max-size-buffers=0 ! "
                << "audioconvert ! audioresample ! "
                << "audio/x-raw,rate=48000,channels=1 ! "
                << "opusenc bitrate=" << config_.audio_bitrate_bps << " ! "
                << "tee name=audio_tee "
                << "audio_tee. ! queue max-size-time=2000000000 max-size-bytes=0 max-size-buffers=0 ! raw_mux.audio_%u "
                << "audio_tee. ! queue max-size-time=2000000000 max-size-bytes=0 max-size-buffers=0 ! processed_mux.audio_%u";
        }

        GError* parse_error = nullptr;
        GstElement* pipeline = gst_parse_launch(launch.str().c_str(), &parse_error);
        if (parse_error) {
            setError(std::string("GStreamer pipeline parse failed: ") +
                     parse_error->message);
            g_error_free(parse_error);
            if (pipeline) {
                gst_object_unref(pipeline);
            }
            return false;
        }
        if (!pipeline) {
            setError("GStreamer did not create a pipeline");
            return false;
        }

        GstElement* processed_src =
            gst_bin_get_by_name(GST_BIN(pipeline), "processed_src");
        GstElement* camera_sink =
            gst_bin_get_by_name(GST_BIN(pipeline), "camera_sink");
        GstElement* raw_valve =
            gst_bin_get_by_name(GST_BIN(pipeline), "raw_valve");
        GstElement* processed_valve =
            gst_bin_get_by_name(GST_BIN(pipeline), "processed_valve");
        GstElement* raw_mux = gst_bin_get_by_name(GST_BIN(pipeline), "raw_mux");
        GstElement* processed_mux =
            gst_bin_get_by_name(GST_BIN(pipeline), "processed_mux");
        GstBus* bus = gst_element_get_bus(pipeline);

        if (!processed_src || !camera_sink || !raw_valve || !processed_valve ||
            !raw_mux || !processed_mux || !bus) {
            setError("Could not retrieve appsrc/appsink/valve/splitmuxsink "
                     "elements or pipeline bus");
            if (processed_src) gst_object_unref(processed_src);
            if (camera_sink) gst_object_unref(camera_sink);
            if (raw_valve) gst_object_unref(raw_valve);
            if (processed_valve) gst_object_unref(processed_valve);
            if (raw_mux) gst_object_unref(raw_mux);
            if (processed_mux) gst_object_unref(processed_mux);
            if (bus) gst_object_unref(bus);
            gst_object_unref(pipeline);
            return false;
        }

        {
            std::lock_guard lock(state_mutex_);
            pipeline_ = pipeline;
            processed_src_ = processed_src;
            camera_sink_ = camera_sink;
            raw_valve_ = raw_valve;
            processed_valve_ = processed_valve;
            raw_mux_ = raw_mux;
            processed_mux_ = processed_mux;
            bus_ = bus;
            audio_active_ = with_audio;
        }

        // Если установленный GStreamer умеет appsrc leaky-type (>= 1.20),
        // включаем отбрасывание самых старых кадров processed: штатный
        // механизм - leaky-очередь ниже, это страховка на случай задержки
        // кодировщика сильнее, чем успевает поглотить очередь.
        if (g_object_class_find_property(G_OBJECT_GET_CLASS(processed_src),
                                         "leaky-type")) {
            guint leaky_downstream = 2; // GST_APP_LEAKY_TYPE_DOWNSTREAM
            g_object_set(processed_src, "leaky-type", leaky_downstream, nullptr);
        }

        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
            GST_STATE_CHANGE_FAILURE) {
            setError(with_audio
                         ? "GStreamer pipeline could not enter PLAYING state "
                           "(проверьте камеру /dev/video* и микрофон)"
                         : "GStreamer pipeline could not enter PLAYING state "
                           "(проверьте камеру /dev/video*)");
            stopInternal(false);
            return false;
        }

        // Прайм-кадр: muxer processed-ветки не должен ждать данных, которые мы
        // сами отдаём только после кадра с камеры. Иначе взаимоблокировка:
        // pipeline ждёт буфер в muxer -> appsink молчит -> кадров нет.
        {
            cv::Mat primer(config_.height, config_.width, CV_8UC3,
                           cv::Scalar(0, 0, 0));
            const gsize primer_bytes =
                static_cast<gsize>(primer.total() * primer.elemSize());
            GstBuffer* primer_buffer =
                gst_buffer_new_allocate(nullptr, primer_bytes, nullptr);
            if (primer_buffer != nullptr) {
                gst_buffer_fill(primer_buffer, 0, primer.data, primer_bytes);
                GST_BUFFER_PTS(primer_buffer) = 0;
                GST_BUFFER_DTS(primer_buffer) = GST_CLOCK_TIME_NONE;
                GST_BUFFER_DURATION(primer_buffer) =
                    GST_SECOND / static_cast<GstClockTime>(config_.fps);
                const GstFlowReturn primer_flow = gst_app_src_push_buffer(
                    GST_APP_SRC(processed_src), primer_buffer);
                if (primer_flow != GST_FLOW_OK) {
                    logMsg("⚠️ Прайм-кадр processed не принят: " +
                           std::string(gst_flow_get_name(primer_flow)));
                } else {
                    std::lock_guard lock(state_mutex_);
                    last_processed_pts_ = 0;
                }
            }
        }

        // Ждём фактического перехода. Без этой проверки preroll-stall выглядел
        // как успешно начатая запись с файлами нулевого размера.
        GstState reached = GST_STATE_VOID_PENDING;
        const GstStateChangeReturn wait_result =
            gst_element_get_state(pipeline, &reached, nullptr, 5 * GST_SECOND);
        if (wait_result == GST_STATE_CHANGE_FAILURE) {
            setError("Пайплайн GStreamer не перешёл в рабочее состояние "
                     "(ошибка элемента). Смотрите предупреждения GStreamer выше.");
            stopInternal(false);
            return false;
        }
        if (wait_result == GST_STATE_CHANGE_ASYNC) {
            logStalledElements(pipeline);
            setError("Пайплайн не вышел в PLAYING за 5 с: не завершён preroll, "
                     "то есть камера или muxer не получают данных. Проверьте "
                     "gst-launch-тест с этой камерой и отключите звук "
                     "(JETBOT_MIC=), если он включён.");
            stopInternal(false);
            return false;
        }

        logMsg(std::string("Запись: пайплайн в состоянии ") +
               gst_element_state_get_name(reached));

        // Применяем текущее состояние задвижек (по умолчанию - открыты).
        (void)setStreamsEnabled(streams_enabled_.load());
        running_ = true;
        return true;
    }

    bool stopInternal(bool send_eos) {
        GstHandle pipeline;
        GstHandle processed_src;
        GstHandle bus;
        bool already_failed = false;
        {
            std::lock_guard lock(state_mutex_);
            running_ = false;
            if (!pipeline_) {
                return !failed_;
            }
            already_failed = failed_;
            pipeline = gstRef(GST_OBJECT(gst_object_ref(pipeline_)));
            if (processed_src_) {
                processed_src =
                    gstRef(GST_OBJECT(gst_object_ref(processed_src_)));
            }
            if (bus_) {
                bus = gstRef(GST_OBJECT(gst_object_ref(bus_)));
            }
        }

        bool ok = true;

        // Если пайплайн уже сломался, EOS ждать бессмысленно: он не придёт, и
        // остановка висела бы 12 секунд. Просто переводим в NULL.
        if (send_eos && !already_failed) {
            if (processed_src) {
                if (gst_app_src_end_of_stream(GST_APP_SRC(processed_src.get())) !=
                    GST_FLOW_OK) {
                    logMsg("⚠️ appsrc EOS не принят; последний сегмент processed "
                           "может быть неполным.");
                    ok = false;
                }
            }
            if (!gst_element_send_event(GST_ELEMENT(pipeline.get()),
                                        gst_event_new_eos())) {
                setError("Не удалось отправить EOS в пайплайн");
                ok = false;
            } else if (bus) {
                GstMessage* message = gst_bus_timed_pop_filtered(
                    GST_BUS(bus.get()), 12 * GST_SECOND,
                    static_cast<GstMessageType>(GST_MESSAGE_EOS |
                                                GST_MESSAGE_ERROR));
                if (!message) {
                    setError("Таймаут ожидания EOS: текущий сегмент может быть "
                            "незавершён");
                    ok = false;
                } else {
                    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                        setError(gstErrorMessage(message));
                        ok = false;
                    }
                    gst_message_unref(message);
                }
            }
        }

        gst_element_set_state(GST_ELEMENT(pipeline.get()), GST_STATE_NULL);

        bool failed = false;
        {
            std::lock_guard lock(state_mutex_);
            teardownLocked();
            failed = failed_;
        }

        return ok && !failed;
    }

    void teardownLocked() {
        if (bus_) { gst_object_unref(bus_); bus_ = nullptr; }
        if (processed_src_) { gst_object_unref(processed_src_); processed_src_ = nullptr; }
        if (camera_sink_) { gst_object_unref(camera_sink_); camera_sink_ = nullptr; }
        if (raw_valve_) { gst_object_unref(raw_valve_); raw_valve_ = nullptr; }
        if (processed_valve_) { gst_object_unref(processed_valve_); processed_valve_ = nullptr; }
        if (raw_mux_) { gst_object_unref(raw_mux_); raw_mux_ = nullptr; }
        if (processed_mux_) { gst_object_unref(processed_mux_); processed_mux_ = nullptr; }
        if (pipeline_) { gst_object_unref(pipeline_); pipeline_ = nullptr; }
        audio_active_ = false;
    }

    bool pollBus(GstObject* bus_object) {
        if (!bus_object) {
            return true;
        }
        GstBus* bus = GST_BUS(bus_object);
        while (GstMessage* message = gst_bus_pop_filtered(
                   bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR |
                                                    GST_MESSAGE_WARNING))) {
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                setError(gstErrorMessage(message));
                gst_message_unref(message);
                return false;
            }
            GError* warning = nullptr;
            gchar* debug = nullptr;
            gst_message_parse_warning(message, &warning, &debug);
            if (warning) {
                g_printerr("GStreamer warning: %s\n", warning->message);
                g_error_free(warning);
            }
            g_free(debug);
            gst_message_unref(message);
        }
        std::lock_guard lock(state_mutex_);
        return !failed_;
    }

    void setError(std::string message) {
        std::lock_guard lock(state_mutex_);
        last_error_ = std::move(message);
        failed_ = true;
    }

    RecordingConfig config_;
    bool audio_requested_{false};
    bool logged_first_frame_{false};

    mutable std::mutex control_mutex_; // сериализует start/stop
    mutable std::mutex state_mutex_;   // короткие критические секции

    GstElement* pipeline_{nullptr};
    GstElement* processed_src_{nullptr};
    GstElement* camera_sink_{nullptr};
    GstElement* raw_valve_{nullptr};
    GstElement* processed_valve_{nullptr};
    GstElement* raw_mux_{nullptr};
    GstElement* processed_mux_{nullptr};
    GstBus* bus_{nullptr};

    std::filesystem::path session_dir_;
    std::string active_camera_device_;
    std::atomic<bool> running_{false};
    std::atomic<bool> streams_enabled_{true};
    std::atomic<bool> audio_active_{false};
    bool failed_{false};
    GstClockTime last_processed_pts_{GST_CLOCK_TIME_NONE};
    std::string last_error_;
};

RecordingManager::RecordingManager(RecordingConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}
RecordingManager::~RecordingManager() = default;

bool RecordingManager::start() { return impl_->start(); }
bool RecordingManager::stop() { return impl_->stop(); }
bool RecordingManager::isRunning() const noexcept { return impl_->isRunning(); }

bool RecordingManager::readCameraFrame(FramePacket& packet,
                                       std::chrono::milliseconds timeout) {
    return impl_->readCameraFrame(packet, timeout);
}

bool RecordingManager::pushProcessedFrame(const cv::Mat& frame,
                                          TimestampNs pts_ns) {
    return impl_->pushProcessedFrame(frame, pts_ns);
}

bool RecordingManager::setStreamsEnabled(bool enabled) {
    return impl_->setStreamsEnabled(enabled);
}

bool RecordingManager::streamsEnabled() const noexcept {
    return impl_->streamsEnabled();
}

bool RecordingManager::audioActive() const noexcept {
    return impl_->audioActive();
}

TimestampNs RecordingManager::timestampNow() const noexcept {
    return impl_->timestampNow();
}

std::string RecordingManager::lastError() const { return impl_->lastError(); }

std::string RecordingManager::currentSessionDirectory() const {
    return impl_->currentSessionDirectory();
}

std::string RecordingManager::latestFinalizedVideo(VideoStream stream,
                                                   const std::string& date) const {
    return impl_->latestFinalizedVideo(stream, date);
}

} // namespace ObserverLoop::Recording
