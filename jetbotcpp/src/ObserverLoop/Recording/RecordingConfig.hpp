#pragma once

#include <chrono>
#include <filesystem>
#include <string>

namespace ObserverLoop::Recording {

struct RecordingConfig {
    std::filesystem::path output_dir{"videos"};

    // Явное устройство V4L2. Значение "auto" включает автопоиск по caps
    // (GstDeviceMonitor), пустая строка равнозначна "auto".
    std::string camera_device{"/dev/video0"};

    // Необязательное предпочтение при автопоиске: узел используется, только
    // если он существует и поддерживает запрошенные width/height/fps. Удобно,
    // когда CAM_INDEX после перезагрузки меняется.
    std::string camera_preferred_device{};

    // Явное устройство ALSA (например "hw:1,0" или "plughw:1,0").
    // Пустая строка = запись видео без звука: отсутствие микрофона не должно
    // лишать пользователя видеозаписи.
    std::string microphone_device{"default"};

    int width{640};
    int height{480};
    int fps{25};

    int video_bitrate_kbps{900};
    int audio_bitrate_bps{64000};
    std::chrono::seconds segment_duration{15};

    // O_SYNC forces synchronous writes through filesink. It can reduce
    // throughput significantly on SD cards; benchmark on the target device.
    bool synchronous_files{true};

    // Generic ARM/Linux fallback. For a known SoC this can be replaced with
    // an available hardware encoder chain, e.g. "mpph264enc ! h264parse".
    std::string video_encoder_chain{};

    // Сколько кадров processed разрешено накопить в очереди перед энкодером.
    // При переполнении отбрасываются самые старые кадры (leaky=downstream),
    // поэтому медленное кодирование не блокирует основной цикл обработки.
    int processed_queue_buffers{8};

    // Предел очереди appsrc в байтах. Служит страховкой, а не основным
    // механизмом: штатно кадры отбрасывает leaky-очередь.
    int processed_appsrc_max_bytes{12582912};

    // true  - исходный поток (camera_*.mkv) пишется непрерывно, а по событию
    //         открывается только processed_*.mkv;
    // false - по событию пишутся оба потока (поведение по умолчанию).
    bool raw_always_recording{false};
};

} // namespace ObserverLoop::Recording
