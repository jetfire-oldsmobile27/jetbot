#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

namespace Utility {

class Effects {
public:
  static cv::Mat generateMemoryDump(int width, int height) {
    int dump_width = width / 6;
    cv::Mat dump = cv::Mat::zeros(height, dump_width, CV_8UC4); // BGRA

    // Полупрозрачный фон
    for (int i = 0; i < dump.rows; ++i) {
      for (int j = 0; j < dump.cols; ++j) {
        dump.at<cv::Vec4b>(i, j) = cv::Vec4b(20, 0, 20, 128);
      }
    }

    // Заголовок
    cv::putText(dump, "MEM DUMP:", cv::Point(10, 20),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                cv::Scalar(0, 200, 255, 255), 1);

    // Генерация случайных hex-данных
    int line_height = 20;
    int num_lines = std::min(12, (height - 60) / line_height);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> addr_dist(0x1000, 0xFFFF);
    std::uniform_int_distribution<> byte_dist(0, 255);

    for (int i = 0; i < num_lines; ++i) {
      int y = 40 + i * line_height;

      // Генерация адреса
      std::stringstream ss;
      ss << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
         << addr_dist(gen);
      std::string address = ss.str();

      // Генерация значений
      std::string values;
      for (int j = 0; j < 4; ++j) {
        ss.str("");
        ss << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
           << byte_dist(gen);
        values += ss.str() + " ";
      }

      cv::putText(dump, address + ": " + values, cv::Point(10, y),
                  cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                  cv::Scalar(0, 200, 255, 255), 1);
    }

    return dump;
  }

  // Генерация звука обнаружения (сохранение в WAV-файл)
  static void generateDetectSound(const std::string &filename) {
    // Параметры звука
    int sample_rate = 22050;
    float duration = 0.3f;
    int num_samples = static_cast<int>(sample_rate * duration);

    // Создаем директорию, если её нет
    std::filesystem::create_directories(
        std::filesystem::path(filename).parent_path());

    // WAV-заголовок (простой 16-битный моно)
    char header[44];
    memset(header, 0, 44);

    // RIFF chunk descriptor
    header[0] = 'R';
    header[1] = 'I';
    header[2] = 'F';
    header[3] = 'F';
    int32_t chunk_size = 36 + num_samples * 2; // 36 + данные
    memcpy(header + 4, &chunk_size, 4);
    header[8] = 'W';
    header[9] = 'A';
    header[10] = 'V';
    header[11] = 'E';

    // Format subchunk
    header[12] = 'f';
    header[13] = 'm';
    header[14] = 't';
    header[15] = ' ';
    int32_t subchunk1_size = 16; // 16 для PCM
    memcpy(header + 16, &subchunk1_size, 4);
    int16_t audio_format = 1; // PCM
    memcpy(header + 20, &audio_format, 2);
    int16_t num_channels = 1; // Моно
    memcpy(header + 22, &num_channels, 2);
    memcpy(header + 24, &sample_rate, 4);
    int32_t byte_rate = sample_rate * num_channels * 2; // 16 бит
    memcpy(header + 28, &byte_rate, 4);
    int16_t block_align = num_channels * 2;
    memcpy(header + 32, &block_align, 2);
    int16_t bits_per_sample = 16;
    memcpy(header + 34, &bits_per_sample, 2);

    // Data subchunk
    header[36] = 'd';
    header[37] = 'a';
    header[38] = 't';
    header[39] = 'a';
    int32_t data_size = num_samples * 2;
    memcpy(header + 40, &data_size, 4);

    // Генерируем звуковые данные
    std::vector<int16_t> sound_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      float wave1 = 0.2f * sinf(2 * M_PI * 800 * t);
      float wave2 = 0.2f * sinf(2 * M_PI * 1200 * t);
      float wave = wave1 + wave2;
      float envelope = expf(-4 * t / duration);
      sound_data[i] =
          static_cast<int16_t>(wave * envelope * 10000.0f); // Громкость 10000
    }

    // Записываем в файл
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(header, 44);
    ofs.write(reinterpret_cast<const char *>(sound_data.data()),
              num_samples * 2);
    ofs.close();
  }

  // Воспроизведение звука
  static void playDetectSound() {
    static bool sound_initialized = false;
    static std::string sound_file = "/tmp/detect_sound.wav";

    if (!sound_initialized) {
      generateDetectSound(sound_file);
      sound_initialized = true;
    }

    system(("nohup aplay -q " + sound_file + " >/dev/null 2>&1 &").c_str());
  }
};
}; // namespace Utility