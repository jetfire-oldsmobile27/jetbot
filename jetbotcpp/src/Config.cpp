#include "Config.hpp"
#include <getopt.h>
#include <iostream>
#include <cstdlib>

ConfigFlags parseCmdlineArgs(int argc, char *argv[]) {
  ConfigFlags flags;
  const struct option long_options[] = {
      {"no-recording", no_argument, nullptr, 'r'},
      {"no-detection", no_argument, nullptr, 'd'},
      {"no-effects", no_argument, nullptr, 'e'},
      {"no-sound", no_argument, nullptr, 's'},
      {"no-cleanup", no_argument, nullptr, 'c'},
      {"no-animation", no_argument, nullptr, 'a'},
      {"no-face", no_argument, nullptr, 'f'},
      {"debug-server", optional_argument, nullptr, 1000},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "h", long_options, nullptr)) != -1) {
    switch (opt) {
    case 'r':
      flags.recording = false;
      break;
    case 'd':
      flags.detection = false;
      break;
    case 'e':
      flags.effects = false;
      break;
    case 's':
      flags.sound = false;
      break;
    case 'c':
      flags.cleanup = false;
      break;
    case 'a':
      flags.animation = false;
      break;
    case 'f':
      flags.face = false;
      break;
    case 1000: // код для --debug-server
      flags.debug_server = true;
      if (optarg) {
        try {
          flags.debug_port = std::stoi(optarg);
        } catch (...) {}
      }
      break;
    case 'h':
      flags.help = true;
      break;
    default:
      break;
    }
  }
  return flags;
}

void printHelp(const char *progname) {
  std::cout
      << "Использование: " << progname << " [ОПЦИИ]\n"
      << "Опции:\n"
      << "  --no-recording      Отключить запись видео на диск\n"
      << "  --no-detection       Отключить детекцию людей (YOLO)\n"
      << "  --no-effects         Отключить визуальные эффекты (красный "
         "оттенок, кинескоп, дамп памяти)\n"
      << "  --no-sound           Отключить звуковой сигнал при обнаружении\n"
      << "  --no-cleanup         Отключить автоматическую очистку старых "
         "видео\n"
      << "  --no-animation       Отключить анимацию при инициализации\n"
      << "  --no-face            Отключить детекцию лиц\n"
      << "  --debug-server[=PORT] Запустить отладочный HTTP сервер для просмотра кадров (порт по умолчанию: 8080)\n"
      << "  --help, -h           Показать эту справку\n";
}