# Запись видео в JetBot (GStreamer)

Подсистема живёт в `src/ObserverLoop/Recording/` и заменяет прежний
`ObserverLoop::VideoRecorder`.

## Что пишется

Один пайплайн GStreamer, один владелец камеры (`/dev/video0`):

```
v4l2src ─┬─ queue ─ valve(raw) ─ x264enc ─ splitmuxsink → camera_%06d.mkv
         └─ queue(leaky) ─ videoconvert/BGR ─ appsink ─→ OpenCV/YOLO/эффекты
                                                          │
                        appsrc ← pushProcessedFrame(pts)  │
                                                          ▼
                queue(leaky) ─ valve(processed) ─ x264enc ─ splitmuxsink → processed_%06d.mkv

alsasrc ─ audioconvert ─ audioresample ─ opusenc ─ tee ─┬─ raw_mux.audio
                                                        └─ processed_mux.audio
```

- `camera_*.mkv` — исходный поток камеры, пишется до YOLO и эффектов, поэтому
  медленный инференс не влияет на него.
- `processed_*.mkv` — кадры после YOLO и эффектов, отправляются с **PTS
  исходного кадра** (`packet.pts_ns`). Если YOLO отстаёт, в файле будут
  пропуски по времени, но не ускоренное воспроизведение.
- Звук — одна дорожка Opus с USB-микрофона в оба контейнера.

Раскладка файлов:

```
$HOME/jetbot/video/
  session_20261120_010512_12345/
    camera_000000.mkv  camera_000001.mkv  ...
    processed_000000.mkv  processed_000001.mkv  ...
```

Сегменты режутся `splitmuxsink` каждые 15 с; `matroskamux streamable=true` и
`filesink o-sync=true`. При внезапном отключении питания пострадает только
текущий сегмент, предыдущие остаются отдельными файлами.

## Политика записи

- `/unstopable off` (по умолчанию, `unstopable_mode=false`) — запись по
  обнаружению: сегменты пишутся, пока идёт событие. Событие активно, если
  детектор что-то нашёл; ещё `RECORDING_DURATION` секунд после последнего
  обнаружения и всё время `INITIAL_RECORDING_DURATION` с запуска.
- `/unstopable on` — непрерывная запись.

Переключение выполняется `valve`-задвижками перед энкодерами: камера и
пайплайн работают всегда, в muxer-ы кадры попадают только при открытой
задвижке. Поэтому переключение мгновенное и не требует перезапуска записи.

Флаг `RecordingConfig::raw_always_recording` (по умолчанию `false`):
- `false` — по событию пишутся оба потока;
- `true` — `camera_*.mkv` пишется непрерывно (≈0.4 ГБ/ч при 900 кбит/с),
  а по событию открывается только `processed_*.mkv`.

## Перегрузка обработанной ветки

- `appsrc block=false max-bytes=12582912` — основной цикл не блокируется;
- очередь перед энкодером ограничена `processed_queue_buffers` (по умолчанию 8)
  и помечена `leaky=downstream` — при отставании кодировщика отбрасываются
  самые старые кадры processed;
- raw-ветка не ограничена и пишется с частотой камеры.

## Запуск и настройка

```bash
export JETBOT_CAMERA=/dev/video0          # по умолчанию /dev/video<CAM_INDEX>
export JETBOT_MIC=hw:1,0                  # arecord -l покажет card/device
# только при наличии аппаратного энкодера:
# export JETBOT_ENCODER_CHAIN="videoconvert ! video/x-raw,format=NV12 ! \
#   mpph264enc ! h264parse config-interval=-1 ! \
#   video/x-h264,stream-format=avc,alignment=au"
```

Если `JETBOT_MIC` не задан, используется ALSA-устройство `default`, и в лог
пишется подсказка.

Отсутствие микрофона не лишает видеозаписи: если аудиоветка не поднялась,
`RecordingManager::start()` повторяет запуск без звука и пишет об этом в лог.
Текущее состояние видно в `/status` («Микрофон: …»).

## Обязательные элементы GStreamer

При старте пайплайн проверяет фабрики элементов и сообщает список отсутствующих:

```bash
for e in v4l2src tee queue valve videoconvert appsink appsrc splitmuxsink \
         matroskamux filesink h264parse x264enc \
         alsasrc audioconvert audioresample opusenc; do
  gst-inspect-1.0 "$e" >/dev/null || echo "MISSING: $e"
done
```

Пакеты (Debian/Armbian): `gstreamer1.0-plugins-base`, `-good`, `-bad`, `-ugly`,
`gstreamer1.0-libav`, плюс `libgstreamer1.0-dev`,
`libgstreamer-plugins-base1.0-dev` для сборки.

## Сборка

```bash
conan install . --output-folder=build --build=missing \
  --profile:build=default --profile:host=./aarch64-rpi3-linux-gnu
cmake --preset conan-release --fresh      # переконфиг нужен: CMake GLOB
cmake --build --preset conan-release
```

GStreamer подключается через `pkg_check_modules`; при кросс-сборке pkg-config
должен смотреть в sysroot целевой системы.

## Проверка на устройстве

1. `v4l2-ctl -d /dev/video0 --list-formats-ext` — камера должна поддерживать
   640x480@25 (иначе поменяйте `JETBOT_*`/`RecordingConfig`).
2. `arecord -D "$JETBOT_MIC" -f S16_LE -r 48000 -c 1 -d 5 /tmp/mic.wav` — микрофон.
3. Запуск: в логе должно появиться `Recording session: ...`, в сессии — пары
   `camera_*`/`processed_*` файлов.
4. `ffprobe -show_streams session_*/camera_000000.mkv` — видео H.264 + аудио Opus.
5. Остановка — Ctrl+C (EOS и финализация сегментов). `kill -9` сегмент не
   финализирует.
6. `/last` в VK отдаёт последний **завершённый** сегмент (текущий файл
   исключается через `current-file` у `splitmuxsink`) и запись не останавливает.

## Известные ограничения

- Синхронизация аудио и видео программная: у камеры и USB-микрофона нет общего
  аппаратного clock, возможен медленный дрейф.
- Два одновременных `x264enc` на ARM могут быть дорогими; на Rockchip с
  `mpph264enc` используйте `JETBOT_ENCODER_CHAIN`.
- Ошибка аудио после успешного старта останавливает запись: перезапуск без
  звука делается только при старте пайплайна.
- `ObserverLoop/VideoRecorder.hpp` больше не используется.
