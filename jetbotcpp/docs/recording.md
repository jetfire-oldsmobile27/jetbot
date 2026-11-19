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

Если детекция выключена (`--no-detection`) или детектор не создан (нет модели),
запись автоматически становится непрерывной: иначе «по событию» означало бы
60 секунд после старта и тишину. В лог попадает предупреждение об этом.

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

## Приведение формата камеры

Камера не обязана поддерживать ровно `width/height/fps` из конфигурации (USB
камеры часто умеют только 640x480@30 или только MJPEG). Поэтому жёсткие caps
на `v4l2src` не навязываются: после источника стоят `videoconvert`,
`videoscale`, `videorate` и только затем caps
`video/x-raw,format=I420,width=W,height=H,framerate=F/1`. Иначе `v4l2src`
падает с `not-negotiated`.

Следствие: **оба** потока (включая raw) нормализуются к размеру и частоте из
конфигурации. Если нужен именно нативный поток камеры — уберите `videoscale`/
`videorate` из строки запуска под конкретную камеру и задайте её caps.

## Выбор камеры

По умолчанию (`camera_device = "auto"`) устройство ищется через
`GstDeviceMonitor`: берётся V4L2-источник, отдающий `video/x-raw` (формат и
размер приводятся в пайплайне, см. раздел выше). Это закрывает случай, когда
после перезагрузки камера оказалась `/dev/video2` вместо `/dev/video0`.

Приоритет настройки: `JETBOT_CAMERA` (полный путь, жёстко) → `CAM_INDEX`
(предпочтение: принимается, если такой узел есть среди видеоисточников) →
автопоиск. Если камера не найдена, в лог попадает диагностика и подсказка
задать `JETBOT_CAMERA`.

OpenCV-путь (режим `--no-recording` и откат при неудачном старте записи)
перебирает индексы 0…9, как и раньше.

## Переменные окружения

| Переменная | Обязательна | Назначение |
|---|---|---|
| `VK_ACCESS_TOKEN`, `VK_GROUP_ID` | да | VK-бот |
| `LOCAL_MACHINE_IP`, `LOCAL_MACHINE_PORT` | для `/power` | управление локальной машиной |
| `JETBOT_MIC` | для аудио | ALSA-устройство микрофона (`arecord -l`). Без него запись без звука |
| `CAM_INDEX` | нет | предпочтение при автопоиске (`/dev/video<N>`), не жёсткое требование |
| `JETBOT_CAMERA` | нет | жёстко задать узел камеры, приоритетнее `CAM_INDEX` |
| `ONNX_MODEL_PATH`, `COCO_NAMES_PATH` | если модели не рядом с бинарником | модель/классы YOLO; при отсутствии берутся `yolov10n.onnx` и `coco.names` из каталога бинарника |
| `JETBOT_ENCODER_CHAIN` | на SoC с аппаратным H.264 | своя цепочка энкодера вместо `x264enc` |
| `OLLAMA_IP` | для `/ask` | сервер распознавания |
| `TG_API_TOKEN` | нет | Telegram (обработчики сейчас закомментированы) |
| `HOME` | да | каталог записей `$HOME/jetbot/video` |

Пример:

```bash
export VK_ACCESS_TOKEN=...  VK_GROUP_ID=...
export LOCAL_MACHINE_IP=192.168.31.149 LOCAL_MACHINE_PORT=8083
export JETBOT_MIC=hw:1,0                       # arecord -l
export ONNX_MODEL_PATH=$HOME/jetbot/yolov10n.onnx
export COCO_NAMES_PATH=$HOME/jetbot/coco.names
```

Аудио включается **только** если задан `JETBOT_MIC`. По умолчанию
`microphone_device` пустой, и запись идёт без звуковой дорожки: это осознанно,
потому что зависший `alsasrc` (устройство открылось, но не отдаёт данные)
блокирует preroll всего пайплайна — файлы создаются, но остаются нулевого
размера, и не пишется даже видео.

Перед сборкой пайплайна устройство проверяется пробой
(`alsasrc ! audioconvert ! audioresample ! fakesink`, несколько буферов). Если
проба не прошла, запись стартует без звука и в лог пишется причина.
Текущее состояние видно в `/status` («Микрофон: …»).

## Диагностика: файлы нулевого размера

Старт записи теперь проверяет, что пайплайн реально дошёл до PLAYING
(`gst_element_get_state`, 5 с). Если preroll не завершён, `start()` возвращает
ошибку, в лог пишется причина, и бот продолжает работу через OpenCV-захват —
пустые «сессии» больше не считаются записью.

Если `camera_000000.mkv` / `processed_000000.mkv` есть, но 0 байт:

1. Ищите в логе строку `Запись: первый кадр камеры WxH`. Если её нет — пайплайн
   не выдал ни одного кадра, и обычно виновата аудиоветка:
   `JETBOT_MIC=` (пусто) и перезапуск покажут, так ли это.
2. `Пайплайн не вышел в PLAYING за 5 с` — preroll не завершён: камера или
   muxer не получают данных. Смотрите пункт 3 и тест из пункта 4.
3. `10 с без кадров от пайплайна` — пайплайн в PLAYING, но appsink пуст;
   проверьте тот же `gst-launch`-тест.
4. Проверьте камеру отдельно:

   ```bash
   v4l2-ctl -d /dev/videoN --list-formats-ext
   gst-launch-1.0 -v v4l2src device=/dev/videoN num-buffers=50 ! \
     videoconvert ! videoscale ! videorate ! \
     video/x-raw,format=I420,width=640,height=480,framerate=25/1 ! \
     x264enc tune=zerolatency speed-preset=ultrafast bitrate=900 ! h264parse ! \
     matroskamux streamable=true ! filesink location=/tmp/t.mkv
   ```

5. Проверьте микрофон отдельно: `arecord -D "$JETBOT_MIC" -f S16_LE -r 48000 -c 1 -d 5 /tmp/mic.wav`.
6. `Can't record audio fast enough` в логе — ALSA не успевает отдавать буферы;
   такая ветка тоже ведёт к простою muxer-а, лечится сменой устройства
   (`plughw:X,Y` вместо `hw:X,Y`) или отключением звука.

## Производительность на Orange Pi Zero 2W

SoC H618 не имеет доступного аппаратного H.264-энкодера в mainline, поэтому
`x264enc` работает software, и таких энкодеров в пайплайне два (raw и
processed), плюс YOLO. `mpph264enc` — только Rockchip, `JETBOT_ENCODER_CHAIN` на
этой плате не поможет.

Если не хватает CPU:
- снизить `RecordingConfig::fps`/`width`/`height` и `video_bitrate_kbps`;
- запустить без детекции (`--no-detection`): processed-ветка будет без
  разметки, но кодирование останется;
- запустить без записи (`--no-recording`) — тогда работает прежний
  OpenCV-захват и фото/статус.

## Обязательные элементы GStreamer

При старте пайплайн проверяет фабрики элементов и сообщает список отсутствующих:

```bash
for e in v4l2src tee queue valve videoconvert videoscale videorate \
         appsink appsrc splitmuxsink matroskamux filesink h264parse x264enc \
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
