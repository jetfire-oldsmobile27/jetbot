import cv2
import numpy as np
import pygame
import random
import time
import sys
import math
import threading
import queue

# $pip install opencv-python pygame numpy

# Инициализация Pygame для звука
pygame.init()
pygame.mixer.init(frequency=22050, channels=1)

# Функция генерации звука обнаружения
def generate_detect_sound():
    sample_rate = 22050
    duration = 0.3
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave1 = 0.2 * np.sin(2 * np.pi * 800 * t)
    wave2 = 0.2 * np.sin(2 * np.pi * 1200 * t)
    wave = wave1 + wave2
    envelope = np.exp(-4 * t / duration)
    wave = wave * envelope
    return (wave * 32767).astype(np.int16)

# Создаем звук
detect_sound = pygame.mixer.Sound(buffer=generate_detect_sound())
detect_sound.set_volume(0.3)

# Функция для открытия камеры
def open_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Камера с индексом {cam_index} успешно открыта")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        return cap
    return None

# Поиск доступной камеры в отдельном потоке
def find_camera(camera_queue):
    # Сначала пробуем камеру по умолчанию (индекс 0)
    cap = open_camera(0)
    if cap:
        cap.release()
        camera_queue.put([0])
        return
    
    # Если камера 0 не доступна, ищем другие
    cam_indexes = []
    for i in range(1, 10):  # Проверяем индексы 1-9
        cap = open_camera(i)
        if cap:
            cam_indexes.append(i)
            cap.release()
    
    camera_queue.put(cam_indexes)

# Генерация псевдо-дампа памяти
def generate_memory_dump(width, height):
    dump_width = width // 6
    dump = np.zeros((height, dump_width, 4), dtype=np.uint8)  # BGRA
    
    # Полупрозрачный фон
    dump[:, :, 0] = 20   # B
    dump[:, :, 1] = 0    # G
    dump[:, :, 2] = 20   # R
    dump[:, :, 3] = 128  # Alpha
    
    # Заголовок
    cv2.putText(dump, "MEM DUMP:", (10, 20), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 200, 255, 255), 1)
    
    # Генерация случайных hex-данных
    line_height = 20
    num_lines = min(12, (height - 60) // line_height)
    for i in range(num_lines):
        y = 40 + i * line_height
        address = f"{random.randint(0x1000, 0xFFFF):04X}"
        values = ' '.join(f"{random.randint(0, 255):02X}" for _ in range(4))
        cv2.putText(dump, f"{address}: {values}", (10, y), 
                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 200, 255, 255), 1)
    
    return dump

# Класс для управления анимацией и инициализацией
class StartupManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.center_x, self.center_y = width // 2, height // 2
        self.circle_radius = 30
        self.circle_distance = 120
        self.circles = []
        self.animation_phase = 0  # 0: пульсация, 1: основная анимация
        self.start_time = time.time()
        self.initialization_complete = False
        self.initialization_progress = 0
        self.cam_index = None
        self.cap = None
        self.net = None
        self.classes = []
        self.output_layers = []
        self.face_cascade = None
        self.frame = None
        self.loading_text = "INITIALIZING SYSTEM"
        self.pulse_value = 0
        self.camera_error_count = 0
        
        # Создаем 3 круга под углом 120 градусов
        for i in range(3):
            angle = 2 * math.pi * i / 3
            x = self.center_x + int(self.circle_distance * math.cos(angle))
            y = self.center_y + int(self.circle_distance * math.sin(angle))
            self.circles.append({
                "x": x, 
                "y": y, 
                "radius": 0, 
                "pulse_direction": 1
            })
    
    def update_animation(self):
        elapsed = time.time() - self.start_time
        anim_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pulse_speed = 0.5  # Скорость пульсации
        
        # Фаза 0: Пульсация кругов во время инициализации
        if self.animation_phase == 0:
            # Пульсация кругов
            self.pulse_value = math.sin(elapsed * pulse_speed * 2 * math.pi) * 0.2 + 0.8
            pulse_size = int(self.circle_radius * self.pulse_value)
            
            for circle in self.circles:
                # Отрисовка круга
                cv2.circle(anim_frame, (circle["x"], circle["y"]), 
                          pulse_size, (255, 255, 255), -1)
            
            # Отображение прогресса инициализации
            progress_text = f"{self.loading_text}: {int(self.initialization_progress * 100)}%"
            text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = self.height - 50
            cv2.putText(anim_frame, progress_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Переход к следующей фазе после завершения инициализации
            if self.initialization_progress >= 1.0:
                self.animation_phase = 1
                self.start_time = time.time()
        
        # Фаза 1: Основная анимация (схлопывание кругов)
        elif self.animation_phase == 1:
            elapsed = elapsed - 0.5  # Задержка перед началом анимации
            
            # Параметры анимации
            text_alpha = 0
            bar_width_left = 0
            bar_width_right = 0
            video_alpha = 0
            max_bar_width = self.width // 2 - self.circle_distance
            
            # Фаза 1.1: Вращение кругов и появление полос (0-1.2 сек)
            if elapsed < 1.2:
                rotation = elapsed * 1.5  # Быстрое вращение
                for i, circle in enumerate(self.circles):
                    angle = 2 * math.pi * i / 3 + rotation
                    circle["x"] = self.center_x + int(self.circle_distance * math.cos(angle))
                    circle["y"] = self.center_y + int(self.circle_distance * math.sin(angle))
                
                # Появление текста
                text_alpha = min(1.0, elapsed / 1.2)
                
                # Появление полос
                bar_progress = min(1.0, elapsed / 1.2)
                bar_width_left = int(max_bar_width * bar_progress)
                bar_width_right = int(max_bar_width * bar_progress)
            
            # Фаза 1.2: Схлопывание кругов и появление видео (1.2-2.0 сек)
            elif elapsed < 2.0:
                progress = (elapsed - 1.2) / 0.8
                for circle in self.circles:
                    # Плавное перемещение к центру
                    circle["x"] = int(circle["x"] + (self.center_x - circle["x"]) * progress)
                    circle["y"] = int(circle["y"] + (self.center_y - circle["y"]) * progress)
                
                # Появление видео
                video_alpha = min(1.0, (elapsed - 1.5) / 0.5)
            
            # Отрисовка кругов
            progress = min(1.0, (elapsed - 1.2) / 0.8)
            for circle in self.circles:
                radius = int(self.circle_radius * (1 - progress) * self.pulse_value)
                cv2.circle(anim_frame, (circle["x"], circle["y"]), 
                          radius, (255, 255, 255), -1)
            
            # Отрисовка полос
            if bar_width_left > 0:
                # Левая полоса
                cv2.rectangle(anim_frame, (0, self.center_y - 2), 
                             (bar_width_left, self.center_y + 2), (255, 255, 255), -1)
                # Правая полоса
                cv2.rectangle(anim_frame, (self.width - bar_width_right, self.center_y - 2), 
                             (self.width, self.center_y + 2), (255, 255, 255), -1)
            
            # Отрисовка текста
            if text_alpha > 0:
                text = "JetVision Systems"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (self.width - text_size[0]) // 2
                text_y = self.height - 50
                cv2.putText(anim_frame, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (int(255 * text_alpha), int(255 * text_alpha), int(255 * text_alpha)), 
                            2, cv2.LINE_AA)
            
            # Если есть кадр с камеры, смешиваем с анимацией
            if self.frame is not None and video_alpha > 0:
                # Применяем эффект кинескопа к видео
                video_frame = self.frame.copy()
                video_frame[1::2, :, :] = 0  # Эффект чересстрочной развертки
                
                # Смешиваем анимацию и видео
                anim_frame = cv2.addWeighted(video_frame, video_alpha, anim_frame, 1 - video_alpha, 0)
        
        return anim_frame
    
    def initialize(self, camera_queue):
        # Шаг 1: Получение индекса камеры
        self.loading_text = "INITIALIZING CAMERA"
        
        # Ждем результаты поиска камеры
        while camera_queue.empty():
            time.sleep(0.1)
        
        cam_indexes = camera_queue.get()
        if cam_indexes:
            self.cam_index = cam_indexes[0]
            print(f"Используется камера с индексом {self.cam_index}")
            self.initialization_progress = 0.3
        else:
            print("Не найдено доступных камер")
        
        # Шаг 2: Инициализация YOLO
        self.loading_text = "LOADING OBJECT DETECTOR"
        try:
            # Пробуем загрузить YOLO-tiny
            net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
            with open("coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Проверяем доступность CUDA
            cuda_available = False
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    cuda_available = True
                    print("Используется CUDA ускорение")
                else:
                    print("CUDA недоступна, используется CPU")
            except:
                print("Ошибка проверки CUDA, используется CPU")
            
            # Если CUDA недоступна, используем CPU
            if not cuda_available:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Используется CPU")
            
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
            
            self.net = net
            self.classes = classes
            self.output_layers = output_layers
            self.initialization_progress = 0.6
        except Exception as e:
            print(f"Не удалось загрузить YOLO: {e}")
            self.net = None
        
        # Шаг 3: Загрузка каскада для лиц
        self.loading_text = "LOADING FACE DETECTOR"
        face_cascade = cv2.CascadeClassifier()
        if face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
            self.face_cascade = face_cascade
            print("Детектор лиц загружен")
            self.initialization_progress = 0.8
        else:
            print("Не удалось загрузить детектор лиц")
        
        # Шаг 4: Открытие камеры (если индекс известен)
        self.loading_text = "CONNECTING CAMERA"
        if self.cam_index is not None:
            # Пробуем открыть камеру
            self.cap = open_camera(self.cam_index)
            if self.cap:
                # Получаем первый кадр
                ret, frame = self.cap.read()
                if ret:
                    self.frame = cv2.resize(frame, (self.width, self.height))
        
        self.initialization_progress = 1.0
        self.initialization_complete = True

# Класс для трекинга объектов
class ObjectTracker:
    def __init__(self, max_disappeared=5):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid, bbox):
        self.objects[self.next_id] = (centroid, bbox)
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        used_detections = [False] * len(detections)
        
        for object_id, (centroid, bbox) in list(self.objects.items()):
            min_dist = float('inf')
            min_idx = -1
            
            for i, (det_centroid, det_bbox) in enumerate(detections):
                if used_detections[i]:
                    continue
                
                dist = np.linalg.norm(np.array(centroid) - np.array(det_centroid))
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    min_idx = i
            
            if min_idx != -1:
                self.objects[object_id] = detections[min_idx]
                self.disappeared[object_id] = 0
                used_detections[min_idx] = True
            else:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        for i, (centroid, bbox) in enumerate(detections):
            if not used_detections[i]:
                self.register(centroid, bbox)
        
        return self.objects

# Оптимизированная детекция с помощью YOLO
def detect_people_yolo(frame, net, output_layers, classes, conf_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    
    # Создаем блоб с уменьшенным размером для ускорения
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold and classes[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    people = []
    if indices is not None:
        # Обработка разных форматов возвращаемого значения
        if isinstance(indices, tuple):
            # Пустой кортеж - нет обнаружений
            if len(indices) == 0:
                indices = np.array([])
            # Кортеж с одним элементом (массивом)
            elif len(indices) == 1:
                indices = indices[0]
            # Неожиданный формат
            else:
                indices = np.array(indices)
        
        # Преобразование в плоский массив
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
        else:
            indices = np.array(indices).reshape(-1)
        
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            centroid = (x + w//2, y + h//2)
            people.append((centroid, (x, y, w, h)))
    
    return people

# Основная функция обработки видео
def jet_vision():
    # Параметры окна
    width, height = 800, 600
    cv2.namedWindow('Jet Vision', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Jet Vision', width, height)
    
    # Очередь для передачи результатов поиска камеры
    camera_queue = queue.Queue()
    
    # Запускаем поиск камер в отдельном потоке
    camera_thread = threading.Thread(target=find_camera, args=(camera_queue,))
    camera_thread.daemon = True
    camera_thread.start()
    
    # Создаем менеджер инициализации и анимации
    startup_manager = StartupManager(width, height)
    
    # Запускаем инициализацию в отдельном потоке
    init_thread = threading.Thread(target=startup_manager.initialize, args=(camera_queue,))
    init_thread.daemon = True
    init_thread.start()
    
    # Основной цикл анимации инициализации
    while not startup_manager.initialization_complete or startup_manager.animation_phase < 1:
        # Обновляем анимацию
        anim_frame = startup_manager.update_animation()
        cv2.imshow('Jet Vision', anim_frame)
        
        # Выход по ESC
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            return
    
    # Проверяем, что камера инициализирована
    if startup_manager.cap is None or not startup_manager.cap.isOpened():
        # Попробуем открыть камеру по умолчанию (индекс 0)
        print("Пробуем открыть камеру по умолчанию (индекс 0)")
        startup_manager.cap = open_camera(0)
        if startup_manager.cap:
            print("Камера по умолчанию успешно открыта")
        else:
            print("Ошибка: камера не доступна")
            cv2.destroyAllWindows()
            return
    
    # Инициализация трекера
    tracker = ObjectTracker(max_disappeared=3)
    frame_counter = 0
    last_dump_time = 0
    memory_dump = None
    
    # Плавный переход от анимации к основному интерфейсу
    transition_start = time.time()
    transition_duration = 1.0
    last_frame = startup_manager.frame if startup_manager.frame is not None else np.zeros((height, width, 3), dtype=np.uint8)
    
    while time.time() - transition_start < transition_duration:
        # Чтение кадра с камеры
        ret, frame = startup_manager.cap.read()
        if ret:
            # Подготовка кадра для отображения
            small_frame = cv2.resize(frame, (640, 480))
            display_frame = cv2.resize(small_frame, (width, height))
            last_frame = display_frame.copy()
        
        # Плавное исчезновение анимации
        alpha = 1.0 - min(1.0, (time.time() - transition_start) / transition_duration)
        
        # Создаем кадр с плавным переходом
        anim_frame = startup_manager.update_animation()
        blend_frame = cv2.addWeighted(last_frame, 1 - alpha, anim_frame, alpha, 0)
        cv2.imshow('Jet Vision', blend_frame)
        
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            return
    
    # Основной цикл обработки видео
    while True:
        start_time = time.time()
        
        # Чтение кадра с камеры
        ret, frame = startup_manager.cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            # Попробуем переоткрыть камеру
            startup_manager.cap.release()
            startup_manager.cap = open_camera(startup_manager.cam_index)
            if startup_manager.cap and startup_manager.cap.isOpened():
                print("Камера переоткрыта")
                continue
            else:
                print("Не удалось переоткрыть камеру")
                break
        
        # Подготовка кадра для отображения
        small_frame = cv2.resize(frame, (640, 480))
        display_frame = cv2.resize(small_frame, (width, height))
        
        # 1. Применение красных оттенков
        red_channel = display_frame[:, :, 2]
        processed = np.zeros_like(display_frame)
        processed[:, :, 2] = red_channel
        processed = cv2.addWeighted(display_frame, 0.3, processed, 0.7, 0)
        
        # 2. Эффект кинескопа (только к видео)
        processed[1::2, :, :] = 0
        
        # 3. Генерация дампа памяти раз в секунду
        current_time = time.time()
        if current_time - last_dump_time > 1.0 or memory_dump is None:
            memory_dump = generate_memory_dump(width, height)
            last_dump_time = current_time
        
        # 4. Конвертируем в BGRA для наложения прозрачности
        processed_bgra = cv2.cvtColor(processed, cv2.COLOR_BGR2BGRA)
        
        # 5. Наложение дампа памяти
        if memory_dump is not None:
            dump_width = width // 6
            dump_height = min(height, memory_dump.shape[0])
            dump = memory_dump[:dump_height, :dump_width]
            
            # Быстрое наложение с прозрачностью
            roi = processed_bgra[:dump_height, :dump_width]
            alpha = dump[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = (dump[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8)
        
        # 6. Детекция людей (используем YOLO только каждый 3-й кадр)
        detections = []
        
        if frame_counter % 3 == 0 and startup_manager.net is not None:
            try:
                people = detect_people_yolo(
                    small_frame, 
                    startup_manager.net, 
                    startup_manager.output_layers, 
                    startup_manager.classes
                )
                detections = people
            except Exception as e:
                print(f"Ошибка детекции людей: {e}")
                # Переключаемся на CPU если ошибка CUDA
                if "CUDA" in str(e):
                    print("Переключаем YOLO на CPU")
                    startup_manager.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    startup_manager.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Обновляем трекер
        tracked_objects = tracker.update(detections)
        
        # 7. Детекция лиц
        detected_faces = []
        if startup_manager.face_cascade is not None and tracked_objects:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            for object_id, (centroid, (x, y, w, h)) in tracked_objects.items():
                roi_y = max(0, y)
                roi_h = min(int(h * 0.7), 480 - roi_y)
                roi_x = max(0, x)
                roi_w = min(w, 640 - roi_x)
                
                if roi_h > 30 and roi_w > 30:
                    roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    try:
                        faces = startup_manager.face_cascade.detectMultiScale(
                            roi, 
                            scaleFactor=1.05,
                            minNeighbors=5,
                            minSize=(40, 40)
                        )
                        
                        for (fx, fy, fw, fh) in faces:
                            detected_faces.append((roi_x + fx, roi_y + fy, fw, fh))
                    except Exception as e:
                        print(f"Ошибка детекции лиц: {e}")
        
        # 8. Отрисовка результатов
        for object_id, (centroid, (x, y, w, h)) in tracked_objects.items():
            # Масштабирование координат
            x = int(x * width / 640)
            y = int(y * height / 480)
            w = int(w * width / 640)
            h = int(h * height / 480)
            centroid = (int(centroid[0] * width / 640), int(centroid[1] * height / 480))
            
            # Рисуем контур человека
            cv2.rectangle(processed_bgra, (x, y), (x + w, y + h), (0, 0, 255, 255), 2)
            
            # Информационная панель
            if w > 60 and h > 100:
                cv2.rectangle(processed_bgra, (x, y - 20), (x + 100, y), (0, 0, 0, 255), -1)
                cv2.putText(processed_bgra, f"Human: {random.randint(85, 99)}%", 
                           (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255, 255), 1)
            
            # Перекрестие
            center_x, center_y = centroid
            cv2.line(processed_bgra, (center_x - 15, center_y), (center_x + 15, center_y), 
                    (0, 255, 0, 255), 1)
            cv2.line(processed_bgra, (center_x, center_y - 15), (center_x, center_y + 15), 
                    (0, 255, 0, 255), 1)
        
        for (x, y, w, h) in detected_faces:
            x = int(x * width / 640)
            y = int(y * height / 480)
            w = int(w * width / 640)
            h = int(h * height / 480)
            
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(processed_bgra, center, axes, 0, 0, 360, (0, 255, 0, 255), 1)
            
            cv2.line(processed_bgra, (center[0]-10, center[1]), (center[0]+10, center[1]), 
                    (0, 255, 0, 255), 1)
            cv2.line(processed_bgra, (center[0], center[1]-10), (center[0], center[1]+10), 
                    (0, 255, 0, 255), 1)
        
        # 9. Звук обнаружения
        if len(tracked_objects) > 0 and frame_counter % 5 == 0:
            detect_sound.play()
        
        # 10. Информационный HUD
        cv2.rectangle(processed_bgra, (width - 180, 10), (width - 10, 70), (0, 0, 0, 200), -1)
        fps = int(1 / (time.time() - start_time + 0.001))
        cv2.putText(processed_bgra, f"TARGETS: {len(tracked_objects)}", 
                   (width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255, 255), 1)
        cv2.putText(processed_bgra, f"FPS: {fps}", 
                   (width - 170, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255, 255), 1)
        
        # 11. Отображение результата
        processed_display = cv2.cvtColor(processed_bgra, cv2.COLOR_BGRA2BGR)
        cv2.imshow('Jet Vision', processed_display)
        
        frame_counter += 1
        
        # Выход по ESC
        if cv2.waitKey(1) == 27:
            break
    
    if startup_manager.cap:
        startup_manager.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    jet_vision()