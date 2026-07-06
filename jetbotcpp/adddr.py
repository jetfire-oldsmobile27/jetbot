import socket
import json
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt

def receive_frame(host='192.168.31.152', port=8080):
    """Получает один кадр с сервера"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Подключаемся к серверу
        client_socket.connect((host, port))
        
        # Читаем длину сообщения (4 байта, big-endian)
        length_data = client_socket.recv(4)
        if not length_data:
            print("Не удалось получить длину сообщения")
            return None
        
        msg_length = int.from_bytes(length_data, 'big')
        print(f"Ожидается сообщение длиной: {msg_length} байт")
        
        # Читаем JSON данные
        json_data = b''
        while len(json_data) < msg_length:
            chunk = client_socket.recv(min(4096, msg_length - len(json_data)))
            if not chunk:
                break
            json_data += chunk
        
        if len(json_data) != msg_length:
            print(f"Получено только {len(json_data)} из {msg_length} байт")
            return None
        
        # Парсим JSON
        response = json.loads(json_data.decode('utf-8'))
        
        # Декодируем base64 в изображение
        b64_frame = response['frame']
        timestamp = response['timestamp']
        
        # Декодируем base64 строку в байты
        jpeg_bytes = base64.b64decode(b64_frame)
        
        # Конвертируем в изображение PIL
        image = Image.open(io.BytesIO(jpeg_bytes))
        
        return image, timestamp
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return None
    finally:
        client_socket.close()

# Пример использования:
if __name__ == "__main__":
    # Получаем один кадр
    image, timestamp = receive_frame()
    
    if image:
        print(f"Получен кадр, timestamp: {timestamp}")
        print(f"Размер изображения: {image.size}")
        print(f"Режим: {image.mode}")
        
        # Показываем изображение
        plt.imshow(image)
        plt.title(f"Кадр с сервера (timestamp: {timestamp})")
        plt.axis('off')
        plt.show()
        
        # Или сохраняем в файл
        image.save(f"frame_{timestamp}.jpg")
        print(f"Изображение сохранено как frame_{timestamp}.jpg")