#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    int camera_index = 0;
    if (argc > 1) {
        camera_index = std::stoi(argv[1]);
    }
    
    cv::VideoCapture cap(camera_index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру " << camera_index << std::endl;
        return 1;
    }
    
    // Принудительно устанавливаем YUYV (поддерживается libcamera)
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y','U','Y','V'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    // Вывод параметров камеры
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    
    std::cout << "Камера открыта:" << std::endl;
    std::cout << "  Ширина: " << width << std::endl;
    std::cout << "  Высота: " << height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  FOURCC: " << char(fourcc & 0xFF) 
              << char((fourcc >> 8) & 0xFF)
              << char((fourcc >> 16) & 0xFF)
              << char((fourcc >> 24) & 0xFF) << std::endl;
    
    // Захват одного кадра для проверки
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Не удалось захватить кадр" << std::endl;
        return 1;
    }
    
    std::cout << "Кадр получен, размер: " << frame.cols << "x" << frame.rows 
              << ", каналов: " << frame.channels() << std::endl;
    
    cv::imwrite("test_capture.jpg", frame);
    std::cout << "Изображение сохранено в test_capture.jpg" << std::endl;
    
    return 0;
}