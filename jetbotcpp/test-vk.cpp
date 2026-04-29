// test-vk.cpp
#include <iostream>
#include <string>
#include <atomic>
#include <csignal>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include "BotBase.hpp"   // из VK API

using JsonType = nlohmann::json;

std::atomic<bool> running{true};

void signalHandler(int) {
    running = false;
}

// Получение RSS памяти процесса (в МБ)
size_t getCurrentRSS_MB() {
    std::ifstream statm("/proc/self/statm");
    if (!statm) return 0;
    size_t rss_pages = 0;
    statm >> rss_pages; // первое значение - размер виртуальной памяти
    statm >> rss_pages; // второе значение - RSS в страницах
    long page_size = sysconf(_SC_PAGESIZE);
    return (rss_pages * page_size) / (1024 * 1024);
}

// Логирование в файл
void logMemory(const std::string& tag) {
    static std::ofstream log("vk_test_mem.log", std::ios::app);
    if (!log) return;
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    log << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S")
        << " [" << tag << "] RSS = " << getCurrentRSS_MB() << " MB\n";
    log.flush();
}

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Инициализация libcurl (глобально, один раз)
    curl_global_init(CURL_GLOBAL_ALL);

    // Получение токена и ID группы из окружения
    const std::string VK_ACCESS_TOKEN = []() -> std::string {
        const char* env = std::getenv("VK_ACCESS_TOKEN");
        return env ? env : "";
    }();
    const std::string VK_GROUP_ID = []() -> std::string {
        const char* env = std::getenv("VK_GROUP_ID");
        return env ? env : "";
    }();

    if (VK_ACCESS_TOKEN.empty() || VK_GROUP_ID.empty()) {
        std::cerr << "Ошибка: переменные окружения VK_ACCESS_TOKEN и VK_GROUP_ID должны быть установлены." << std::endl;
        return 1;
    }

    logMemory("START");

    vk::base::bot::BotBase bot(VK_GROUP_ID);

    try {
        if (!bot.Auth(VK_ACCESS_TOKEN)) {
            std::cerr << "Ошибка аутентификации VK API." << std::endl;
            return 1;
        }
        std::cout << "VK API аутентификация успешна. Начинаем прослушивание..." << std::endl;
        logMemory("AUTH_OK");
    } catch (const std::exception& e) {
        std::cerr << "Исключение при аутентификации: " << e.what() << std::endl;
        return 1;
    }

    // Периодическое логирование памяти (каждые 10 секунд)
    std::thread memory_logger([]() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (running) logMemory("PERIODIC");
        }
    });

    // Основной цикл обработки событий
    while (running) {
        try {
            auto event = bot.WaitForEvent();

            // Если событие пустое (например, нет обновлений) – пропускаем
            if (event.parameters.is_null()) {
                continue;
            }

            // Обработка только сообщений, чтобы нагрузить API минимально
            if (event.type == vk::base::bot::BotBase::EVENTS::MESSAGE_NEW) {
                try {
                    auto& params = event.parameters;
                    if (params.contains("object") && params["object"].contains("message")) {
                        auto& msg = params["object"]["message"];
                        int64_t peer_id = msg["peer_id"].get<int64_t>();
                        std::string text;
                        if (msg.contains("text") && msg["text"].is_string())
                            text = msg["text"].get<std::string>();

                        // Эхо-ответ
                        JsonType reply_params;
                        reply_params["peer_id"] = std::to_string(peer_id);
                        reply_params["message"] = "Echo: " + text;
                        reply_params["random_id"] = std::to_string(vk::base::ClientBase::GetRandomId());
                        bot.SendRequest(vk::base::bot::BotBase::METHODS::SEND_MESSAGE, reply_params);

                        std::cout << "Обработано сообщение от " << peer_id << ": " << text << std::endl;
                        logMemory("AFTER_MSG");
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Ошибка обработки сообщения: " << e.what() << std::endl;
                }
            }
        } catch (const nlohmann::json::type_error& e) {
            // Игнорируем ошибки парсинга пустого пакета
            if (std::string(e.what()).find("cannot use at() with null") != std::string::npos) {
                continue;
            }
            std::cerr << "JSON ошибка: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Ошибка в цикле событий: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    memory_logger.join();
    curl_global_cleanup();
    std::cout << "Тест VK API завершён. Лог памяти сохранён в vk_test_mem.log" << std::endl;
    return 0;
}