#include "OutAPI/VkBot.hpp"

#include <format>
#include <vkbot/BotBase.hpp>        
#include <vkbot/ClientBase.hpp>    

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Globals.hpp"
#include "Logging.hpp"
#include "ObserverLoop/VideoRecorder.hpp"
#include "OllamaServer.hpp"
#include "Utility/Settings.hpp"


static int64_t generate_random_id() {
    // Используем random_id() из библиотеки — он thread_local и безопасен
    return static_cast<int64_t>(vk::base::ClientBase::random_id());
}


using Bot = vk::bot::BotBase;
using Json = vk::base::JsonType;

static void sendVkMessage(Bot& bot, int64_t peer_id, const std::string& message) {
    Json params;
    params["peer_id"]   = std::to_string(peer_id);
    params["message"]   = message;
    params["random_id"] = std::to_string(generate_random_id());
    auto response = bot.send_request(Bot::Method::SendMessage, params);
    if (response.contains("error"))
        std::cerr << "VK sendMessage error: " << response.dump() << '\n';
}

static void sendVkPhoto(Bot& bot, int64_t peer_id, const std::string& file_path) {
    try {
        Json params;
        params["peer_id"] = std::to_string(peer_id);
        // Строковый вариант send_request — без изменений
        auto upload_server = bot.send_request("photos.getMessagesUploadServer", params);
        if (upload_server.contains("error")) {
            std::cerr << "getMessagesUploadServer error: " << upload_server.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка получения сервера для загрузки фото.");
            return;
        }
        if (!upload_server.contains("response") ||
            !upload_server["response"].contains("upload_url")) {
            std::cerr << "Invalid upload server response: " << upload_server.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Неверный ответ от сервера VK.");
            return;
        }
        std::string upload_url = upload_server["response"]["upload_url"].get<std::string>();

        Json json_response = bot.send_file_request(upload_url, file_path, "photo");

        if (!json_response.contains("photo") || !json_response.contains("server") ||
            !json_response.contains("hash")) {
            std::cerr << "Missing fields in upload response: " << json_response.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Неполный ответ от сервера загрузки.");
            return;
        }

        Json save_params;
        auto& photo_val = json_response["photo"];
        save_params["photo"] = photo_val.is_string()
            ? photo_val.get<std::string>()
            : std::to_string(photo_val.get<int64_t>());

        auto& server_val = json_response["server"];
        save_params["server"] = server_val.is_number()
            ? std::to_string(server_val.get<int>())
            : server_val.get<std::string>();

        save_params["hash"] = json_response["hash"].get<std::string>();

        auto saved = bot.send_request("photos.saveMessagesPhoto", save_params);
        if (saved.contains("error")) {
            std::cerr << "saveMessagesPhoto error: " << saved.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка сохранения фото в VK.");
            return;
        }
        if (!saved.contains("response")) {
            std::cerr << "No response in save result: " << saved.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка: пустой ответ от сервера.");
            return;
        }

        Json photo_info;
        if (saved["response"].is_array() && !saved["response"].empty())
            photo_info = saved["response"][0];
        else if (saved["response"].is_object())
            photo_info = saved["response"];
        else {
            std::cerr << "Unexpected response type: " << saved["response"].dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка формата ответа.");
            return;
        }

        if (!photo_info.contains("owner_id") || !photo_info.contains("id")) {
            std::cerr << "Missing owner_id or id in photo info: " << photo_info.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка получения данных фото.");
            return;
        }

        int64_t owner_id = photo_info["owner_id"].is_number()
            ? photo_info["owner_id"].get<int64_t>()
            : std::stoll(photo_info["owner_id"].get<std::string>());

        int64_t id = photo_info["id"].is_number()
            ? photo_info["id"].get<int64_t>()
            : std::stoll(photo_info["id"].get<std::string>());

        std::string attachment = "photo" + std::to_string(owner_id) + "_" + std::to_string(id);

        Json msg_params;
        msg_params["peer_id"]    = std::to_string(peer_id);
        msg_params["attachment"] = attachment;
        msg_params["random_id"]  = std::to_string(generate_random_id());

        auto msg_response = bot.send_request(Bot::Method::SendMessage, msg_params);
        if (msg_response.contains("error")) {
            std::cerr << "send photo message error: " << msg_response.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка отправки сообщения с фото.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in sendVkPhoto: " << e.what() << '\n';
        sendVkMessage(bot, peer_id, "❌ Внутренняя ошибка при отправке фото.");
    }
}

static void sendVkDocument(Bot& bot, int64_t peer_id,
                           const std::string& file_path,
                           const std::string& /*type_hint*/ = "") {
    try {
        Json params;
        params["peer_id"] = std::to_string(peer_id);
        params["type"] = "doc";
        auto upload_server = bot.send_request("docs.getMessagesUploadServer", params);
        if (upload_server.contains("error") ||
            !upload_server["response"].contains("upload_url")) {
            std::cerr << "getMessagesUploadServer error: " << upload_server.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка получения сервера для загрузки.");
            return;
        }
        std::string upload_url = upload_server["response"]["upload_url"];

        Json upload_response = bot.send_file_request(upload_url, file_path, "file");
        if (!upload_response.contains("file")) {
            std::cerr << "Upload failed, response: " << upload_response.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка загрузки файла на сервер VK.");
            return;
        }

        Json save_params;
        save_params["file"] = upload_response["file"].get<std::string>();
        auto saved = bot.send_request("docs.save", save_params);
        if (saved.contains("error") || !saved["response"].contains("doc")) {
            std::cerr << "docs.save error: " << saved.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка сохранения документа.");
            return;
        }

        auto& doc = saved["response"]["doc"];
        std::string attachment = "doc" + std::to_string(doc["owner_id"].get<int>())
                                 + "_" + std::to_string(doc["id"].get<int>());
        Json msg_params;
        msg_params["peer_id"]    = std::to_string(peer_id);
        msg_params["attachment"] = attachment;
        msg_params["random_id"]  = std::to_string(generate_random_id());
        auto msg_response = bot.send_request(Bot::Method::SendMessage, msg_params);
        if (msg_response.contains("error")) {
            std::cerr << "send document message error: " << msg_response.dump() << '\n';
            sendVkMessage(bot, peer_id, "❌ Ошибка отправки сообщения с документом.");
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in sendVkDocument: " << e.what() << '\n';
        sendVkMessage(bot, peer_id, "❌ Внутренняя ошибка при отправке документа.");
    }
}

static void sendVkMessageWithKeyboard(Bot& bot, int64_t peer_id,
                                      const std::string& message,
                                      const Json& keyboard) {
    Json params;
    params["peer_id"]   = std::to_string(peer_id);
    params["message"]   = message;
    params["random_id"] = std::to_string(generate_random_id());
    params["keyboard"]  = keyboard.dump();
    auto response = bot.send_request(Bot::Method::SendMessage, params);
    if (response.contains("error"))
        std::cerr << "VK sendMessage with keyboard error: " << response.dump() << '\n';
}

static Json createMainMenuKeyboard() {
    std::cout << "createMainMenuKeyboard()";
    Json keyboard;
    keyboard["one_time"] = false;
    keyboard["buttons"] = Json::array({
        Json::array({
            Json{{"action", Json{{"type","text"},{"payload","{\"command\":\"/photo\"}"},{"label","📸 Фото"}}},   {"color","primary"}},
            Json{{"action", Json{{"type","text"},{"payload","{\"command\":\"/last\"}"},{"label","🎥 Последнее видео"}}}, {"color","secondary"}}
        }),
        Json::array({
            Json{{"action", Json{{"type","text"},{"payload","{\"command\":\"/status\"}"},{"label","📊 Статус"}}},      {"color","secondary"}},
            Json{{"action", Json{{"type","text"},{"payload","{\"command\":\"/alert\"}"},{"label","🔔 Уведомления"}}},  {"color","secondary"}}
        })
    });
    return keyboard;
}


void vk_bot_thread(Utility::Settings& settings,
                   ObserverLoop::VideoRecorder& recorder,
                   vk::bot::BotBase& bot_vk)
{
    std::unordered_map<std::string, CommandHandler> commands;


    auto ollama_srv = OutAPI::OllamaServer()
                                                        .set_ip_adress(" 192.168.31.106")
                                                        .set_model("huihui_ai/qwen3-vl-abliterated:4b");

    commands["start"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        int64_t uid = peer_id;
        if (authorizedUserId.load() == 0) {
            authorizedUserId = uid;
            { std::lock_guard<std::mutex> lock(settings_mutex);
              settings.authorizedUserId = uid;
              settings.save(settings_path); }
            logMsg("🔓 VK Доступ выдан: ID " + std::to_string(uid));
            sendVkMessageWithKeyboard(bot, peer_id,
                "✅ Вы авторизованы!\nВаш ID: " + std::to_string(uid),
                createMainMenuKeyboard());
        } else if (uid == authorizedUserId.load()) {
            sendVkMessageWithKeyboard(bot, peer_id, "ℹ️ Вы уже авторизованы.",
                createMainMenuKeyboard());
        } else {
            logMsg("❌ VK Попытка /start от неразрешённого: ID " + std::to_string(uid));
            sendVkMessage(bot, peer_id, "⛔ Доступ имеет только первый обратившийся.");
        }
    };

    commands["photo"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        try {
            Json empty;
            bot.send_request(Bot::Method::MarkAsRead, empty);
        } catch(...) {
            logMsg("❌ Не удалось поставить статус \"Прочитано\" ");
        }
        cv::Mat frame_copy;
        { 
            std::lock_guard<std::mutex> lock(frame_mutex);
          if (last_raw_frame.empty()) {
              sendVkMessage(bot, peer_id, "⚠️ Нет данных с камеры.");
              return;
          }
          frame_copy = last_raw_frame.clone(); 
        }
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream fn;
        fn << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S") << ".jpg";
        std::filesystem::path tmp = std::filesystem::temp_directory_path() / fn.str();
        if (!cv::imwrite(tmp.string(), frame_copy)) {
            sendVkMessage(bot, peer_id, "❌ Ошибка сохранения кадра.");
            return;
        }
        try {
            auto response = bot.send_photo(peer_id, tmp.string());
            if (response.contains("error")) {
                std::cerr << "send_photo error: " << response.dump() << '\n';
                sendVkMessage(bot, peer_id, "❌ Ошибка отправки фото.");
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in sendVkPhoto: " << e.what() << '\n';
            sendVkMessage(bot, peer_id, "❌ Внутренняя ошибка при отправке фото.");
        }
        std::filesystem::remove(tmp);
    };

    commands["ask"] = [&ollama_srv](Bot& bot, int64_t peer_id, const std::string& args) {
        try {
            Json empty;
            bot.send_request(Bot::Method::MarkAsRead, empty);
        } catch(...) {
            logMsg("❌ Не удалось поставить статус \"Прочитано\" ");
        }
        cv::Mat frame_copy;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
          if (last_raw_frame.empty()) {
              sendVkMessage(bot, peer_id, "⚠️ Нет данных с камеры.");
              return;
          }
          frame_copy = last_raw_frame.clone(); 
        }
        auto response =  ollama_srv.send_request(frame_copy,  "Что ты видишь на этом изображении?Опиши подробно.");

        if(response.has_value()) {
            sendVkMessage(bot, peer_id, *response);
        } else if(response.error() == OutAPI::OllamaServerError::BAD_ANSWER){
            sendVkMessage(bot, peer_id, "Код возврата сервера отличен от ОК");
        }else if(response.error() == OutAPI::OllamaServerError::BAD_CONNECTION){
            sendVkMessage(bot, peer_id, "❌  Неусточивое соединение с сервисом распознавания");
        }else if(response.error() == OutAPI::OllamaServerError::NO_IP_ADDDR){
            sendVkMessage(bot, peer_id, "❌  IP адрес сервера распознавания задан неверно");
        } else if(response.error() == OutAPI::OllamaServerError::NO_MODEL){
            sendVkMessage(bot, peer_id, "❌  Указаннная модель распознавания не найдена на сервере");
        } else {
            sendVkMessage(bot, peer_id, "❌  Неизвестная ошибка при запросе на сервер");
        };
    };

    commands["alert"] = [&](Bot& bot, int64_t peer_id, const std::string& args) {
        if (args == "on") {
            alert_enabled = true;
            { std::lock_guard<std::mutex> lock(settings_mutex);
              settings.alert_enabled = true; settings.save(settings_path); }
            sendVkMessage(bot, peer_id, "🔔 Уведомления включены");
        } else if (args == "off") {
            alert_enabled = false;
            { std::lock_guard<std::mutex> lock(settings_mutex);
              settings.alert_enabled = false; settings.save(settings_path); }
            sendVkMessage(bot, peer_id, "🔕 Уведомления выключены");
        } else {
            sendVkMessage(bot, peer_id, "Использование: /alert on или /alert off");
        }
    };

    commands["last"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        if (!flag_recording.load()) {
            sendVkMessage(bot, peer_id, "⚠️ Запись видео отключена (флаг --no-recording).");
            return;
        }
        auto now   = std::chrono::system_clock::now();
        time_t now_c = std::chrono::system_clock::to_time_t(now);
        tm now_tm = *std::localtime(&now_c);
        char buf[20]; strftime(buf, sizeof(buf), "%Y%m%d", &now_tm);
        std::string today = buf;
        std::string video_path = recorder.getVideoPathForDate(today);

        if (!std::filesystem::exists(video_path) ||
            std::filesystem::file_size(video_path) <= 48) {
            for (int i = 1; i <= 7; ++i) {
                auto date   = now - std::chrono::hours(24 * i);
                time_t dc   = std::chrono::system_clock::to_time_t(date);
                tm dtm      = *std::localtime(&dc);
                char db[20]; strftime(db, sizeof(db), "%Y%m%d", &dtm);
                std::string path = recorder.getVideoPathForDate(db);
                if (std::filesystem::exists(path) &&
                    std::filesystem::file_size(path) > 48) {
                    video_path = path; break;
                }
            }
        }

        if (video_path.empty() || !std::filesystem::exists(video_path) ||
            std::filesystem::file_size(video_path) <= 48) {
            sendVkMessage(bot, peer_id, "Нет записанных видео");
        } else {
            try {
                if (recorder.isRecording()) {
                    recorder.stopRecording();
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    recorder.startRecording();
                }
                sendVkDocument(bot, peer_id, video_path, "video/mp4");
            } catch (const std::exception& e) {
                std::cerr << "Ошибка отправки видео VK: " << e.what() << '\n';
                sendVkMessage(bot, peer_id, "❌ Ошибка отправки видео: " + std::string(e.what()));
            }
        }
    };

    commands["unstopable"] = [&](Bot& bot, int64_t peer_id, const std::string& args) {
        if (args == "on") {
            unstopable_mode = true;
            { std::lock_guard<std::mutex> lock(settings_mutex);
              settings.unstopable_mode = true; settings.save(settings_path); }
            sendVkMessage(bot, peer_id, "🔄 Режим непрерывной записи ВКЛЮЧЕН");
        } else if (args == "off") {
            unstopable_mode = false;
            { std::lock_guard<std::mutex> lock(settings_mutex);
              settings.unstopable_mode = false; settings.save(settings_path); }
            sendVkMessage(bot, peer_id, "🔄 Режим непрерывной записи ВЫКЛЮЧЕН");
        } else {
            sendVkMessage(bot, peer_id, "Использование: /unstopable on или /unstopable off");
        }
    };

    commands["status"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        std::string status = "📊 Статус системы:\n";
        status += "Запись: "    + std::string(recorder.isRecording() ? "✅ активна" : "❌ неактивна") + "\n";
        status += "Режим: "     + std::string(unstopable_mode.load() ? "🔄 непрерывная запись" : "🎯 запись при обнаружении") + "\n";
        status += "Детекция: "  + std::string(flag_detection.load() ? "✅ активна" : "❌ отключена") + "\n";
        status += "Лица: "      + std::string(flag_face.load()      ? "✅ активна" : "❌ отключена") + "\n";
        status += "Эффекты: "   + std::string(flag_effects.load()   ? "✅ включены": "❌ отключены") + "\n";
        status += "Звук: "      + std::string(flag_sound.load()     ? "✅ включён" : "❌ отключён")  + "\n";
        status += "Очистка: "   + std::string(flag_cleanup.load()   ? "✅ активна" : "❌ отключена") + "\n";
        status += "Люди в кадре: " + std::string(detection_active.load() ? "⚠️ обнаружены" : "✅ не обнаружены") + "\n";
        status += "Уведомления: "  + std::string(alert_enabled.load()    ? "🔔 включены" : "🔕 выключены") + "\n";
        status += "Авторизован: ID " + std::to_string(authorizedUserId.load());
        sendVkMessage(bot, peer_id, status);
    };

    commands["cpuinfo"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        auto info = readFile("/proc/cpuinfo");
        if (info.empty()) {
            sendVkMessage(bot, peer_id, "❌ Не удалось прочитать /proc/cpuinfo");
        } else if (info.size() < 3500) {
            sendVkMessage(bot, peer_id, info);
        } else {
            std::string tmp = "/tmp/cpuinfo.txt";
            std::ofstream(tmp) << info;
            sendVkDocument(bot, peer_id, tmp, "text/plain");
            std::filesystem::remove(tmp);
        }
    };

    commands["temp"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        try {
            Json empty;
            bot.send_request(Bot::Method::MarkAsRead, empty);
        } catch(...) {
            logMsg("❌ Не удалось поставить статус \"Прочитано\" ");
        }
        std::ostringstream report;
        bool found = false;
        const std::filesystem::path thermalDir{"/sys/class/thermal"};
        if (std::filesystem::exists(thermalDir) && std::filesystem::is_directory(thermalDir)) {
            for (auto& entry : std::filesystem::directory_iterator(thermalDir)) {
                auto name = entry.path().filename().string();
                if (name.rfind("thermal_zone", 0) != 0) continue;
                std::filesystem::path typeFile = entry.path() / "type";
                std::filesystem::path tempFile = entry.path() / "temp";
                if (!std::filesystem::exists(typeFile) || !std::filesystem::exists(tempFile)) continue;
                std::string typeStr, tempStr;
                std::ifstream(typeFile) >> typeStr;
                std::ifstream(tempFile) >> tempStr;
                try {
                    double tc = std::stod(tempStr) / 1000.0;
                    report << "[" << name << "] " << typeStr << ": "
                           << std::fixed << std::setprecision(2) << tc << " °C\n";
                    found = true;
                } catch (...) {
                    report << "[" << name << "] " << typeStr << ": invalid (" << tempStr << ")\n";
                    found = true;
                }
            }
        }
        const std::filesystem::path hwmonDir{"/sys/class/hwmon"};
        if (std::filesystem::exists(hwmonDir) && std::filesystem::is_directory(hwmonDir)) {
            for (auto& entry : std::filesystem::directory_iterator(hwmonDir)) {
                std::string chip;
                std::filesystem::path nameFile = entry.path() / "name";
                if (std::filesystem::exists(nameFile)) std::ifstream(nameFile) >> chip;
                else chip = entry.path().filename().string();
                for (auto& f : std::filesystem::directory_iterator(entry.path())) {
                    auto fname = f.path().filename().string();
                    if (fname.rfind("temp", 0) != 0 || fname.find("_input") == std::string::npos) continue;
                    std::string idx = fname.substr(4, fname.find("_input") - 4);
                    std::string tempStr, label;
                    std::ifstream(f.path()) >> tempStr;
                    std::filesystem::path labelFile = entry.path() / ("temp" + idx + "_label");
                    if (std::filesystem::exists(labelFile)) std::getline(std::ifstream(labelFile), label);
                    else label = "temp" + idx;
                    try {
                        double tc = std::stod(tempStr) / 1000.0;
                        report << "[" << entry.path().filename() << "] " << chip << " " << label
                               << ": " << std::fixed << std::setprecision(2) << tc << " °C\n";
                        found = true;
                    } catch (...) {
                        report << "[" << entry.path().filename() << "] " << chip << " " << label
                               << ": invalid (" << tempStr << ")\n";
                        found = true;
                    }
                }
            }
        }
        if (!found)
            sendVkMessage(bot, peer_id, "❌ Не найден ни один температурный датчик.");
        else
            sendVkMessage(bot, peer_id, report.str());
    };

    commands["logs"] = [&](Bot& bot, int64_t peer_id, const std::string&) {
        auto logPath = getLogFilePath();
        if (!std::filesystem::exists(logPath))
            sendVkMessage(bot, peer_id, "📄 Лог за сегодня не найден");
        else
            sendVkDocument(bot, peer_id, logPath.string(), "text/plain");
    };

    commands["notify"] = commands["alert"];

    // ---------- Основной цикл ----------
    while (running) {
        try {

            while (running) {
                Bot::EventData event{};
 
                try {
                    event = bot_vk.wait_for_event();
                } catch (const nlohmann::json::type_error& e) {
                    if (std::string(e.what()).find("cannot use at() with null")
                            != std::string::npos) {
                        std::cerr << "VK: json type error, continuing...\n";
                        continue;
                    }
                    throw;
                } catch (const vk::ex::NetworkException& e) {
                    std::cerr << "VK network error: " << e.what() << ", retry in 2s\n";
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    continue;
                }
 
                
                if (event.type == Bot::Event::Unknown) {
                    logMsg(std::format("❌ Бот получил неизвестный ивент, payload: {}", event.payload.dump()));
                    continue;
                }

                switch (event.type) {
                    case Bot::Event::MessageNew: {
                        std::cout << "Новое сообщение VK!\n";

                        if (!event.payload.contains("object") || event.payload["object"].is_null())
                            break;
                        auto object = event.payload["object"];
                        if (!object.contains("message") || object["message"].is_null())
                            break;
                        auto message_data = object["message"];
                        if (!message_data.contains("peer_id")) {
                            std::cerr << "peer_id not found, skipping\n";
                            break;
                        }

                        int64_t peer_id = 0;
                        if (message_data["peer_id"].is_number_integer())
                            peer_id = message_data["peer_id"].get<int64_t>();
                        else { std::cerr << "peer_id not integer, skipping\n"; break; }

                        std::string text;
                        if (message_data.contains("text") && !message_data["text"].is_null()) {
                            auto& tv = message_data["text"];
                            text = tv.is_string() ? tv.get<std::string>() : tv.dump();
                        } else {
                            text = "⚠️ [нет текста]";
                        }

                        std::string command_to_execute, command_args;

                        if (message_data.contains("payload") && !message_data["payload"].is_null()) {
                            try {
                                Json payload;
                                if (message_data["payload"].is_string())
                                    payload = Json::parse(message_data["payload"].get<std::string>());
                                else
                                    payload = message_data["payload"];
                                if (payload.contains("command") && payload["command"].is_string())
                                    command_to_execute = payload["command"].get<std::string>();
                            } catch (const std::exception& e) {
                                std::cerr << "Ошибка обработки payload: " << e.what() << '\n';
                            }
                        } else if (!text.empty() && text[0] == '/') {
                            std::istringstream iss(text);
                            iss >> command_to_execute;
                            std::getline(iss >> std::ws, command_args);
                            // if (!command_to_execute.empty() && command_to_execute[0] == '/')
                            //     {command_to_execute = command_to_execute.substr(1);}
                        }

                        if (!command_to_execute.empty()) {
                            auto it = commands.find(command_to_execute);
                            if (it != commands.end()) {
                                if (peer_id != authorizedUserId.load() &&
                                    command_to_execute != "start")
                                    sendVkMessage(bot_vk, peer_id, "⛔ У вас нет доступа.");
                                else
                                    it->second(bot_vk, peer_id, command_args);
                            } else {
                                sendVkMessage(bot_vk, peer_id, "❓ Неизвестная команда");
                            }
                        } else {
                            sendVkMessage(bot_vk, peer_id, "Вы написали: " + text + "\nЯ вас приветствую!");
                        }
                        break;
                    }

                    case Bot::Event::GroupJoin: {
                        std::cout << "Новый участник вступил!\n";
                        if (!event.payload.contains("user_id") ||
                            !event.payload["user_id"].is_number_integer()) {
                            std::cerr << "user_id not found or not integer\n";
                            break;
                        }
                        int64_t user_id = event.payload["user_id"].get<int64_t>();

                        Json params;
                        params["peer_id"]   = std::to_string(user_id);
                        params["message"]   = "Привет, новый участник! Добро пожаловать в нашу группу!";
                        params["random_id"] = std::to_string(generate_random_id());

                        auto response = bot_vk.send_request(Bot::Method::SendMessage, params);
                        if (response.contains("error"))
                            std::cerr << "Ошибка отправки VK: " << response.dump() << '\n';
                        break;
                    }

                    default:
                        break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "\nVK thread error: " << e.what()
                      << " — reconnecting in 3s...\n";
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }
    std::cout << "VK thread finished\n";
}