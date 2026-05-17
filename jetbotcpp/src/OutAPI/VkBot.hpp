#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

// Forward declarations
namespace Utility      { class Settings;      }
namespace ObserverLoop { class VideoRecorder; }
namespace vk::bot      { class BotBase;       } 

using CommandHandler =
    std::function<void(vk::bot::BotBase&, int64_t, const std::string&)>;

void vk_bot_thread(Utility::Settings& settings,
                   ObserverLoop::VideoRecorder& recorder,
                    vk::bot::BotBase& bot_vk);