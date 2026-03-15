#pragma once

#include <functional>
#include <unordered_map>
#include <cstdint>
#include <string>

// Forward declarations
namespace Utility { class Settings; }
namespace ObserverLoop { class VideoRecorder; }
namespace vk::base::bot { class BotBase; }

using CommandHandler = std::function<void(vk::base::bot::BotBase&, int64_t, const std::string&)>;

void vk_bot_thread(Utility::Settings& settings, ObserverLoop::VideoRecorder& recorder);