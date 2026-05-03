#include <iostream>
#include <vkbot/BotBase.hpp>

int main() {
    const std::string token = []() -> std::string {
        const char* env = std::getenv("VK_ACCESS_TOKEN");
        return env ? env : "";
    }();
    const std::string group_id = []() -> std::string {
        const char* env = std::getenv("VK_GROUP_ID");
        return env ? env : "";
    }();

    if (token.empty() || group_id.empty()) {
        std::cerr << "VK_ACCESS_TOKEN and VK_GROUP_ID required\n";
        return 1;
    }

    vk::bot::BotBase bot(group_id);

    if (!bot.auth(token)) {
        std::cerr << "Auth failed\n";
        return 1;
    }

    while (true) {
        auto event = bot.wait_for_event();

        if (event.type == vk::bot::BotBase::Event::MessageNew) {
            const auto& obj = event.payload["object"]["message"];
            const std::string text = obj.value("text", "");
            const auto peer_id = obj.value("peer_id", 0);
            
            std::cout << "Message from " << peer_id << ": " << text << '\n';
            
            bot.send_request(vk::bot::BotBase::Method::SendMessage, {
                {"peer_id", std::to_string(peer_id)},
                {"message", "Echo: " + text},
                {"random_id", std::to_string(vk::base::ClientBase::random_id())}
            });
        }
    }
    
    return 0;
}