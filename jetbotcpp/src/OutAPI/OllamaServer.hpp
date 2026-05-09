#pragma once
#include <expected>
#include <string>

namespace cv {
    class Mat;
} // fwd declaration



namespace OutAPI {
enum class OllamaServerError {
    SUCCESS = 0,
    BAD_CONNECTION,
    NO_MODEL,
    NO_IP_ADDDR,
    BAD_ANSWER
};


class OllamaServer {
    public:
    OllamaServer();
    ~OllamaServer();
    OllamaServer& set_model(const std::string& model_name);
    OllamaServer& set_ip_adress(const std::string& ollama_ip);
    private:
    auto send_request(const cv::Mat& frame, 
                       const std::string& prompt) -> std::expected<std::string, OllamaServerError>;

    std::string model_name_;
    std::string ollama_ip_;

};

}