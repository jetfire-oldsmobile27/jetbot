#include "OutAPI/OllamaServer.hpp"
#include "Utility/thirdparty/base64.hpp"
#include <boost/json/object.hpp>
#include <expected>
#include <opencv2/core/mat.hpp>
#include <boost/asio.hpp>
#include <boost/beast/http.hpp>


namespace OutAPI {
    OllamaServer& OllamaServer::set_model(const std::string& model_name) {
        model_name_ = model_name;
        return *this;
    }

    OllamaServer& OllamaServer::set_ip_adress(const std::string& ollama_ip) {
        ollama_ip_ = ollama_ip;
        return *this;
    }

    auto OllamaServer::send_request(const cv::Mat& frame, 
                       const std::string& prompt) -> std::expected<std::string, OllamaServerError> {
        
        if(model_name_.empty()) {
            return std::unexpected(OllamaServerError::NO_MODEL);
        } else if(ollama_ip_.empty()) {
            return std::unexpected(OllamaServerError::NO_IP_ADDDR);
        }

        /*{
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
        }*/
        boost::json::object payload;

    }
}

