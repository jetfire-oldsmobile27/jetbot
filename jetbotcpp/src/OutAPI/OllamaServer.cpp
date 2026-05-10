#include "OutAPI/OllamaServer.hpp"
#include "Utility/thirdparty/base64.hpp"
#include <boost/beast/http.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/beast/version.hpp>
#include <boost/json/object.hpp>
#include <boost/json/serialize.hpp>
#include <expected>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <boost/beast/core.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>



 namespace json = boost::json;
 namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;

namespace OutAPI {

    OllamaServer::OllamaServer() {};

    OllamaServer::~OllamaServer() {};

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

        std::vector<uchar> buffer;
        cv::imencode(".jpg",frame, buffer);
        

        json::object payload;
        payload["model"] = model_name_;
        payload[prompt] = prompt;
        payload["images"] = {
            base64_encode(reinterpret_cast<unsigned char*>(buffer.data()), buffer.size())
        };
        payload["stream"] = false;

        asio::io_context io_ctx;
        asio::ip::tcp::resolver resolver(io_ctx);
        asio::ip::tcp::socket socket(io_ctx);
        std::string srv_api{"192.168.31.106"};
        asio::connect(socket, resolver.resolve(srv_api, "11434"));
        http::request<http::string_body> ollama_ask(http::verb::post, "/api/generate", 11);
        ollama_ask.set(http::field::host, srv_api);
        ollama_ask.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        ollama_ask.set(http::field::content_type, "application/json");
        ollama_ask.body() = json::serialize(payload);
        ollama_ask.content_length(ollama_ask.body().size());

        http::write(socket, ollama_ask);
        std::string response;
        {
            boost::beast::flat_buffer buffer;
            http::response<http::dynamic_body> res;
            http::read(socket, buffer, res);
            if (res.result() != http::status::ok) {
                return std::unexpected(OllamaServerError::BAD_ANSWER);
            }
            response = boost::beast::buffers_to_string(res.body().data());
        }

        auto response_json = json::parse(response);
        auto response_json_value = response_json.at("response").as_string().c_str();
        std::cout << "parsed str from json ollama response " << response_json_value << std::endl;
        
        socket.shutdown(asio::ip::tcp::socket::shutdown_both);
        std::cout << "ollama(full json): \n " << response << '\n' << std::endl;
        return response_json_value;
    }
}

