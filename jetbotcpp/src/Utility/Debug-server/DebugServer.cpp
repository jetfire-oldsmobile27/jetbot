#include "DebugServer.hpp"
#include "Utility/thirdparty/base64.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

extern cv::Mat last_frame;
extern cv::Mat last_raw_frame;
extern cv::Mat last_recognition_frame;
extern std::mutex frame_mutex;

namespace Utility::DebugServer {

static void handle_session(tcp::socket socket) {
    try {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        beast::error_code ec;

        http::read(socket, buffer, req, ec);
        if (ec) return;

        http::response<http::string_body> res;
        res.version(req.version());
        res.keep_alive(false);

        // Логика обработки разных endpoint'ов
        if (req.method() == http::verb::get) {
            cv::Mat frame;
            std::string target = req.target();
            
            if (target == "/get_image") {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (!last_raw_frame.empty()) frame = last_raw_frame.clone();
            } else if (target == "/get_recognition") {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (!last_recognition_frame.empty()) frame = last_recognition_frame.clone();
            } else {
                std::lock_guard<std::mutex> lock(frame_mutex);
                if (!last_frame.empty()) frame = last_frame.clone();
            }

            if (!frame.empty()) {
                std::vector<uchar> buf;
                cv::imencode(".jpg", frame, buf);
                
                if (target == "/" || target == "/index.html") {
                    std::string b64 = base64_encode(buf.data(), buf.size());
                    res.body() = "<html><body><img src='data:image/jpeg;base64," + b64 + "'/></body></html>";
                    res.set(http::field::content_type, "text/html");
                } else {
                    res.body() = std::string(buf.begin(), buf.end());
                    res.set(http::field::content_type, "image/jpeg");
                }
                res.result(http::status::ok);
            } else {
                res.result(http::status::not_found);
                res.body() = "No frame";
            }
        } else {
            res.result(http::status::method_not_allowed);
        }

        res.prepare_payload();
        http::write(socket, res, ec);
        
    } catch (...) {
        // Игнорируем любые исключения в сессии
    }
}

DebugServer::DebugServer(unsigned short port) 
    : port_(port)
    , acceptor_(ioc_, tcp::endpoint(tcp::v4(), port)) {
    acceptor_.set_option(net::socket_base::reuse_address(true));
}

DebugServer::~DebugServer() {
    stop();
}

void DebugServer::start() {
    if (server_thread_.joinable()) return;
    
    work_guard_.emplace(ioc_.get_executor());
    
    server_thread_ = std::jthread([this](std::stop_token st) {
        // Выводим информацию о запуске
        struct ifaddrs *ifaddr;
        if (getifaddrs(&ifaddr) == 0) {
            std::cout << "\n--> Debug Server started on port " << port_ << "\n";
            for (auto* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
                if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
                if (strcmp(ifa->ifa_name, "lo") == 0) continue;
                char host[256];
                getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, sizeof(host), nullptr, 0, NI_NUMERICHOST);
                std::cout << "   ➜ http://" << host << ":" << port_ << "  (" << ifa->ifa_name << ")\n";
            }
            std::cout << "   ➜ http://localhost:" << port_ << "\n\n";
            freeifaddrs(ifaddr);
        }
        
        // Запускаем асинхронный accept
        do_accept();
        
        // Запускаем io_context - он будет работать пока есть работа
        ioc_.run();
    });
}

void DebugServer::stop() {
    work_guard_.reset();
    ioc_.stop();
    
    // jthread автоматически join() в деструкторе
    if (server_thread_.joinable()) {
        server_thread_.request_stop();
    }
}

void DebugServer::do_accept() {
    acceptor_.async_accept(
        [self = shared_from_this()](beast::error_code ec, tcp::socket socket) {
            if (!ec) {
                // Запускаем сессию в отдельном потоке через std::async
                std::thread([sock = std::move(socket)]() mutable {
                    handle_session(std::move(sock));
                }).detach();
            }
            
            if (self->ioc_.stopped()) return;
            self->do_accept();
        });
}

} // namespace Utility::DebugServer