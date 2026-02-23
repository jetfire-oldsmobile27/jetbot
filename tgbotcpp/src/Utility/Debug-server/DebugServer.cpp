#include "DebugServer.hpp"
#include "Utility/thirdparty/base64.hpp"
#include <opencv2/opencv.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <sstream>
#include <mutex>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <memory>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// from main.cpp
extern cv::Mat last_frame;
extern std::mutex frame_mutex;

namespace Utility::DebugServer {

struct ServerContext {
    net::io_context ioc;
    tcp::acceptor acceptor;
    std::thread server_thread;
    bool running = true;

    ServerContext(unsigned short port) 
        : acceptor(ioc, tcp::endpoint(tcp::v4(), port)) {}
};

std::string getLocalIPs() {
    std::string result;
    struct ifaddrs *ifaddr, *ifa;
    
    if (getifaddrs(&ifaddr) == -1) {
        return "Unable to get IP addresses";
    }

    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        
        if (ifa->ifa_addr->sa_family == AF_INET) {
            char host[NI_MAXHOST];
            int s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                               host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);
            if (s == 0) {
                if (strcmp(ifa->ifa_name, "lo") != 0) {
                    if (!result.empty()) result += ", ";
                    result += ifa->ifa_name;
                    result += ": ";
                    result += host;
                }
            }
        }
    }
    
    freeifaddrs(ifaddr);
    return result.empty() ? "127.0.0.1" : result;
}

void handle_request(tcp::socket& socket) {
    beast::flat_buffer buffer;
    http::request<http::string_body> req;
    beast::error_code ec;

    http::read(socket, buffer, req, ec);
    if (ec) {
        std::cerr << "DebugServer: Failed to read request: " << ec.message() << std::endl;
        return;
    }

    http::response<http::string_body> res;
    res.version(req.version());
    res.keep_alive(false);

    if (req.method() == http::verb::get && req.target() == "/") {
        cv::Mat frame_copy;
        
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!last_frame.empty()) {
                frame_copy = last_frame.clone();
            }
        }

        if (!frame_copy.empty()) {
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame_copy, buf);
            
            std::string base64_data = base64_encode(buf.data(), buf.size());
            
            std::string html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>Camera Stream</title>
    <meta http-equiv="refresh" content="1">
    <style>
        body { background: #111; color: #0f0; font-family: monospace; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        img { width: 100%; border: 2px solid #0f0; border-radius: 5px; }
        .info { margin: 10px 0; color: #0f0; }
        .footer { margin-top: 20px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Camera Stream</h1>
        <div class="info">Last frame (auto-refresh every 1s)</div>
        <img src="data:image/jpeg;base64,)" + base64_data + R"(" alt="Camera frame">
        <div class="footer">Frame size: )" + std::to_string(frame_copy.cols) + "x" + std::to_string(frame_copy.rows) + R"(</div>
    </div>
</body>
</html>)";

            res.result(http::status::ok);
            res.set(http::field::content_type, "text/html");
            res.body() = html;
        } else {
            std::string error_html = R"(
<!DOCTYPE html>
<html>
<head><title>Camera Stream</title></head>
<body>
    <h1>No frame available</h1>
    <p>Camera not initialized or no frames received yet.</p>
</body>
</html>)";
            
            res.result(http::status::ok);
            res.set(http::field::content_type, "text/html");
            res.body() = error_html;
        }
    } else if (req.method() == http::verb::get && req.target() == "/frame.jpg") {
        cv::Mat frame_copy;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!last_frame.empty()) {
                frame_copy = last_frame.clone();
            }
        }

        if (!frame_copy.empty()) {
            std::vector<uchar> buf;
            cv::imencode(".jpg", frame_copy, buf);
            
            res.result(http::status::ok);
            res.set(http::field::content_type, "image/jpeg");
            res.body() = std::string(buf.begin(), buf.end());
        } else {
            res.result(http::status::not_found);
            res.body() = "No frame available";
        }
    } else {
        res.result(http::status::not_found);
        res.set(http::field::content_type, "text/plain");
        res.body() = "404 Not Found";
    }

    res.prepare_payload();
    
    http::write(socket, res, ec);
    if (ec) {
        std::cerr << "DebugServer: Failed to send response: " << ec.message() << std::endl;
    }
}

void server_loop(ServerContext* ctx) {
    std::cout << "DebugServer: Starting on port " 
              << ctx->acceptor.local_endpoint().port() << std::endl;
    
    while (ctx->running) {
        try {
            tcp::socket socket(ctx->ioc);
            ctx->acceptor.accept(socket);
            auto handler = [sock = std::move(socket)]() mutable {
                handle_request(sock);
            };
            std::thread(std::move(handler)).detach();
            
        } catch (const std::exception& e) {
            if (ctx->running) {
                std::cerr << "DebugServer: Accept error: " << e.what() << std::endl;
            }
        }
    }
}

DebugServer::DebugServer(unsigned short port) 
    : port_(port), running_(false), io_context_(nullptr), acceptor_(nullptr), server_thread_(nullptr) {}

DebugServer::~DebugServer() {
    stop();
}

void DebugServer::start() {
    if (running_) return;
    
    try {
        auto ctx = std::make_unique<ServerContext>(port_);
        io_context_ = &ctx->ioc;
        acceptor_ = &ctx->acceptor;
        running_ = true;      
        std::string ips = getLocalIPs();
        std::cout << "\n-->Debug Server started!" << std::endl;
        std::cout << "   Connect at:" << std::endl;
        
        size_t start = 0;
        size_t end = ips.find(", ");
        while (end != std::string::npos) {
            std::string ip_entry = ips.substr(start, end - start);
            size_t colon = ip_entry.find(": ");
            if (colon != std::string::npos) {
                std::string iface = ip_entry.substr(0, colon);
                std::string ip = ip_entry.substr(colon + 2);
                std::cout << "   ➜ http://" << ip << ":" << port_ << "  (" << iface << ")" << std::endl;
            }
            start = end + 2;
            end = ips.find(", ", start);
        }
        size_t colon = ips.find(": ", start);
        if (colon != std::string::npos) {
            std::string iface = ips.substr(start, colon - start);
            std::string ip = ips.substr(colon + 2);
            std::cout << "   ➜ http://" << ip << ":" << port_ << "  (" << iface << ")" << std::endl;
        } else {
            std::cout << "   ➜ http://" << ips << ":" << port_ << std::endl;
        }
        std::cout << "   ➜ http://localhost:" << port_ << std::endl;
        std::cout << std::endl;
        
        ctx->server_thread = std::thread(server_loop, ctx.get());
        
        server_thread_ = &ctx->server_thread;
        static std::unique_ptr<ServerContext> global_ctx;
        global_ctx = std::move(ctx);
        
        io_context_ = &global_ctx->ioc;
        acceptor_ = &global_ctx->acceptor;
        
    } catch (const std::exception& e) {
        std::cerr << "DebugServer: Failed to start: " << e.what() << std::endl;
        running_ = false;
    }
}

void DebugServer::stop() {
    if (!running_) return;
    
    running_ = false;
    
    if (acceptor_) {
        auto* acceptor = static_cast<tcp::acceptor*>(acceptor_);
        try {
            acceptor->close();
        } catch (...) {}
    }
    
    if (io_context_) {
        auto* ioc = static_cast<net::io_context*>(io_context_);
        ioc->stop();
    }
    
    if (server_thread_) {
        auto* thread = static_cast<std::thread*>(server_thread_);
        if (thread->joinable()) {
            thread->join();
        }
    }
    
    io_context_ = nullptr;
    acceptor_ = nullptr;
    server_thread_ = nullptr;
}

} // namespace Utility::DebugServer