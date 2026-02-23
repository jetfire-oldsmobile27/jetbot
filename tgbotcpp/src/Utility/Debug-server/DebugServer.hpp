#pragma once

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <thread>

namespace Utility::DebugServer {

class DebugServer : public std::enable_shared_from_this<DebugServer> {
public:
    DebugServer(unsigned short port = 8080);
    ~DebugServer();

    void start();
    void stop();

private:
    void do_accept();
    void on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket);
    
    unsigned short port_;
    boost::asio::io_context ioc_;
    boost::asio::ip::tcp::acceptor acceptor_;
    std::optional<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>> work_guard_;
    std::jthread server_thread_; 
};

} // namespace Utility::DebugServer