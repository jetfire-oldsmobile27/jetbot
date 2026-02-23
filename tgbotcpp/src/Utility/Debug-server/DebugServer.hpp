#pragma once

#include <string>

namespace Utility::DebugServer {

class DebugServer {
public:
    DebugServer(unsigned short port = 8080);
    ~DebugServer();

    void start();
    void stop();
    bool isRunning() const { return running_; }

private:
    unsigned short port_;
    bool running_;
    void* io_context_;     
    void* acceptor_;        
    void* server_thread_;
};

} // namespace Utility::DebugServer