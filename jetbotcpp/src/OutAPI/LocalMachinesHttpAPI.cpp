#include "OutAPI/LocalMachinesHttpAPI.hpp"
#include "Globals.hpp"
#include "Logging.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include <chrono>
#include <format>
#include <string>

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using tcp       = net::ip::tcp;

namespace LocalMachinesHttpAPI {

namespace {

constexpr std::chrono::seconds REQUEST_TIMEOUT{10};

auto sendGetRequest(const std::string& target) -> std::expected<void, RequestError> {
  if (LOCAL_MACHINE_IP.empty() || LOCAL_MACHINE_PORT.empty()) {
    logMsg("LocalMachinesHttpAPI: LOCAL_MACHINE_IP/LOCAL_MACHINE_PORT не заданы");
    return std::unexpected(RequestError::NO_ADDRESS);
  }

  try {
    net::io_context ioc;

    // tcp_stream умеет ограничивать время всех последующих операций (connect/write/read)
    beast::tcp_stream stream(ioc);
    stream.expires_after(REQUEST_TIMEOUT);

    tcp::resolver resolver(ioc);
    const auto endpoints = resolver.resolve(LOCAL_MACHINE_IP, LOCAL_MACHINE_PORT);
    stream.connect(endpoints);

    http::request<http::empty_body> req{http::verb::get, target, 11};
    req.set(http::field::host, LOCAL_MACHINE_IP);
    req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);

    http::write(stream, req);

    beast::flat_buffer buffer;
    http::response<http::string_body> res;
    http::read(stream, buffer, res);

    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both, ec);

    if (res.result() != http::status::ok) {
      logMsg(std::format("LocalMachinesHttpAPI: {} -> HTTP {}",
                         target, res.result_int()));
      return std::unexpected(RequestError::BAD_RESPONSE);
    }

    logMsg(std::format("LocalMachinesHttpAPI: {} -> 200 OK", target));
    return {};
  } catch (const beast::system_error& e) {
    if (e.code() == beast::error::timeout) {
      logMsg(std::format("LocalMachinesHttpAPI: {} -> таймаут", target));
      return std::unexpected(RequestError::TIMEOUT);
    }
    logMsg(std::format("LocalMachinesHttpAPI: {} -> system_error: {}",
                       target, e.what()));
    return std::unexpected(RequestError::BAD_CONNECTION);
  } catch (const std::exception& e) {
    logMsg(std::format("LocalMachinesHttpAPI: {} -> exception: {}",
                       target, e.what()));
    return std::unexpected(RequestError::BAD_CONNECTION);
  }
}

} // namespace

auto pushPowerButton() -> std::expected<void, RequestError> {
  return sendGetRequest("/push-power-button");
}

auto turnOffPcImmediately() -> std::expected<void, RequestError> {
  return sendGetRequest("/turn-off-pc-immidiatelly");
}

} // namespace LocalMachinesHttpAPI