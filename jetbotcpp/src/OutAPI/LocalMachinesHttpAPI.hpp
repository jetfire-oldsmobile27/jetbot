#pragma once
#include <expected>

namespace LocalMachinesHttpAPI {

enum class RequestError {
  SUCCESS = 0,
  NO_ADDRESS,    
  BAD_CONNECTION,  
  BAD_RESPONSE,    
  TIMEOUT,         
};

// GET /push-power-button
auto pushPowerButton() -> std::expected<void, RequestError>;

// GET /turn-off-pc-immidiatelly
auto turnOffPcImmediately() -> std::expected<void, RequestError>;

} // namespace LocalMachinesHttpAPI