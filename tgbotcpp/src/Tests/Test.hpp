#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utility/thirdparty/base64.hpp"
#include <chrono>
#include <indicators/indeterminate_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>
#include <thread>

class ImageEncoder {
public:
  static std::string matToBase64(const cv::Mat &frame) {
    std::vector<uchar> buffer;
    cv::imencode(".jpg", frame, buffer);

    return base64_encode(buffer.data(), buffer.size(), false);
  }
};

class OllamaClient {
private:
  std::string host_;
  std::string port_;

public:
  OllamaClient(const std::string &host = "localhost",
               const std::string &port = "11434")
      : host_(host), port_(port) {}

  std::string sendRequest(const std::string &prompt,
                          const std::string &image_base64) {
    try {
      boost::asio::io_context ioc;
      boost::asio::ip::tcp::resolver resolver(ioc);
      boost::beast::tcp_stream stream(ioc);

      auto const results = resolver.resolve(host_, port_);
      stream.connect(results);

      std::string request_body = "{\n"
                                 "  \"model\": \"llava:7b\",\n"
                                 "  \"prompt\": \"" +
                                 escapeJsonString(prompt) +
                                 "\",\n"
                                 "  \"images\": [\"" +
                                 image_base64 +
                                 "\"],\n"
                                 "  \"stream\": false\n"
                                 "}";

      boost::beast::http::request<boost::beast::http::string_body> req{
          boost::beast::http::verb::post, "/api/generate", 11};
      req.set(boost::beast::http::field::host, host_);
      req.set(boost::beast::http::field::user_agent,
              BOOST_BEAST_VERSION_STRING);
      req.set(boost::beast::http::field::content_type, "application/json");
      req.body() = request_body;
      req.prepare_payload();

      boost::beast::http::write(stream, req);

      boost::beast::flat_buffer buffer;
      boost::beast::http::response<boost::beast::http::string_body> res;
      boost::beast::http::read(stream, buffer, res);

      boost::beast::error_code ec;
      stream.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);

      if (res.result() == boost::beast::http::status::ok) {
        std::istringstream response_stream(res.body());
        boost::property_tree::ptree json_response;
        boost::property_tree::read_json(response_stream, json_response);

        if (auto response_str =
                json_response.get_optional<std::string>("response")) {
          return *response_str;
        }
      } else {
        std::cerr << "HTTP Error: " << res.result() << std::endl;
        std::cerr << "Response: " << res.body() << std::endl;
      }

      return "";

    } catch (const std::exception &e) {
      std::cerr << "Error in sendRequest: " << e.what() << std::endl;
      return "";
    }
  }

private:
  std::string escapeJsonString(const std::string &input) {
    std::string output;
    for (char c : input) {
      switch (c) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output += c;
        break;
      }
    }
    return output;
  }
};

class CommandDispatcher {
private:
  std::unordered_map<int, std::function<void()>> commands_;

public:
  CommandDispatcher() { setupCommands(); }

  void setupCommands() {
    commands_[59] = []() { std::cout << "Executing weighing..." << std::endl; };

    commands_[121] = []() { std::cout << "Raising table..." << std::endl; };

    commands_[122] = []() { std::cout << "Lowering table..." << std::endl; };

    commands_[11] = []() {
      std::cout << "Recalculating checksum..." << std::endl;
    };

    for (int i = 20; i <= 31; i++) {
      commands_[i] = []() { std::cout << "No operation..." << std::endl; };
    }
  }

  void executeCommand(int command_id) {
    auto it = commands_.find(command_id);
    if (it != commands_.end()) {
      it->second();
    } else {
      std::cout << "Unknown command: " << command_id << std::endl;
    }
  }

  void parseAndExecute(const std::string &response) {
    std::vector<std::string> lines;
    boost::split(lines, response, boost::is_any_of("\n"));

    for (const auto &line : lines) {
      size_t pos = line.find(';');
      if (pos != std::string::npos) {
        std::string cmd_str = line.substr(0, pos);
        boost::trim(cmd_str);

        try {
          int command_id = std::stoi(cmd_str);
          executeCommand(command_id);
        } catch (const std::exception &e) {
          std::cerr << "Error parsing command: " << e.what() << std::endl;
        }
      }
    }
  }
};

class AIActionAgent {
private:
  OllamaClient ollama_client_;
  CommandDispatcher dispatcher_;
  std::string prompt_template_;

public:
  AIActionAgent() { setupPrompt(); }

  void setupPrompt() {
    prompt_template_ =
        "Выполни одно из действий исходя из изображения на экране. "
        "Дополнительно в конце напиши что происходит в кадре. \"<номер "
        "команды>;\". "
        "Список доступных функций:\n"
        "Выполнить <ЗАРЕЗЕРВИРОВАНО> - 59;\n"
        "Поднять стол - 121;\n"
        "Опустить стол - 122;\n"
        "Ничего не делать (диапазон от 20 до 31);\n"
        "Пересчитать контрольную сумму - 11;";
  }

  void makeSpecific(const int &camera_index) {
    std::cout << "============SPECIFIC=========" << std::endl;
    try {
      cv::VideoCapture cap;
      cap.open(camera_index, cv::CAP_V4L);
      if (!cap.isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        throw std::logic_error("Could not open camera");
      }

      cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
      cap.set(cv::CAP_PROP_FPS, 25);

      cv::Mat frame;
      if (!cap.read(frame)) {
        std::cerr << "Could not read frame from camera" << std::endl;
        throw std::logic_error("Could not read frame from camera");
      }

      cv::resize(frame, frame, cv::Size(640, 480));

      showImageAndWait(frame);

      std::string image_base64 = ImageEncoder::matToBase64(frame);
      std::cout << "Image encoded, size: " << image_base64.length() << " chars"
                << std::endl;

      std::cout << "Sending request to Ollama..." << std::endl;
      std::string response =
          ollama_client_.sendRequest(prompt_template_, image_base64);

      if (!response.empty()) {
        std::cout << "AI Response: " << response << std::endl;
        dispatcher_.parseAndExecute(response);
      } else {
        std::cerr << "Empty response from AI" << std::endl;
        throw std::logic_error("Empty response from AI");
      }

      std::cout << "============SPECIFIC END=========" << std::endl;

    } catch (const std::exception &e) {
      std::cerr << "Error in makeSpecific: " << e.what() << std::endl;
    }
  }

private:
  void showImageAndWait(const cv::Mat &frame) {
    const std::string window_name = "Camera Preview";

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, frame);
    cv::resizeWindow(window_name, 640, 480);

    std::cout
        << "Image displayed. Press any key to continue or close the window..."
        << std::endl;

    bool window_open = true;
    while (window_open) {
      int key = cv::waitKey(100);

      try {
        if (cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1) {
          window_open = false;
        }
      } catch (const cv::Exception &) {
        window_open = false;
      }

      if (key >= 0) {
        window_open = false;
      }
    }

    try {
      cv::destroyWindow(window_name);
    } catch (const cv::Exception &) {
      // Window already closed
    }

    std::cout << "Window closed, continuing..." << std::endl;
  }
};

namespace Tests {

bool TestImageAIagent(const int &camera_index) {
  try {
    AIActionAgent agent;

    std::cout << "=== Test 1: Basic functionality ===" << std::endl;
    agent.makeSpecific(camera_index);

    std::cout << "=== Test 2: Command parsing ===" << std::endl;
    CommandDispatcher test_dispatcher;
    test_dispatcher.parseAndExecute("59;\n121;");

    std::cout << "=== Test 3: Image encoding ===" << std::endl;
    cv::Mat test_frame(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    std::string encoded = ImageEncoder::matToBase64(test_frame);
    std::cout << "Image encoded to base64, length: " << encoded.length()
              << std::endl;

    return true;

  } catch (const std::exception &e) {
    return false;
  }
};

bool TestThreadPool() {
  try {
    boost::asio::thread_pool pool(4); // 4 потока, но они НЕ запущены!

    auto executor = pool.get_executor();

    boost::asio::post(executor, []() {
      std::cout << "Task 1 executed by thread " << std::this_thread::get_id()
                << std::endl;
    });

    boost::asio::post(executor, []() {
      std::cout << "Task 2 executed by thread " << std::this_thread::get_id()
                << std::endl;
    });

    pool.join();
    return true;
  } catch (...) {
    return false;
  }
};

bool TestOpenCVCapabilities() {
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Cannot open camera" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    cap >> frame;
    
    cv::VideoWriter writer;
    std::vector<std::pair<std::string, int>> codecs = {
        {"MJPG", cv::VideoWriter::fourcc('M','J','P','G')},
        {"XVID", cv::VideoWriter::fourcc('X','V','I','D')},
        {"I420", cv::VideoWriter::fourcc('I','4','2','0')},
        {"H264", cv::VideoWriter::fourcc('H','2','6','4')}
    };
    
    for (auto& [name, codec] : codecs) {
        writer.open("test_" + name + ".avi", codec, 25.0, frame.size());
        std::cout << name << ": " << (writer.isOpened() ? "OK" : "FAIL") << std::endl;
        writer.release();
    }
    return true;
}

void TestProgressBar() {
  indicators::IndeterminateProgressBar bar{
      indicators::option::BarWidth{40},
      indicators::option::Start{"["},
      indicators::option::Fill{"·"},
      indicators::option::Lead{"/\\"},
      indicators::option::End{"]"},
      indicators::option::PostfixText{"Checking for Updates"},
      indicators::option::ForegroundColor{indicators::Color::magenta},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::italic}}
  };

  indicators::show_console_cursor(false);

  auto job = [&bar]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));
    bar.mark_as_completed();
    std::cout << termcolor::bold << termcolor::green 
        << "Indicators test accepted\n" << termcolor::reset;
  };
  std::thread job_completion_thread(job);

  // Update bar state
  while (!bar.is_completed()) {
    bar.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  job_completion_thread.join();
  
  indicators::show_console_cursor(true);  

}

bool makeTests(const int &camera_index) {
  //bool ai_res_test = TestImageAIagent(camera_index);
  bool pool_res_test = TestThreadPool();
  TestOpenCVCapabilities();
  TestProgressBar();

  if ( //! ai_res_test ||
      !pool_res_test) {
    std::cout << "Tests failed!" << std::endl;
    return false;
  }
  return true;
};

} // namespace Tests