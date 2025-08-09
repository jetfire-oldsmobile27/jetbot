cmake_minimum_required(VERSION 3.25)
project(telegram_camera_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(tgbot CONFIG REQUIRED)
message(STATUS "CV LIBS=${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bot src/proto.cpp)

target_link_libraries(bot
    PRIVATE
        tgbot::tgbot
        ${OpenCV_LIBS}
)
