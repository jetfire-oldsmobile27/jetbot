cmake_minimum_required(VERSION 3.25)
project(telegram_camera_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*|aarch64|AArch64")
    set(ON_SBC TRUE)
    message(STATUS "Platform: Single-board computer (ARM-based)")
else()
    set(ON_SBC FALSE)
    message(STATUS "Platform: x86/x86_64")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV LIBS = ${OpenCV_LIBS}")

add_executable(bot src/proto.cpp)

if(ON_SBC)
    find_package(Threads REQUIRED)
    find_package(OpenSSL REQUIRED)
    find_package(Boost COMPONENTS system REQUIRED)
    find_package(CURL REQUIRED)

    include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/include
        ${OPENSSL_INCLUDE_DIR}
        ${Boost_INCLUDE_DIR}
    )
    if(CURL_FOUND)
        include_directories(${CURL_INCLUDE_DIRS})
        target_compile_definitions(bot PRIVATE HAVE_CURL)
    endif()

    target_link_libraries(bot PRIVATE
        /usr/local/lib/libTgBot.a
        ${CMAKE_THREAD_LIBS_INIT}
        ${OPENSSL_LIBRARIES}
        ${Boost_LIBRARIES}
        ${CURL_LIBRARIES}
        ${OpenCV_LIBS}
    )

else()
    find_package(tgbot CONFIG REQUIRED)
    target_link_libraries(bot PRIVATE
        tgbot::tgbot
        ${OpenCV_LIBS}
    )
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()