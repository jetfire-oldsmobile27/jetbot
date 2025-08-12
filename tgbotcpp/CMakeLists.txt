cmake_minimum_required(VERSION 3.25)
project(telegram_camera_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV LIBS = ${OpenCV_LIBS}")

add_executable(${PROJECT_NAME} src/proto.cpp)

if(ON_SBC)
    message(STATUS "Platform: Single-board computer (ARM-based)")
    find_package(Threads REQUIRED)
    find_package(OpenSSL REQUIRED)
    set(Boost_NO_WARN_NEW_VERSIONS ON)
    find_package(Boost 1.83 REQUIRED COMPONENTS system regex json container)
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
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_CURL)
    endif()

    target_link_libraries(${PROJECT_NAME} PRIVATE
        /usr/local/lib/libTgBot.a
        ${CMAKE_THREAD_LIBS_INIT}
        ${OPENSSL_LIBRARIES}
        ${Boost_LIBRARIES}
        boost_json
        boost_container
        ${CURL_LIBRARIES}
        ${OpenCV_LIBS}
    )

else()
    message(STATUS "Platform: x86/x86_64")
    find_package(tgbot CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        tgbot::tgbot
        ${OpenCV_LIBS}
    )
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/coco.names ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.cfg ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.weights ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/haarcascade_frontalface_default.xml ${CMAKE_BINARY_DIR}
)
