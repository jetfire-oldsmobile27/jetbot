cmake_minimum_required(VERSION 3.25)
project(telegram_camera_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/Utility/*.cpp
    ${CMAKE_SOURCE_DIR}/src/Utility/thirdparty/*.cpp  
    ${CMAKE_SOURCE_DIR}/src/ObserverLoop/*.cpp
    ${CMAKE_SOURCE_DIR}/src/Startup/*.cpp
    ${CMAKE_SOURCE_DIR}/src/Tests/*.cpp
)

function(add_all_include_directories base_dir)
    file(GLOB children RELATIVE ${base_dir} ${base_dir}/*)
    foreach(child ${children})
        if(IS_DIRECTORY ${base_dir}/${child})
            target_include_directories(${PROJECT_NAME} PRIVATE ${base_dir}/${child})
            add_all_include_directories(${base_dir}/${child})
        endif()
    endforeach()
endfunction()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV LIBS = ${OpenCV_LIBS}")

add_executable(${PROJECT_NAME} 
    ${PROJECT_SOURCES}
    src/proto.cpp  
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_all_include_directories(${CMAKE_SOURCE_DIR}/src)

if(ON_SBC)
    message(STATUS "Platform build: Single-board computer (ARM-based)")
    find_package(Threads REQUIRED)
    find_package(OpenSSL REQUIRED)
    set(Boost_NO_WARN_NEW_VERSIONS ON)
    find_package(Boost 1.83 REQUIRED COMPONENTS system regex json container)
    find_package(CURL REQUIRED)

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
    
    set(Boost_NO_WARN_NEW_VERSIONS ON)
    find_package(Boost 1.83 REQUIRED COMPONENTS system)
    find_package(CURL REQUIRED)
    
    target_link_libraries(${PROJECT_NAME} PRIVATE
        tgbot::tgbot
        ${OpenCV_LIBS}
        Boost::boost
        CURL::libcurl
        -static-libstdc++
    )
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/coco.names ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.cfg ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.weights ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/haarcascade_frontalface_default.xml ${CMAKE_BINARY_DIR}
)