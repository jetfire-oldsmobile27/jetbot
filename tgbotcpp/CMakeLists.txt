cmake_minimum_required(VERSION 3.25)
project(telegram_camera_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/Utility/*.cpp
    ${CMAKE_SOURCE_DIR}/src/Utility/Debug-server/*.cpp
    ${CMAKE_SOURCE_DIR}/src/Utility/thirdparty/*.cpp  
    ${CMAKE_SOURCE_DIR}/src/ObserverLoop/*.cpp
    ${CMAKE_SOURCE_DIR}/src/ObserverLoop/Detection/*.cpp
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
find_package(indicators REQUIRED)
find_package(onnxruntime REQUIRED)

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
    find_package(Boost REQUIRED COMPONENTS system regex json container thread)
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
        indicators::indicators
        onnxruntime::onnxruntime
    )

else()
    message(STATUS "Platform: x86/x86_64")
    find_package(tgbot CONFIG REQUIRED)
    
    set(Boost_NO_WARN_NEW_VERSIONS ON)
    find_package(Boost REQUIRED COMPONENTS system regex json container thread)
    find_package(CURL REQUIRED)
    
    target_link_libraries(${PROJECT_NAME} PRIVATE
        tgbot::tgbot
        ${OpenCV_LIBS}
        Boost::boost
        CURL::libcurl
        indicators::indicators
        -static-libstdc++
        onnxruntime::onnxruntime
    )
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/coco.names ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.cfg ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/yolov3-tiny.weights ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/../test-vision/haarcascade_frontalface_default.xml ${CMAKE_BINARY_DIR}
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.2)
    message(STATUS "CMake ${CMAKE_VERSION} supports CPack AppImage generator")
    
    install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
    )
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/src/assets/telegram_camera_bot.png")
        message(STATUS "Installing icon to AppDir root: ${CMAKE_SOURCE_DIR}/src/assets/telegram_camera_bot.png -> ${PROJECT_NAME}.png")
        
        install(FILES "${CMAKE_SOURCE_DIR}/src/assets/telegram_camera_bot.png"
            DESTINATION .
            RENAME "${PROJECT_NAME}.png"
        )
    else()
        message(FATAL_ERROR "Icon file not found: ${CMAKE_SOURCE_DIR}/src/assets/telegram_camera_bot.png")
    endif()

    set(DESKTOP_FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.desktop")
    file(WRITE ${DESKTOP_FILE}
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Telegram Camera Bot\n"
        "Comment=Telegram bot for camera monitoring\n"
        "Exec=${PROJECT_NAME}\n"
        "Icon=${PROJECT_NAME}\n"
        "Terminal=true\n"
        "Categories=Utility;\n"
    )
    
    install(FILES ${DESKTOP_FILE}
        DESTINATION share/applications
    )
    
    install(CODE [[
        file(GET_RUNTIME_DEPENDENCIES
            EXECUTABLES $<TARGET_FILE:telegram_camera_bot>
            RESOLVED_DEPENDENCIES_VAR resolved_deps
        )

        foreach(dep ${resolved_deps})
            # copy the symlink
            file(COPY ${dep} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

            # Resolve the real path of the dependency (follows symlinks)
            file(REAL_PATH ${dep} resolved_dep_path)

            # Copy the resolved file to the destination
            file(COPY ${resolved_dep_path} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        endforeach()
    ]])

    
    # Настройка CPack
    set(CPACK_GENERATOR "AppImage")
    set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
    set(CPACK_PACKAGE_VERSION "1.0.0")
    set(CPACK_PACKAGE_VENDOR "JetSystems")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Telegram Camera Bot")
    set(CPACK_PACKAGE_CONTACT "contact@example.com")
    
    # Настройки для AppImage
    set(CPACK_APPIMAGE_DESKTOP_FILE "${PROJECT_NAME}.desktop")
    set(CPACK_PACKAGE_ICON "telegram_camera_bot")
    
    # Опциональные настройки AppImage
    # set(CPACK_APPIMAGE_UPDATE_INFORMATION "zsync|https://example.com/appimage/latest.AppImage.zsync")
    # set(CPACK_APPIMAGE_GUESS_UPDATE_INFORMATION ON)
    # set(CPACK_APPIMAGE_COMPRESSOR "xz")
    # set(CPACK_APPIMAGE_SIGN ON)
    # set(CPACK_APPIMAGE_SIGN_KEY "your-gpg-key-id")
    set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
    
    include(CPack)
else()
    message(WARNING "CPack AppImage generator requires CMake 4.2 or higher. Current version: ${CMAKE_VERSION}")
endif()