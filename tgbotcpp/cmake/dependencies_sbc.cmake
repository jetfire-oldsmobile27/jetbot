# =============================================================================
# Установка зависимостей для одноплатника (Debian 12)
# Запустите эти команды в терминале:
# =============================================================================
#
# sudo apt update
# sudo apt install -y \
#     build-essential \
#     cmake \
#     git \
#     libopencv-dev \
#     libssl-dev \
#     libboost-system-dev \
#     libcurl4-openssl-dev \
#     libpthread-stubs0-dev
#echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free" | sudo tee /etc/apt/sources.list.d/backports.list
#sudo apt install -t bookworm-backports libboost1.83-all-dev libboost-json1.83-dev libboost-system1.83-dev libboost-regex1.83-dev
#
# =============================================================================
# Рекомендуется установить TgBot++ вручную из исходников:
# =============================================================================
#
# git clone https://github.com/reo7sp/tgbot-cpp.git
# cd tgbot-cpp
# mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)
# sudo make install
#
# Это установит:
#   - /usr/local/lib/libTgBot.a
#   - /usr/local/include/tgbot/*.h
#
# =============================================================================

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(CURL REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV: ${OpenCV_LIBS}")
endif()

if(Threads_FOUND)
    message(STATUS "Threads: OK")
endif()

if(OPENSSL_FOUND)
    message(STATUS "OpenSSL: ${OPENSSL_LIBRARIES}")
endif()

if(Boost_FOUND)
    message(STATUS "Boost: ${Boost_LIBRARIES}")
endif()

if(CURL_FOUND)
    message(STATUS "CURL: ${CURL_LIBRARIES}")
    add_compile_definitions(HAVE_CURL)
endif()

# Проверка наличия libTgBot.a
if(EXISTS "/usr/local/lib/libTgBot.a")
    message(STATUS "libTgBot.a found: /usr/local/lib/libTgBot.a")
else()
    message(FATAL_ERROR "libTgBot.a not found! Please install TgBot++ from source.")
endif()

# Проверка заголовков TgBot
if(EXISTS "/usr/local/include/tgbot/TgBot.h")
    message(STATUS "TgBot headers found: /usr/local/include/tgbot/")
else()
    message(FATAL_ERROR "TgBot headers not found! Please install TgBot++ from source.")
endif()