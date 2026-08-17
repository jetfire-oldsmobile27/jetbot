#!/usr/bin/env bash

# Before download this repository and add its recipes to Conan
#git clone https://github.com/tttapa/docker-arm-cross-toolchain.git
#conan remote add tttapa-docker-arm-cross-toolchain ./docker-arm-cross-toolchain

conan install . --output-folder=build --build=missing --profile:build=default --profile:host=./aarch64-rpi3-linux-gnu
cmake --build --parallel --preset conan-release # тут ошибка
cmake --preset conan-release --fresh
