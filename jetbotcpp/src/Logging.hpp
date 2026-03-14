#pragma once
#include <string>
#include <filesystem>

std::filesystem::path getLogFilePath();
void logMsg(const std::string &msg);
std::string readFile(const std::string &path);
void cleanupOldVideos();