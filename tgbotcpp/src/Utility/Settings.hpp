#include <boost/json.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>
#include <fstream>

namespace Utility {
namespace fs = std::filesystem;

class Settings {  
public:
  int64_t authorizedUserId = 0;
  bool alert_enabled = true;
  bool unstopable_mode = false;   


  static std::string GetResourceDirFromExePath() {
    char path[1024];
    std::string resource_dir;
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) {
      path[len] = '\0';
      resource_dir = fs::path(path).parent_path().string();
      std::cout << "Ресурсы ищем в: " << resource_dir << std::endl;
    } else {
      resource_dir = ".";
      std::cout << "Не удалось определить путь к исполняемому файлу, "
                   "используем текущую директорию"
                << std::endl;
    }
    return resource_dir;
  };

  void load(const std::string &path) {
    try {
      if (!std::filesystem::exists(path))
        return;
      std::ifstream ifs(path);
      std::string content((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
      boost::json::value jv = boost::json::parse(content);
      boost::json::object obj = jv.as_object();
      if (obj.contains("authorizedUserId"))
        authorizedUserId = obj["authorizedUserId"].as_int64();
      if (obj.contains("alert_enabled"))
        alert_enabled = obj["alert_enabled"].as_bool();
      if (obj.contains("unstopable_mode"))
        unstopable_mode = obj["unstopable_mode"].as_bool();
    } catch (...) {
      std::cerr << "Error loading settings" << std::endl;
    }
  }

  void save(const std::string &path) {
    boost::json::object obj;
    obj["authorizedUserId"] = authorizedUserId;
    obj["alert_enabled"] = alert_enabled;
    obj["unstopable_mode"] = unstopable_mode;
    std::ofstream ofs(path);
    ofs << boost::json::serialize(obj);
  }
};

} // namespace Utility