from conan import ConanFile

class BotConan(ConanFile):
    name = "telegram_camera_bot"
    version = "0.1"
    settings = "os", "compiler", "build_type", "arch"
    
    requires = [
        "tgbot/1.8",
        "opencv/4.11.0"
    ]

    generators = "CMakeToolchain", "CMakeDeps"
