from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    # Полностью переопределяем conan_data
    conan_data = {
        "sources": {
            "6.10.1": {
                "url": ["https://ftp.nluug.nl/languages/qt/archive/qt/6.10/6.10.1/single/qt-everywhere-src-6.10.1.tar.xz"],
                "sha256": "0ed08b079719394303cd2054b66b2dc0c5895ceeb88fb6131c18991c980bf00f"
            }
        }
    }

    def requirements(self):
        self.requires("qt/6.10.1")

    def layout(self):
        cmake_layout(self)