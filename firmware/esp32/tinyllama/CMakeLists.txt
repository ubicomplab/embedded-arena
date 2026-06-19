# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

# ESP-IDF writes the build tree under <project>/build. CMakeCache.txt stores absolute
# paths to this project and to IDF_PATH. Do not copy `build/` when seeding the project
# elsewhere (e.g. AgenticHL sandbox): run `idf.py fullclean` or omit `build/` — the
# harness copy step skips `build/` so the first sandbox build configures fresh.

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32-llm-demo)
