#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

namespace gpu_huffman {

enum class LogLevel {
    INFO,
    SUCCESS,
    WARNING,
    ERROR,
    DEBUG
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        
        std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << "] ";

        switch (level) {
            case LogLevel::INFO:
                std::cout << "\033[1;34m[INFO]\033[0m ";
                break;
            case LogLevel::SUCCESS:
                std::cout << "\033[1;32m[SUCCESS]\033[0m ";
                break;
            case LogLevel::WARNING:
                std::cout << "\033[1;33m[WARNING]\033[0m ";
                break;
            case LogLevel::ERROR:
                std::cerr << "\033[1;31m[ERROR]\033[0m ";
                break;
            case LogLevel::DEBUG:
                std::cout << "\033[1;36m[DEBUG]\033[0m ";
                break;
        }

        if (level == LogLevel::ERROR) {
            std::cerr << message << std::endl;
        } else {
            std::cout << message << std::endl;
        }
    }

    static void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    static void success(const std::string& msg) { log(LogLevel::SUCCESS, msg); }
    static void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    static void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    static void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
};

} // namespace gpu_huffman
