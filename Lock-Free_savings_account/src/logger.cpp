#include "include/logger.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

static ofstream output_file;
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;

void initLogger(const string &filename)
{
    output_file.open(filename);
    if (!output_file.is_open())
    {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
}

void closeLogger()
{
    if (output_file.is_open())
        output_file.close();
}

void logMessage(const string &message)
{
    pthread_mutex_lock(&log_mutex);
    output_file << message << std::endl;
    output_file.flush();
    pthread_mutex_unlock(&log_mutex);
}

string getCurrentTime()
{
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;

    stringstream ss;
    ss << put_time(localtime(&time_t), "%H:%M:%S");
    ss << "." << setfill('0') << setw(3) << ms.count();
    return ss.str();
}
