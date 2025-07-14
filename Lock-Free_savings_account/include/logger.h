#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <pthread.h>
using namespace std;


void initLogger(const string &filename);
void closeLogger();

void logMessage(const string &message);
string getCurrentTime();

#endif
