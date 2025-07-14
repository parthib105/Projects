#include "include/account.h"
#include "include/logger.h"
#include <iostream>
#include <string>
#include <unistd.h>

using namespace std;

SavingsAccount::SavingsAccount()
    : balance(0.0), preferred_waiting(0), ordinary_waiting(0)
{
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&condition, NULL);
}

SavingsAccount::~SavingsAccount()
{
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&condition);
}

void SavingsAccount::deposit(double amount, int thread_id)
{
    string time_str = getCurrentTime();
    logMessage("Thr" + to_string(thread_id) + " requests for a deposit of amount " +
               to_string((int)amount) + " into the account at " + time_str + ".");
    printf("Thr%d requests for a deposit of amount %d at %s.\n", thread_id, (int)amount, time_str.c_str());

    pthread_mutex_lock(&mutex);

    time_str = getCurrentTime();
    logMessage("Thr" + to_string(thread_id) + " enters the CS to perform a deposit of amount " +
               to_string((int)amount) + " and wakes all threads at " + time_str + ".");
    printf("Thr%d enters the CS to perform a deposit of amount %d and wakes all threads at %s.\n", thread_id, (int)amount, time_str.c_str());

    balance += amount;
    pthread_cond_broadcast(&condition);

    pthread_mutex_unlock(&mutex);
}

bool SavingsAccount::withdrawOrdinary(double amount, int thread_id)
{
    pthread_mutex_lock(&mutex);
    while (balance < amount || preferred_waiting > 0)
    {
        string time_str = getCurrentTime();
        logMessage("Thr" + to_string(thread_id) + " requesting for withdrawal of amount " +
                   to_string((int)amount) + " blocks at " + time_str + ".");
        printf("Thr%d requesting for withdrawal of amount %d blocks at %s.\n", thread_id, (int)amount, time_str.c_str());

        ordinary_waiting++;
        pthread_cond_wait(&condition, &mutex);
        ordinary_waiting--;
    }

    balance -= amount;

    string time_str = getCurrentTime();
    logMessage("Thr" + to_string(thread_id) + " wakes up and deducts " +
               to_string((int)amount) + " at " + time_str + ".");
    printf("Thr%d wakes up and deducts %d at %s.\n", thread_id, (int)amount, time_str.c_str());

    pthread_mutex_unlock(&mutex);
    return true;
}

bool SavingsAccount::withdrawPreferred(double amount, int thread_id)
{
    pthread_mutex_lock(&mutex);
    while (balance < amount)
    {
        string time_str = getCurrentTime();
        logMessage("Thr" + to_string(thread_id) + " requesting for preferred withdrawal of amount " +
                   to_string((int)amount) + " blocks at " + time_str + ".");
        printf("Thr%d requesting for preferred withdrawal of amount %d blocks at %s.\n", thread_id, (int)amount, time_str.c_str());

        preferred_waiting++;
        pthread_cond_wait(&condition, &mutex);
        preferred_waiting--;
    }

    balance -= amount;

    string time_str = getCurrentTime();
    logMessage("Thr" + to_string(thread_id) + " wakes up and deducts " +
               to_string((int)amount) + " at " + time_str + ".");
    printf("Thr%d wakes up and deducts %d at %s.\n", thread_id, (int)amount, time_str.c_str());

    pthread_mutex_unlock(&mutex);
    return true;
}

// Accessors
pthread_mutex_t* SavingsAccount::getMutex() { return &mutex; }
pthread_cond_t* SavingsAccount::getCondition() { return &condition; }

double SavingsAccount::getBalance() { return balance; }
void SavingsAccount::setBalance(double amount) { balance = amount; }

int SavingsAccount::getPreferredWaiting() { return preferred_waiting; }
int SavingsAccount::getOrdinaryWaiting() { return ordinary_waiting; }
void SavingsAccount::incrementPreferredWaiting() { preferred_waiting++; }
void SavingsAccount::decrementPreferredWaiting() { preferred_waiting--; }
void SavingsAccount::incrementOrdinaryWaiting() { ordinary_waiting++; }
void SavingsAccount::decrementOrdinaryWaiting() { ordinary_waiting--; }
