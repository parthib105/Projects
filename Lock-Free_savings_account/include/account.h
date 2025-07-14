#ifndef ACCOUNT_H
#define ACCOUNT_H

#include <pthread.h>

class SavingsAccount {
private:
    double balance;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int preferred_waiting;
    int ordinary_waiting;

public:
    SavingsAccount();
    ~SavingsAccount();

    void deposit(double amount, int thread_id);
    bool withdrawOrdinary(double amount, int thread_id);
    bool withdrawPreferred(double amount, int thread_id);

    pthread_mutex_t* getMutex();
    pthread_cond_t* getCondition();

    double getBalance();
    void setBalance(double amount);

    int getPreferredWaiting();
    int getOrdinaryWaiting();
    void incrementPreferredWaiting();
    void decrementPreferredWaiting();
    void incrementOrdinaryWaiting();
    void decrementOrdinaryWaiting();
};

#endif