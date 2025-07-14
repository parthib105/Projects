#include "include/account.h"
#include "include/logger.h"
#include <random>
#include <vector>
#include <unistd.h>

extern vector<SavingsAccount> accounts;
extern int p, t;
extern double alpha;

void* thread_function(void* arg) {
    int thread_id = *((int*)arg);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> account_dist(0, p - 1);
    uniform_int_distribution<> op_dist(0, 2);
    uniform_int_distribution<> amount_dist(50, 500);
    exponential_distribution<> wait_dist(1.0 / alpha);

    for (int i = 0; i < t; ++i) {
        int acc_id = account_dist(gen);
        int op = op_dist(gen);
        double amount = amount_dist(gen);

        switch (op) {
            case 0:
                accounts[acc_id].deposit(amount, thread_id);
                break;
            case 1:
                accounts[acc_id].withdrawOrdinary(amount, thread_id);
                break;
            case 2:
                accounts[acc_id].withdrawPreferred(amount, thread_id);
                break;
        }

        usleep((int)(wait_dist(gen) * 1000));
    }

    pthread_exit(NULL);
}