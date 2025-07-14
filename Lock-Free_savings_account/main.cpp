#include "include/account.h"
#include "include/logger.h"
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <vector>

int n, p, t;
double alpha;
vector<SavingsAccount> accounts;

extern void* thread_function(void* arg);

int main() {
    ifstream input("inp-params.txt");
    if (!input.is_open()) {
        cerr << "Failed to open inp-params.txt" << endl;
        return 1;
    }
    input >> n >> p >> t >> alpha;
    input.close();

    accounts.resize(p);
    initLogger("output.txt");
    logMessage("Program started with parameters: n=" + to_string(n) + ", p=" + to_string(p) + ", t=" + to_string(t) + ", alpha=" + to_string(alpha));

    srand(time(NULL));
    for (int i = 0; i < p; ++i) {
        accounts[i].setBalance(100 + rand() % 901);
        logMessage("Account a" + to_string(i) + " initialized with balance: " + to_string(accounts[i].getBalance()));
    }

    vector<pthread_t> threads(n);
    vector<int> ids(n);

    for (int i = 0; i < n; ++i) {
        ids[i] = i + 1;
        pthread_create(&threads[i], NULL, thread_function, &ids[i]);
    }

    for (int i = 0; i < n; ++i) pthread_join(threads[i], NULL);

    logMessage("Final account balances:");
    for (int i = 0; i < p; ++i) {
        logMessage("Account a" + to_string(i) + ": " + to_string(accounts[i].getBalance()));
    }

    closeLogger();
    cout << "Program finished. Check output.txt for logs." << endl;
    return 0;
}
