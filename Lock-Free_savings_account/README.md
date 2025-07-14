## 🏦 Banking Simulation (Multithreaded using `pthreads`)

This project simulates a multithreaded banking system where multiple users (threads) perform deposits and withdrawals (ordinary and preferred) on shared accounts. It demonstrates synchronization using mutexes and condition variables.

---

### 📁 Project Structure

```
banking_simulation/
├── include/
│   ├── account.h            # Declaration of SavingsAccount class
│   ├── logger.h             # Thread-safe logging functions
│   └── transactions.h       # Deposit & withdrawal wrappers
├── src/
│   ├── account.cpp          # Account method implementations
│   ├── logger.cpp           # Logging implementation
│   └── thread_routine.cpp   # Thread execution logic
├── inp-params.txt           # Input parameters file
├── output.txt               # Output log (generated after execution)
├── main.cpp                 # Program entry point
└── README.md                # Project documentation
```

---

### ⚙️ Compilation Instructions

To compile the project, use the following command from the root of the project directory:

```bash
g++ -std=c++11 -pthread src/*.cpp -Iinclude -o banking_simulation
```

This generates an executable called `banking_simulation`.

---

### 📥 Input File: `inp-params.txt`

The program reads the following 4 parameters from this file (all space-separated):

```
n p t alpha
```

Where:

* `n`: Number of threads (users)
* `p`: Number of accounts
* `t`: Number of transactions per thread
* `alpha`: Mean of the exponential distribution used for wait time between transactions

🧪 **Example**:

```
5 4 10 1.2
```

---

### 🚀 How to Run

```bash
./banking_simulation
```

Output is written to both console and the file `output.txt`.

---

### 🧾 Output File: `output.txt`

The output file logs all events with timestamps, including:

* Requests
* Critical section entry
* Block and wake-up notifications
* Final account balances

---

### 🔐 Features

* **Mutual exclusion** using `pthread_mutex_t`
* **Condition-based blocking and signaling** for preferred/ordinary withdrawal
* **Preferred withdrawal priority** over ordinary
* **Thread-safe logging**
* **Exponential wait time** simulation between transactions

---

### 👨‍💻 Author

**Parthib Ghosh**
Computational Engineering,
IIT Hyderabad

---
