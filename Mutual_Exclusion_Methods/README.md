# Dynamic Matrix Squaring

## Overview

This project involves performing parallel matrix multiplication using a dynamic mechanism in C++. The goal is to compute the square of a matrix A in parallel, with dynamic allocation of rows to threads. The project includes the implementation of different mutual exclusion algorithms to handle synchronization issues.

## Objectives

- Implement parallel matrix squaring with dynamic row allocation.
- Use the following mutual exclusion algorithms:
  - Test-And-Set (TAS)
  - Compare-And-Swap (CAS)
  - Bounded Compare-And-Swap (Bounded CAS)
  - Atomic increment (provided by the C++ atomic library)
- Measure and compare the performance of each mutual exclusion method.

## Mutual Exclusion Methods

### Test-And-Set (TAS)
The Test-And-Set method is a simple locking mechanism where a thread repeatedly tests and sets a lock variable to gain access to a critical section. If the lock is already held by another thread, the calling thread will continue to test the lock until it becomes available.

### Compare-And-Swap (CAS)
The Compare-And-Swap method is an atomic instruction used in multithreading to achieve synchronization. It compares the contents of a memory location to a given value and, if they are the same, modifies the contents of that memory location to a new given value. This is used to implement lock-free data structures.

### Bounded Compare-And-Swap (Bounded CAS)
Bounded CAS is an extension of the CAS method where a limit is set on the number of times a thread can attempt the CAS operation. This helps in reducing the contention and spinning that can occur in a high-contention environment, thereby improving performance in some scenarios.

### Atomic Increment
Atomic increment utilizes atomic operations provided by the C++ atomic library to safely increment a shared counter without the need for explicit locking. This method leverages hardware support for atomic operations, ensuring that increments are performed without race conditions.

## Input and Output

- **Input File (inp.txt)**:
  - The number of rows in matrix A (N)
  - The number of threads (K)
  - Row increment value (rowInc)
  - Matrix A in row-major order

- **Output File (out.txt)**:
  - The resulting squared matrix produced by each mutual exclusion method
  - Time taken to compute the squared matrix for each method
