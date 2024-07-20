#include <iostream>
#include <fstream>
#include <pthread.h>
#include <atomic>
#include <unistd.h>
using namespace std;

atomic_flag lock = ATOMIC_FLAG_INIT;   // 0 indicate unlocked, 1 indicates lock

// Global variable matrix C
int** B; 

// Global variable matrix A
int** A;

// Global variable N (no. of rows of NxN matrix)
int N;

// Global variable K (no. of threads)
int K;

// Global variable rowInc
int rowInc;

// Declearing a global counter
int C = 0;

void* multi(void* arg)
{
    int* tid = (int*)arg;
    while (C < N)
    {
        // Critical section
        while(lock.test_and_set(memory_order_acquire));
        int st = C;
        C = C + rowInc;
        int ed = C;
        lock.clear(memory_order_release);

        // remainder section
        // Performing matrix multiplication
        for (int i = st; (i < ed) && (i < N); i++)
        {
            for (int j = 0; j < N; j++)
            {
                int sum = 0;
                for (int k = 0; k < N; k++)
                {
                    sum += A[i][k] * A[k][j];
                }
                B[i][j] = sum;
            }
        }
    }
    pthread_exit(NULL);
}

// Function to deallocate memory for matrix
void deallocateMatrix(int** mat, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        delete[] mat[i];
    }
    delete[] mat;
}

int main()
{
    // Reading from input file
    ifstream inputFile("inp.txt");

    // Check whether it fails to read input
    if (!inputFile) {
        cerr << "Unable to open file input.txt\n";
        return 1;
    }

    // Read N and K from the input file
    inputFile >> N >> K >> rowInc;

    // Define the matrix C and A dynamically
    B = new int*[N];
    A = new int*[N];
    for (int i = 0; i < N; i++)
    {
        B[i] = new int[N];
        A[i] = new int[N];
    }

    // Storing the input in A
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            inputFile >> A[i][j];
        }
    }

    // Close the input file
    inputFile.close(); 
    
    // creating thread ids
    pthread_t threads[K];
    int tID[K];

    // Starting clock time
    clock_t start = clock();

    // Creating threads
    for (int i = 0; i < K; i++)
    {
        tID[i] = i;
        // Creating thread and pass the necessary parameter using dinamically allocated array
        int id = pthread_create(&threads[i], NULL, multi, (void*)&tID[i]);

        // Checking if pthread_create returns error
        if (id != 0)
        {
            cerr << "Aborting! thread not created" << endl;
            break;
        }
    }

    // Wait for other threads to complete tasks
    for (int j = 0; j < K; j++)
    {
        pthread_join(threads[j], NULL);
    }

    // Ending clock time
    clock_t end = clock();

    // Calculating time
    double time = double (end - start) / (double) CLOCKS_PER_SEC;

    // printing time taken
    cout << "Time taken : " << time << endl;
    cout << "N : " << N << " , K : " << K << " , rowInc : " << rowInc << " " <<  endl;

    // Opeining a file to store output
    ofstream outFile("out_tas.txt");

    // printing time taken into the file
    outFile << "Time taken : " << time << endl;

    // Printing into output file
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            outFile << B[i][j] << " ";
        }
        outFile << endl;
    }
    
    // Closing file
    outFile.close();

    // Dealocate memory
    deallocateMatrix(B, N);
    deallocateMatrix(A, N);
    
    return 0;
}