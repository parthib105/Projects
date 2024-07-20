#include <iostream>
#include <fstream>
#include <random>

using namespace std;

int main() 
{
    int N = 2048;
    int K = 16;
    int rowInc = 1;

    ofstream outFile("inp.txt");

    if (!outFile.is_open())
    {
        cerr << "Error : Unable to open the file" << endl;
        return 1;
    }

    outFile << N << " " << K << " " << rowInc << endl;

    // seed the random number generator
    random_device rd;
    mt19937 gen(rd());

    // create a uniform distribution for integers within the range
    uniform_int_distribution<> distrib(1, N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int rn = distrib(gen);
            outFile << rn << " ";
        }
        outFile << endl;
    }
    
    outFile.close();

    return 0;
}