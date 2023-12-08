#include </opt/homebrew/Cellar/libomp/17.0.6/include/omp.h>
#include </opt/homebrew/Cellar/libomp/17.0.6/include/ompt.h>
#include </opt/homebrew/Cellar/libomp/17.0.6/include/omp-tools.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <chrono>
using namespace std;
inline void calc() {
    omp_set_dynamic(0);
    int sum = 1;
    #pragma omp parallel default(shared) num_threads(8) 
    {
        int id = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        for (int j = 1;j <= 1000; ++j) {
            if (j % numThreads != id) continue;
            for (int i = 1; i <= 1000000; i++) {   
                sum = sum * 2 - 1;
            }
        }
    }
}
int main() {
    auto start = chrono::high_resolution_clock::now();
	calc();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken by function: " << elapsed.count() << " seconds" << endl;
}