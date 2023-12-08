#include </opt/homebrew/Cellar/libomp/17.0.6/include/omp.h>
#include </opt/homebrew/Cellar/libomp/17.0.6/include/ompt.h>
#include </opt/homebrew/Cellar/libomp/17.0.6/include/omp-tools.h>
#include <stdio.h>
#include <iostream>
using namespace std;
int main() {
	omp_set_num_threads(4);
#pragma omp parallel for
	for (int i = 0; i < 3; i++)
		printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
	getchar();
}