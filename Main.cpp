#include <iostream>
#include "Tensor.hpp"
//#include <ctime>
//#include <chrono>
using namespace ts;
using namespace std;
int main() {
	{
		// Task 1.1
		puts("Task 1.1: Copy Initialization");
		Tensor<int> t1(2, new int[2]{3, 3}, new int[9]{1, 2, 3, 4, 5, 6, 7, 8, 9});
		t1.print();
		// Task 1.2
		puts("Task 1.2: Random Initialization");
		Tensor<double> t2a = rand<double>(2, new int[2]{3, 3});
		t2a.print();
		Tensor<double> t2b = rand_like<double>(t2a);
		t2b.print();
		// Task 1.3
		puts("Task 1.3.1: Zeros Initialization");
		Tensor<int> t3a = zeros<int>(2, new int[2]{3, 3});
		t3a.print();
		Tensor<int> t3b = zeros_like<int>(t3a);
		t3b.print();
		// Task 1.4
		puts("Task 1.3.2: Ones Initialization");
		Tensor<int> t4a = ones<int>(2, new int[2]{3, 3});
		t4a.print();
		Tensor<int> t4b = ones_like<int>(t4a);
		t4b.print();
		// Task 1.5
		puts("Task 1.3.3: Value Initialization");
		Tensor<int> t5a = full<int>(2, new int[2]{3, 3}, 5);
		t5a.print();
		Tensor<int> t5b = full_like<int>(t5a, 6);
		t5b.print();
		puts("Task 1.4: Pattern Initialization");
		Tensor<int> t6 = eye<int>(3);
		t6.print();
		Tensor<int> t7 = excrange<int>(1, 10, 3);
		t7.print();
		Tensor<int> t8 = incrange<int>(1, 10, 3);
		t8.print();
		puts("End of Task 1.\n");
	}

	{
		// Task 2.1.1
		puts("Task 2.1.1: Indexing");
		Tensor<int> t1(2, new int[2]{5, 5}, new int[25]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
			11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
		t1.print();
		printf("t1[1][3] = %d\n", t1.index(new int[2]{1, 3}));
		// Task 2.1.2
		puts("Task 2.1.2: Slicing");
		Tensor<int> t2 = t1.slice(new pair<int, int> [2]{{1, 3}, {2, 4}});
		printf("t1[1:3][2:4] = \n");
		t2.print();
		// Task 2.2.1
		puts("Task 2.2.1: Concatenation");
		Tensor<int> t3 = t1 + full_like<int>(t1, 25);
		concat(t1, t3, 0).print();
		// Task 2.2.2
		puts("Task 2.2.2: Tile");
		tile(t1, new int[2]{3, 2}).print();
		// Task 2.3
		puts("Task 2.3: Mutating");
		t1.index(new int[2]{1, 3}) = 100;
		t1.print();
		// Task 2.4.1
		puts("Task 2.4.1: Transpose");
		Tensor<int> t4(3, new int[3]{2, 3, 4}, new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
			11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21, 22, 23, 24});
		t4.transpose(1, 2).print();
		// Task 2.4.2
		puts("Task 2.4.2: Permute");
		t4.permute(new int[3]{2, 0, 1}).print();
		// Task 2.5
		puts("Task 2.5: View");
		t4.print(2, new int[2]{4, 6});
	}

	{
        //Task 3.1
        puts("Task 3.1:");
        int a1[2 * 2 * 2 * 2], b1[2 * 2 * 2 * 2];
        for (int i = 0 ; i < 2 * 2 * 2 * 2 ; ++i) {
            a1[i] = rand() % 107;
            b1[i] = rand() % 107 + 1;
        }
        Tensor <int> t1a(4, new int[4]{2, 2, 2, 2}, new int[16]{a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9], a1[10],
         a1[11], a1[12], a1[13], a1[14], a1[15], a1[0]});
        Tensor <int> t1b(4, new int[4]{2, 2, 2, 2}, new int[16]{b1[1], b1[2], b1[3], b1[4], b1[5], b1[6], b1[7], b1[8], b1[9], b1[10],
         b1[11], b1[12], b1[13], b1[14], b1[15], b1[0]});
        t1a.print(); t1b.print(); 
        //ADD
        Tensor <int> t1c_1 = t1a + t1b;
        Tensor <int> t1c_2 = t1a + 3;
        t1c_1.print(); t1c_2.print();
        //SUB
        Tensor <int> t1d_1 = t1a - t1b;
        Tensor <int> t1d_2 = t1a - 3;
        t1d_1.print(); t1d_2.print();
        //MUL
        Tensor <int> t1e_1 = t1a * t1b;
        Tensor <int> t1e_2 = t1a * 3;
        t1e_1.print(); t1e_2.print();
        //DIV
        Tensor <int> t1f_1 = t1a / t1b;
        Tensor <int> t1f_2 = t1a / 3;
        t1f_1.print(); t1f_2.print();
        //LOG
        Tensor <int> t1g = t1a.Log();
        t1g.print();
    }

    {
        //Task 3.2
        puts("Task 3.2:");
        int a1[2 * 2 * 2 * 2], b1[2 * 2 * 2 * 2];
        for (int i = 0 ; i < 2 * 2 * 2 * 2 ; ++i) {
            a1[i] = rand() % 107;
            b1[i] = rand() % 107 + 1;
        }
        Tensor <int> t1a(4, new int[4]{2, 2, 2, 2}, new int[16]{a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9], a1[10],
         a1[11], a1[12], a1[13], a1[14], a1[15], a1[0]});
        Tensor <int> t1b(4, new int[4]{2, 2, 2, 2}, new int[16]{b1[1], b1[2], b1[3], b1[4], b1[5], b1[6], b1[7], b1[8], b1[9], b1[10],
         b1[11], b1[12], b1[13], b1[14], b1[15], b1[0]});
        t1a.print(); t1b.print(); 
        //SUM
        Tensor <int> t1c_1 = sum(t1a, 0);
        Tensor <int> t1c_2 = sum(t1a, 1);
        Tensor <int> t1c_3 = sum(t1a, 2);
        Tensor <int> t1c_4 = sum(t1a, 3);
        t1c_1.print(); t1c_2.print(); t1c_3.print(); t1c_4.print();
        //mean
        int Mean1 = t1a.mean(), Mean2 = t1b.mean();
        cout << Mean1 << " " << Mean2 << endl;
        //max
        int Max1 = t1a.Max(), Max2 = t1b.Max();
        cout << Max1 << " " << Max2 << endl;
        //min
        int Min1 = t1a.Min(), Min2 = t1b.Min();
        cout << Min1 << " " << Min2 << endl;
    }

    {
        //Task 3.3
        puts("Task 3.3:");
        int a1[2 * 2 * 2 * 2], b1[2 * 2 * 2 * 2];
        for (int i = 0 ; i < 2 * 2 * 2 * 2 ; ++i) {
            a1[i] = rand() % 107;
            b1[i] = rand() % 107 + 1;
        }
        Tensor <int> t1a(4, new int[4]{2, 2, 2, 2}, new int[16]{a1[1], a1[2], a1[3], a1[4], a1[5], a1[6], a1[7], a1[8], a1[9], a1[10],
         a1[11], a1[12], a1[13], a1[14], a1[15], a1[0]});
        Tensor <int> t1b(4, new int[4]{2, 2, 2, 2}, new int[16]{b1[1], b1[2], b1[3], b1[4], b1[5], b1[6], b1[7], b1[8], b1[9], b1[10],
         b1[11], b1[12], b1[13], b1[14], b1[15], b1[0]});
        t1a.print(); t1b.print(); 
        t1a.eq(t1b).print(); t1a.ne(t1b).print();
        t1a.gt(t1b).print(); t1a.ge(t1b).print();
        t1a.lt(t1b).print(); t1a.le(t1b).print();
    }

	// {
	// 	// Task 4.1
	// 	puts("Task 4.1: File I/O");
	// 	Tensor<int> t1(3, new int[3]{2, 3, 4}, new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	// 		11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21, 22, 23, 24});
	// 	t1.save("output.txt");
	// 	FILE *fin = fopen("input.txt", "r");
	// 	Tensor<int> t2(fin);
	// 	t2.print();
	// 	fclose(fin);
	// }
	return 0;
}
