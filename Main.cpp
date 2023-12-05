#include <iostream>
#include "Tensor.hpp"
using namespace ts;
using namespace std;

int main() {
    // Tensor<int> t1(3, new int[3]{2, 3, 4},
    //     new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //                 21, 22, 23, 24}
    // );
    // Tensor<int> t2(3, new int[3]{2, 3, 4},
    //     new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //                 21, 22, 23, 24}
    // );
    // Tensor<int> t3(t1.add(t2));
    // t3.print();
    // Tensor<int> t4(t1 + t2);
    // t4.print();
    // Tensor<int> t5(add(t1, t2));
    // t5.print();
    // Tensor<int> t1(3, new int[3]{2, 3, 4},
    //     new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //                 21, 22, 23, 24}
    // );
    // Tensor<int> t2(3, new int[3]{2, 3, 4},
    //     new int[24]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //                 21, 22, 23, 24}
    // );
    // Tensor<int> t3(t1.sub(t2));
    // t3.print();
    // Tensor<int> t4(t1 - t2);
    // t4.print();
    // Tensor<int> t5(sub(t1, t2));
    // t5.print();
    // Tensor<int> t1(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t2(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t3(t1.mul(t2));
    // t3.print();
    // Tensor<int> t4(t1 * t2);
    // t4.print();
    // Tensor<int> t5(mul(t1, t2));
    // t5.print();
    // Tensor<int> t1(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t2(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t3(t1.Div(t2));
    // t3.print();
    // Tensor<int> t4(t1 / t2);
    // t4.print();
    // Tensor<int> t5(Div(t1, t2));
    // t5.print();
    // Tensor<int> t1(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t3(t1.Log());
    // t3.print();
    // Tensor<int> t4(Log(t1));
    // t4.print();
    // Tensor<int> t1(3, new int[3]{2, 2, 2},
    //     new int[8]{11, 22, 333, 44, 5, 66, 7, 8}
    // );
    // Tensor<int> t2(3, new int[3]{2, 2, 2},
    //     new int[8]{1, 2, 3, 4, 5, 6, 7, 8}
    // );
    // Tensor<int> t3(t1.sum(1));
    // //t3.print();
    // Tensor<int> t4(sum(t1, 1));
    //t4.print();
    //printf("%d %d %d %d %d %d\n", t4.mean(), mean(t4), t4.Min(), t4.Max(), Min(t4), Max(t4));
    // t1.eq(t2).print(); (t1 == t2).print();
    // t3.eq(t4).print(); (t3 == t4).print(); 
    // t1.ne(t2).print(); (t1 != t2).print();
    // t3.ne(t4).print(); (t3 != t4).print(); 
    // t1.gt(t2).print(); (t1 > t2).print();
    // t3.gt(t4).print(); (t3 > t4).print();
    // t1.ge(t2).print(); (t1 >= t2).print();
    // t3.ge(t4).print(); (t3 >= t4).print(); 
    // t1.lt(t2).print(); (t1 < t2).print();
    // t3.lt(t4).print(); (t3 < t4).print(); 
    // t1.le(t2).print(); (t1 <= t2).print();
    // t3.le(t4).print(); (t3 <= t4).print(); 
    //t1.print();
    // t1.print(2, new int[2]{4, 6});
    //Tensor<int> t2(t1.slice(new pair<int, int>[3]{make_pair(0, 2), make_pair(2, 3), make_pair(2, 4)}));
    // t2.print();
    // Tensor<int> t3(t1.slice(new pair<int, int>[3]{make_pair(0, 2), make_pair(2, -1), make_pair(2, 4)}));
    // t3.print();
    // t3.tile(new int[2]{2, 3}).print();
    // Tensor<int> t4(2, new int[2]{2, 2}, new int[4]{1, 2, 3, 4});
    // t1.modify(new pair<int, int>[3]{make_pair(0, 2), make_pair(2, 3), make_pair(2, 4)}, t4);
    // t1.print();
    // t1.permute(new int[3]{2, 0, 1}).print();
    // t1.transpose(1, 2).print();
    // t1.transpose(0, 2).print();
    // print(t1);
    // rand<double>(3, new int[3]{2, 3, 4}).print();
    // rand_like<double>(t1).print();
    // full<int>(3, new int[3]{2, 3, 4}, 5).print();
    // full_like<int>(t1, 6).print();
    // eye<int>(3).print();
    // Tensor<int> t2 = excrange<int>(1, 10, 3);
    // t2.print();
    // incrange<int>(1, 10, 3).print();
    // Tensor<int> t1(1, new int[1]{3}, new int[3]{1, 2, 3});
    // t1.print();
    // Tensor<bool> t1(3, new int[3]{2, 2, 2},
    //     new bool[8]{true, false, true, false, true, false, true, false}
    // );
    // t1.print();
    // print(t1);
    // Tensor<bool> t2(1, new int[1]{8},
    //     new bool[8]{true, false, true, false, true, false, true, false}
    // );
    // t2.print();
    // print(t2);
    int *arr = new int[512];
    for (int i = 0; i < 512; i++) {
        arr[i] = i;
    }
    FILE *fout = fopen("test.txt", "w");
    Tensor<int> t4(3, new int[3]{8, 8, 8}, arr);
    t4.print(fout, 0);
    fclose(fout);
    FILE *fin = fopen("test.txt", "r");
    Tensor<int> t5(fin);
    t5.print();
    fclose(fin);
    return 0;
}