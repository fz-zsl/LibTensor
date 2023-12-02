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
    // t1.print();
    // t1.print(2, new int[2]{4, 6});
    // Tensor<int> t2(t1.slice(new pair<int, int>[3]{make_pair(0, 2), make_pair(2, 3), make_pair(2, 4)}));
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
    return 0;
}