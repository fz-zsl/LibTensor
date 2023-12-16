#include <iostream>
#include "Tensor.hpp"
#include <ctime>
#include <chrono>
using namespace ts;
using namespace std;
int main() {
    Tensor<int> t1(2, new int[2]{3, 3}, new int[9]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    cout << *einsum(string("ii"), t1) << endl;
    Tensor<int> t2(1, new int[1]{9}, new int[9]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<int> t3(1, new int[1]{9}, new int[9]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    cout << *einsum(string("i,j->ij"), t2, t3) << endl;
    Tensor<int> t4(2, new int[2]{3, 3}, new int[9]{1, 2, 3, 4, 5, 6, 7, 8, 9});
    // cout << *einsum(string("ik,kj->ij"), t1, t4) << endl;
    Tensor<int> t5(3, new int[3]{2, 3, 3}, new int[18]{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor<int> t6(3, new int[3]{2, 3, 3}, new int[18]{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    cout << *einsum(string("bik,bkj->bij"), t5, t6) << endl;
    // cout << *einsum(string("bik,kj->bij"), t5, t4) << endl;
    Tensor<double> t7(3, new int[3]{3, 5, 4}, new double[60] {
        0.6542, 0.7486, 0.9158, 0.6722,
        0.6426, 0.5420, 0.6805, 0.9508,
        0.2257, 0.8992, 0.8890, 0.1071,
        0.2404, 0.2429, 0.0213, 0.6407,
        0.2503, 0.6149, 0.4670, 0.8534,
        0.7803, 0.4479, 0.1876, 0.3266,
        0.5837, 0.6906, 0.7561, 0.2645,
        0.1560, 0.1896, 0.5526, 0.3981,
        0.3586, 0.4266, 0.2531, 0.2883,
        0.2547, 0.9360, 0.9020, 0.0631,
        0.9858, 0.8774, 0.1167, 0.4705,
        0.0710, 0.1224, 0.5544, 0.6838,
        0.5775, 0.3225, 0.9105, 0.4762,
        0.4499, 0.6557, 0.7880, 0.5779,
        0.1033, 0.3359, 0.3628, 0.1540
    });

    Tensor<double> t8(2, new int[2]{2, 5}, new double[10] {
        0.1169, 0.9707, 0.7564, 0.2771, 0.9437,
        0.9117, 0.6582, 0.7966, 0.5469, 0.7585
    });

    Tensor<double> t9(2, new int[2]{2,4}, new double[8] {
        0.1720, 0.6713, 0.1147, 0.7775,
        0.5898, 0.4999, 0.0623, 0.6685
    });

    cout << *einsum(string("bn,anm,bm->ba"), t8, t7, t9) << endl;
    return 0;
}