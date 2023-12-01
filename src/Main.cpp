#include "Tensor.h"
using namespace ts;

int main() {
    Tensor<int> t1(3, new int[3]{2, 3, 4});
    // t1.print();
    return 0;
}