#include <iostream>
#include "Tensor.hpp"
#include <ctime>
#include <chrono>
using namespace ts;
using namespace std;

int main() {
    auto start = chrono::high_resolution_clock::now();
    int n = 16777216;
    Tensor<int> t1(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<double> t2(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<float> t3(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<long long> t4(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<bool> t5(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<char> t6(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    for (int i = 0; i < n; i++) {
        t1.data[i] = i; t2.data[i] = 1.0 / i; t3.data[i] = 2.0 / i; 
        t4.data[i] = i * i; t5.data[i] = i % 2; t6.data[i] = i % 128;
    }
    Tensor<int> a1(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<double> a2(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<float> a3(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<long long> a4(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<bool> a5(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<char> a6(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    for (int i = 0; i < n; i++) {
        a1.data[i] = 5; a2.data[i] = i; a3.data[i] = i;
        a4.data[i] = 4; a5.data[i] = 1; a6.data[i] = i % 64;
    }
    Tensor<int> b1(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<double> b2(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<float> b3(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<long long> b4(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<bool> b5(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<char> b6(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    b1 = add(a1, t1);  b1 = b1.add(a1); b1 = b1.add(t1); b1 = b1.add(5); 
    b2 = add(a2, t2);  b2 = b2.add(a2); b2 = b2.add(t2); b2 = b2.add(5.0);
    b3 = add(a3, t3);  b3 = b3.add(a3); b3 = b3.add(t3); b3 = b3.add(5.0f);
    b4 = add(a4, t4);  b4 = b4.add(a4); b4 = b4.add(t4); b4 = b4.add(5ll);
    b5 = add(a5, t5);  b5 = b5.add(a5); b5 = b5.add(t5); b5 = b5.add(true);
    b6 = add(a6, t6);  b6 = b6.add(a6); b6 = b6.add(t6); b6 = b6.add('a');
    b1 = sub(a1, t1);  b1 = b1.sub(a1); b1 = b1.sub(t1); b1 = b1.sub(5);
    b2 = sub(a2, t2);  b2 = b2.sub(a2); b2 = b2.sub(t2); b2 = b2.sub(5.0);
    b3 = sub(a3, t3);  b3 = b3.sub(a3); b3 = b3.sub(t3); b3 = b3.sub(5.0f);
    b4 = sub(a4, t4);  b4 = b4.sub(a4); b4 = b4.sub(t4); b4 = b4.sub(5ll);
    b5 = sub(a5, t5);  b5 = b5.sub(a5); b5 = b5.sub(t5); b5 = b5.sub(true);
    b6 = sub(a6, t6);  b6 = b6.sub(a6); b6 = b6.sub(t6); b6 = b6.sub('a');
    b1 = mul(a1, t1);  b1 = b1.mul(a1); b1 = b1.mul(t1); b1 = b1.mul(5);
    b2 = mul(a2, t2);  b2 = b2.mul(a2); b2 = b2.mul(t2); b2 = b2.mul(5.0);
    b3 = mul(a3, t3);  b3 = b3.mul(a3); b3 = b3.mul(t3); b3 = b3.mul(5.0f);
    b4 = mul(a4, t4);  b4 = b4.mul(a4); b4 = b4.mul(t4); b4 = b4.mul(5ll);
    b5 = mul(a5, t5);  b5 = b5.mul(a5); b5 = b5.mul(t5); b5 = b5.mul(true);
    b6 = mul(a6, t6);  b6 = b6.mul(a6); b6 = b6.mul(t6); b6 = b6.mul('a');
    b1 = Div(a1, t1);  b1 = b1.Div(a1); b1 = b1.Div(t1); b1 = b1.Div(5);
    b2 = Div(a2, t2);  b2 = b2.Div(a2); b2 = b2.Div(t2); b2 = b2.Div(5.0);
    b3 = Div(a3, t3);  b3 = b3.Div(a3); b3 = b3.Div(t3); b3 = b3.Div(5.0f);
    b4 = Div(a4, t4);  b4 = b4.Div(a4); b4 = b4.Div(t4); b4 = b4.Div(5ll);
    b5 = Div(a5, t5);  b5 = b5.Div(a5); b5 = b5.Div(t5); b5 = b5.Div(true);
    b6 = Div(a6, t6);  b6 = b6.Div(a6); b6 = b6.Div(t6); b6 = b6.Div('a');
    int ans1; double ans2; float ans3; long long ans4; bool ans5; char ans6;
    b1 = t1.Log(); b2 = t2.Log(); b3 = t3.Log(); b4 = t4.Log(); b6 = t6.Log();
    ans1 = t1.mean(); ans2 = t2.mean(); ans3 = t3.mean(); ans4 = t4.mean(); ans6 = t6.mean();
    ans2 = t1.Max(); ans2 = t2.Max(); ans3 = t3.Max(); ans4 = t4.Max(); ans6 = t6.Max();
    ans2 = t1.Min(); ans2 = t2.Min(); ans3 = t3.Min(); ans4 = t4.Min(); ans6 = t6.Min();
    Tensor<bool> com(12, new int[12]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<int> c1(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<double> c2(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<float> c3(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<long long> c4(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<bool> c5(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    Tensor<char> c6(11, new int[11]{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
    for (int i = 0; i < 12; ++i) c1 = t1.sum(i);
    for (int i = 0; i < 12; ++i) c2 = t2.sum(i);
    for (int i = 0; i < 12; ++i) c3 = t3.sum(i);
    for (int i = 0; i < 12; ++i) c4 = t4.sum(i);
    for (int i = 0; i < 12; ++i) c5 = t5.sum(i);
    for (int i = 0; i < 12; ++i) c6 = t6.sum(i);
    for (int i = 0; i < 12; ++i) c1 = sum(t1, i);
    for (int i = 0; i < 12; ++i) c2 = sum(t2, i);
    for (int i = 0; i < 12; ++i) c3 = sum(t3, i);
    for (int i = 0; i < 12; ++i) c4 = sum(t4, i);
    for (int i = 0; i < 12; ++i) c5 = sum(t5, i);
    for (int i = 0; i < 12; ++i) c6 = sum(t6, i);
    com = (t1 == a1); com = (t2 == a2); com = (t3 == a3); com = (t4 == a4); com = (t5 == a5); com = (t6 == a6);
    com = (t1 != a1); com = (t2 != a2); com = (t3 != a3); com = (t4 != a4); com = (t5 != a5); com = (t6 != a6);
    com = (t1 > a1); com = (t2 > a2); com = (t3 > a3); com = (t4 > a4); com = (t5 > a5); com = (t6 > a6);
    com = (t1 < a1); com = (t2 < a2); com = (t3 < a3); com = (t4 < a4); com = (t5 < a5); com = (t6 < a6);
    com = (t1 >= a1); com = (t2 >= a2); com = (t3 >= a3); com = (t4 >= a4); com = (t5 >= a5); com = (t6 >= a6);
    com = (t1 <= a1); com = (t2 <= a2); com = (t3 <= a3); com = (t4 <= a4); com = (t5 <= a5); com = (t6 <= a6);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken by function: " << elapsed.count() << " seconds" << endl;
    return 0;
}