#include <iostream>
#include <cstdlib>
#include <cstring>
//PREPEND END

struct Data
{
    int *entry;
    size_t row, col;
    size_t ref_cnt;

    Data(size_t row, size_t col):
        row(row), col(col), ref_cnt(0)
    { entry = new int[row * col]{}; }

    ~Data()
    { delete[] entry; }
};

struct Matrix
{
    Data *data;         // the ptr pointing to the entries
    size_t start;       // the starting index of ROI
    size_t row, col;    // the shape of ROI
    
    Matrix():
        data(nullptr), start(0), row(0), col(0) {}
    
    ~Matrix()
    {} // something invisible
};

void print_matrix(Matrix &mat) {
    for (size_t r = 0; r < mat.row; r++)
    {
        size_t head = mat.start + r * mat.data -> col;
        for (size_t c = 0; c < mat.col; c++)
            std::cout << mat.data -> entry[head + c] << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;
}
void unload_data(Matrix& mat) {
    if (mat.data != nullptr) {
        mat.data->ref_cnt--;
        if (mat.data->ref_cnt == 0) {
            mat.data->~Data();
        }
        mat.data = nullptr;
    }
}
void load_data(Matrix& mat, Data* data, size_t start, size_t row, size_t col) {
    unload_data(mat);
    mat.data = data, mat.start = start, mat.row = row, mat.col = col; data->ref_cnt++;
}
void shallow_copy(Matrix& dest, Matrix& src) {
    unload_data(dest);
    dest.data = src.data, dest.start = src.start, dest.row = src.row, dest.col = src.col;
    if (src.data != nullptr) ++src.data->ref_cnt;
}
void deep_copy(Matrix& dest, Matrix& src) {
    unload_data(dest);
    if (src.data != nullptr) {
        dest.data = new Data(src.row, src.col);
        dest.start = src.start, dest.row = src.row, dest.col = src.col; dest.data->ref_cnt = 1;
        for (size_t r = 0; r < dest.row; r++) {
            size_t head = dest.start + r * dest.data -> col;
            for (size_t c = 0; c < dest.col; c++)
                dest.data->entry[head + c] = src.data->entry[head + c];
        }
        //memcpy(dest.data->entry, src.data->entry, sizeof(int) * src.row * src.col);
    }
}
bool equal_matrix(Matrix& a, Matrix& b) {
    if (a.row != b.row || a.col != b.col) return false;
    if (a.data == nullptr || b.data == nullptr) return false;
    for (size_t r = 0; r < a.row; r++) {
        size_t head1 = a.start + r * a.data -> col;
        size_t head2 = b.start + r * b.data -> col;
        for (size_t c = 0; c < b.col; c++)
            if (a.data->entry[head1 + c] != b.data->entry[head2 + c]) return false;
    }
    return true;
}
void add_matrix(Matrix& dest, Matrix& a, Matrix& b) {
    unload_data(dest);
    dest.data = new Data(a.row, b.col);
    dest.start = 0, dest.row = a.row, dest.col = b.col; dest.data->ref_cnt = 1;
    for (size_t r = 0; r < a.row; r++) {
        size_t head1 = a.start + r * a.data -> col;
        size_t head2 = b.start + r * b.data -> col;
        size_t head = dest.start + r * dest.data -> col;
        for (size_t c = 0; c < b.col; c++)
            dest.data->entry[head + c] = a.data->entry[head1 + c] + b.data->entry[head2 + c];
    }
}

void minus_matrix(Matrix& dest, Matrix& a, Matrix& b) {
    unload_data(dest);
    dest.data = new Data(a.row, b.col);
    dest.start = 0, dest.row = a.row, dest.col = b.col; dest.data->ref_cnt = 1;
    for (size_t r = 0; r < a.row; r++) {
        size_t head1 = a.start + r * a.data -> col;
        size_t head2 = b.start + r * b.data -> col;
        size_t head = dest.start + r * dest.data -> col;
        for (size_t c = 0; c < b.col; c++)
            dest.data->entry[head + c] = a.data->entry[head1 + c] - b.data->entry[head2 + c];
    }
}

void multiply_matrix(Matrix& dest, Matrix& a, Matrix& b) {
    unload_data(dest);
    dest.data = new Data(a.row, b.col);
    dest.start = 0, dest.row = a.row, dest.col = b.col; dest.data->ref_cnt = 1;
    for (size_t i = 0; i < a.row; i++) {
        for (size_t j = 0; j < b.col; j++) {
            int sum = 0;
            for (size_t k = 0; k < a.col; k++)
                sum += a.data->entry[a.start + i * a.data->col + k] * b.data->entry[b.start + k * b.data->col + j];
            dest.data->entry[dest.start + i * dest.col + j] = sum;
        }
    }
}
int main()
{
    // Sample code on how to use your library
    Data *da = new Data(3, 2), *db = new Data(2, 3);
    for (size_t i = 0; i < 6; i++)
        da->entry[i] = db->entry[i] = i;

    Matrix a, b, c;
    load_data(a, da, 0, 2, 2);  // the ROI is the whole matrix
    load_data(b, db, 0, 2, 2);
    print_matrix(a);
    /*
        0 1 
        2 3 
        4 5 
    */
    print_matrix(b);
    /*
        0 1 2 
        3 4 5
    */
    multiply_matrix(c, a, b);
    print_matrix(c);
    /*
        3 4 5 
        9 14 19 
        15 24 33
    */

    Matrix d, e, f;
    shallow_copy(d, c); // d, c -> (the same) data
    deep_copy(e, c);    // e->data (that have the exactly same content with) c->data
                        // but their addresses are different and ref_cnts are possibly
    load_data(f, c.data, 1, 3, 2);
    print_matrix(f);
    /*
        4 5 
        14 19 
        24 33
    */
    add_matrix(b, a, f);   // notice that the original b.data->ref_cnt becomes 0 and should be deleted
    print_matrix(b);
    /*
        4 6 
        16 22 
        28 38
    */

    std::cout << a.data->ref_cnt << ' ' << b.data->ref_cnt << ' '
        << c.data->ref_cnt << ' ' << d.data->ref_cnt << ' '
        << e.data->ref_cnt << ' ' << f.data->ref_cnt << std::endl;
    /*
        1 1 3 3 1 3
    */
    return 0;
}
//APPEND END