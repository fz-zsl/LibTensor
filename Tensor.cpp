#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "Tensor.hpp"

// Part 1: creation and initialization

// Part 2: tensor operations

// Part 3: Mathemathical Operations

template <typename T>
ts::Tensor<T> ts::operator + (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] + src2.data[i];
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator + (ts::Tensor<T> src1, T src2) {
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] + src2;
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator - (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] - src2.data[i];
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator - (ts::Tensor<T> src1, T src2) {
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] - src2;
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator * (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim - 2; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	if (src1.shape[src1.dim - 1] != src2.shape[src2.dim - 2] || src1.shape[src1.dim - 2] != src2.shape[src2.dim - 1]) throw std::invalid_argument("step cannot be zero.");
	int *tmp_shape = new int[src1.dim];
	for (int i = 0; i < src1.dim - 1; ++i) tmp_shape[i] = src1.shape[i]; tmp_shape[src1.dim - 1] = src2.shape[src2.dim - 1];
	ts::Tensor<T> result(src1.dim, tmp_shape);
	int base = src1.shape[src1.dim - 2] * src2.shape[src2.dim - 1];
	int size = 1, row = src1.shape[src1.dim - 2], col = src2.shape[src2.dim - 1];
	int base1 = src1.shape[src1.dim - 1] * src1.shape[src1.dim - 2];
	int base2 = src2.shape[src2.dim - 1] * src2.shape[src2.dim - 2];
	for (int i = 0; i < src1.dim - 2; ++i) size = size * src1.shape[i];
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < row; ++j) {
			for (int k = 0; k < col; ++k) {
				for (int l = 0; l < src1.shape[src1.dim - 1]; ++l)
				result[i * base + j * col + k] += src1.data[i * base1 + j * src1.shape[src1.dim - 1] + l] * src2.data[i * base2 + l * src2.shape[src2.dim - 1] + k];
			}
		}
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator * (ts::Tensor<T> src1, T src2) {
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] * src2;
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator / (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] / src2.data[i];
	return result;
}

template <typename T>
ts::Tensor<T> ts::operator / (ts::Tensor<T> src1, T src2) {
	ts::Tensor<T> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = src1.data[i] / src2;
	return result;
}

template <typename T>
ts::Tensor<T> ts::log(ts::Tensor<T> src) {
	ts::Tensor<T> result(src.dim, src.shape);
	int size = 1;
	for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
	for (int i = 0; i < size; ++i) result.data[i] = log(src.data[i]);
	return result;
}

template <typename T>
ts::Tensor<T> ts::sum(ts::Tensor<T> src, int dim) {
	if (dim == 0) {
		int size = 1;
		for (int i = 1; i < src.dim; ++i) {
			size *= src.shape[i];
		}
		int *tmp_shape = new int[size];
		for (int i = 1; i < src.dim; ++i) {
			tmp_shape[i - 1] = src.shape[i];
		}
		ts::Tensor<T> result(src.dim - 1, tmp_shape);
		for (int i = 0; i < size; ++i) {
			T cur = (T)0;
			for (int j = 0; j < src.shape[0]; ++j) cur += src.data[i + j * size];
			result.data[i] = cur;
		}
		return result;
	} else 
	if (dim == src.dim) {
		int size = 1;
		for (int i = 0; i < src.dim - 1; ++i) {
			size *= src.shape[i];
		}
		int *tmp_shape = new int[size];
		for (int i = 0; i < src.dim - 1; ++i) {
			tmp_shape[i] = src.shape[i];
		}
		ts::Tensor<T> result(src.dim - 1, tmp_shape);
		for (int i = 0; i < size; ++i) {
			T cur = (T)0;
			for (int j = 0; j < src.shape[src.dim - 1]; ++j) cur += src.data[i * src.shape[src.dim - 1] + j];
			result.data[i] = cur;
		}
		return result;
	} else {
		int suf_size = 1, size = 1, pre_size = 1;
		for (int i = dim + 1; i < src.dim; ++i) suf_size = suf_size * src.shape[i];
		for (int i = 0; i < dim; ++i) pre_size = pre_size * src.shape[i];
		for (int i = 0; i < src.dim; ++i) 
			if (i != dim) size = size * src.shape[i];
		int *tmp_shape = new int[size];
		ts::Tensor<T> result(src.dim - 1, tmp_shape);
		for (int i = 0; i < pre_size; ++i) {
			for (int j = 0; j < suf_size; ++j) {
				for (int k = 0; k < src.shape[dim]; ++k) 
				result.data[i * suf_size + j] += src.data[i * suf_size * src.shape[dim] + k * suf_size + j];
			}
		}
		return result;
	}
}

template <typename T>
T ts::mean(ts::Tensor<T> src) {
	int size = 1;
	for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
	T cur = (T)0;
	for (int i = 0; i < size; ++i) cur = cur + src.data[i];
	return cur / size;
}

template <typename T>
T ts::min(ts::Tensor<T> src) {
	int size = 1;
	for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
	T cur = src.shape[0];
	for (int i = 1; i < size; ++i) cur = min(src.data[i], cur);
	return cur;
}

template <typename T>
T ts::max(ts::Tensor<T> src) {
	int size = 1;
	for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
	T cur = src.shape[0];
	for (int i = 1; i < size; ++i) cur = max(src.data[i], cur);
	return cur;
}

template <typename T>
ts::Tensor<bool> ts::operator == (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] == src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}

template <typename T>
ts::Tensor<bool> ts::operator != (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] != src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}

template <typename T>
ts::Tensor<bool> ts::operator > (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] > src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}

template <typename T>
ts::Tensor<bool> ts::operator >= (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] >= src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}

template <typename T>
ts::Tensor<bool> ts::operator < (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] < src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}

template <typename T>
ts::Tensor<bool> ts::operator <= (ts::Tensor<T> src1, ts::Tensor<T> src2) {
	if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
	for (int i = 0; i < src1.dim; ++i) 
		if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
	ts::Tensor<bool> result(src1.dim, src1.shape);
	int size = 1;
	for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
	for (int i = 0; i < size; ++i) if (src1.data[i] <= src2.data[i]) result.data[i] = true; else result.data[i] = false;
	return result;
}