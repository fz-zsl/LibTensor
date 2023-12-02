#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "Tensor.h"

template <typename T>
ts::Tensor<T>::Tensor(int src_dim, int src_shape[]) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	dim = src_dim;
	shape = new int[dim];
	int size = 1;
	for (int i = 0; i < dim; ++i) {
		shape[i] = src_shape[i];
		size *= shape[i];
	}
	data = new T[size];
	for (int i = 0; i < size; ++i) data[i] = (T)0;
}

template <typename T>
ts::Tensor<T>::Tensor(int src_dim, int src_shape[], T *src_data) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	if (src_data == nullptr) {
		throw std::invalid_argument("src_data is a NULL pointer.");
	}
	dim = src_dim;
	shape = new int[dim];
	int size = 1;
	for (int i = 0; i < dim; ++i) {
		shape[i] = src_shape[i];
		size *= shape[i];
	}
	data = new T[size];
	for (int i = 0; i < size; ++i) {
		data[i] = src_data[i];
	}
}

template <typename T>
ts::Tensor<T>::~Tensor() {
	dim = -1;
	delete[] shape;
	delete[] data;
}

template <typename T>
T ts::Tensor<T>::getVal(int idx[]) {
	int pos = 0, size = 1;
	for (int i = dim - 1; i >= 0; --i) {
		pos += (idx[i] % shape[i]) * size;
	}
	return data[pos];
}

template <typename T>
ts::Tensor<T> ts::Tensor<T>::slice(std::pair<int,int> range[]) {
	return ts::slice(*this, range);
}

template <typename T>
ts::Tensor<T> ts::Tensor<T>::tile(int reps[]) {
	return ts::tile(*this, reps);
}

template <typename T>
void ts::Tensor<T>::modify(std::pair<int,int> range[], Tensor<T> val) {
	return ts::modify(*this, range, val);
}

template <typename T>
ts::Tensor<T> ts::Tensor<T>::transpose(int src_dim1, int src_dim2) {
	return ts::transpose(*this, src_dim1, src_dim2);
}

template <typename T>
ts::Tensor<T> ts::Tensor<T>::permute(int src_order[]) {
	return ts::permute(*this, src_order);
}

template <typename T>
void ts::Tensor<T>::print(int newDim, int newShape[]) {
	return ts::print(*this, newDim, newShape);
}

template <typename T>
void ts::Tensor<T>::print() {
	return ts::print(*this, this->dim, this->shape);
}

// Part 1: creation and initialization

template <typename T>
ts::Tensor<T> ts::zeros(int src_dim, int src_shape[]) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	ts::Tensor<T> result(src_dim, src_shape);
	int size = 1;
	for (int i = 0; i < src_dim; ++i) {
		size *= src_shape[i];
	}
	for (int i = 0; i < size; ++i) {
		result.data[i] = 0;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::zeros_like(ts::Tensor<T> src) {
	return ts::zeros(src.dim, src.shape);
}

template <typename T>
ts::Tensor<T> ts::ones(int src_dim, int src_shape[]) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	ts::Tensor<T> result(src_dim, src_shape);
	int size = 1;
	for (int i = 0; i < src_dim; ++i) {
		size *= src_shape[i];
	}
	for (int i = 0; i < size; ++i) {
		result.data[i] = 1;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::ones_like(ts::Tensor<T> src) {
	return ts::ones(src.dim, src.shape);
}

template <typename T>
ts::Tensor<T> ts::eye(int src_dim) {
	if (src_dim != 2) {
		throw std::invalid_argument("src_dim must be 2.");
	}
	ts::Tensor<T> result(2, new int[2]{src_dim, src_dim});
	for (int i = 0; i < src_dim; ++i) {
		result.data[i * src_dim + i] = 1;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::rand(int src_dim, int src_shape[]) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	ts::Tensor<T> result(src_dim, src_shape);
	int size = 1;
	srand(time(NULL));
	for (int i = 0; i < src_dim; ++i) {
		size *= src_shape[i];
	}
	for (int i = 0; i < size; ++i) {
		result.data[i] = std::rand() / (T)RAND_MAX;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::rand_like(ts::Tensor<T> src) {
	return ts::rand(src.dim, src.shape);
}

template <typename T>
ts::Tensor<T> ts::full(int src_dim, int src_shape[], T src_val) {
	if (src_dim <= 0) {
		throw std::invalid_argument("Non-positive dimention.");
	}
	if (src_shape == nullptr) {
		throw std::invalid_argument("src_shape is a NULL pointer.");
	}
	ts::Tensor<T> result(src_dim, src_shape);
	int size = 1;
	for (int i = 0; i < src_dim; ++i) {
		size *= src_shape[i];
	}
	for (int i = 0; i < size; ++i) {
		result.data[i] = src_val;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::full_like(ts::Tensor<T> src, T src_val) {
	return ts::full(src.dim, src.shape, src_val);
}

template <typename T>
ts::Tensor<T> ts::excrange(T start, T end, T step) {
	if (step == 0) {
		throw std::invalid_argument("step cannot be zero.");
	}
	int size = ceil((double)(end - start) / step);
	ts::Tensor<T> result(1, new int[1]{size});
	for (int i = 0; i < size; ++i) {
		result.data[i] = start + i * step;
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::incrange(T start, T end, T step) {
	if (step == 0) {
		throw std::invalid_argument("step cannot be zero.");
	}
	int size = floor((double)(end - start) / step) + 1;
	ts::Tensor<T> result(1, new int[1]{size});
	for (int i = 0; i < size; ++i) {
		result.data[i] = start + i * step;
	}
	return result;
}

// Part 2: tensor operations

template <typename T>
ts::Tensor<T> ts::slice(ts::Tensor<T> src, std::pair<int,int> range[]) {
	if (range == nullptr) {
		throw std::invalid_argument("range is a NULL pointer.");
	}
	for (int i = 0; i < src.dim; ++i) {
		if (range[i].second == -1) {
			// -1 as a label of ignoring this dimension
			range[i].second = range[i].first + 1;
		}
		if (range[i].first < 0 || range[i].first >= src.shape[i]) {
			char message[50];
			sprintf(message, "Left bound of dimension %d out of range.", i);
			throw std::invalid_argument(message);
		}
		if (range[i].second <= range[i].first || range[i].second > src.shape[i]) {
			char message[50];
			sprintf(message, "Right bound of dimension %d out of range.", i);
			throw std::invalid_argument(message);
		}
	}
	int *oldIdx = new int[src.dim];
	int newDim = 0;
	for (int i = 0; i < src.dim; ++i) {
		oldIdx[i] = range[i].first;
		if (range[i].second - range[i].first > 1) {
			++newDim;
		}
	}
	int *newShape = new int[newDim];
	newDim = 0;
	for (int i = 0; i < src.dim; ++i) {
		if (range[i].second - range[i].first > 1) {
			newShape[newDim] = range[i].second - range[i].first;
			++newDim;
		}
	}
	ts::Tensor<T> result(newDim, newShape);
	delete[] newShape;
	int newPos = 0;
	while (true) {
		int oldPos = 0, weight = 1;
		for (int i = src.dim - 1; i >= 0; --i) {
			oldPos += oldIdx[i] * weight;
			weight *= src.shape[i];
		}
		result.data[newPos] = src.data[oldPos];
		++newPos;
		++oldIdx[src.dim - 1];
		for (int i = src.dim - 1; i > 0; ++i) {
			if (oldIdx[i] == range[i].second) {
				oldIdx[i] = range[i].first;
				++oldIdx[i - 1];
			}
		}
		if (oldIdx[0] == range[0].second) {
			break;
		}
	}
	delete[] oldIdx;
	return result;
}

template <typename T>
ts::Tensor<T> ts::concat(ts::Tensor<T> src1, ts::Tensor<T> src2, int dim) {
	if (src1.dim != src2.dim) {
		throw std::invalid_argument("Unconcatenatable sources (different dimension).");
	}
	for (int i = 0; i < src1.dim; ++i) {
		if (i != dim && src1.shape[i] != src2.shape[i]) {
			throw std::invalid_argument("Unconcatenatable sources (shape error).");
		}
	}
	int *newShape = new int[src1.dim];
	for (int i = 0; i < src1.dim; ++i) {
		if (i != dim) {
			newShape[i] = src1.shape[i];
		}
		else {
			newShape[i] = src1.shape[i] + src2.shape[i];
		}
	}
	ts::Tensor<T> result(src1.dim, newShape);
	int weight1 = 1, weight2 = 1;
	for (int i = src1.dim - 1; i > dim; --i) {
		weight1 *= src1.shape[i];
	}
	weight1 *= src1.shape[dim];
	weight2 *= src2.shape[dim];
	int blkCnt = 1;
	for (int i = dim - 1; i >= 0; --i) {
		blkCnt *= src1.shape[i];
	}
	for (int i = 0; i < blkCnt; ++i) {
		for (int j = 0; j < weight1; ++j) {
			result.data[i * (weight1 + weight2) + j] = src1.data[i * weight1 + j];
		}
		for (int j = 0; j < weight1; ++j) {
			result.data[i * (weight1 + weight2) + weight1 + j] = src2.data[i * weight2 + j];
		}
	}
	return result;
}

template <typename T>
ts::Tensor<T> tile(ts::Tensor<T> src, int reps[]) {
	for (int i = 0; i <= src.dim; ++i) {
		if (reps[i] <= 0) {
			throw std::invalid_argument("Non-positive reps.");
		}
	}
	int *newShape = new int[src.dim];
	int size = 1;
	for (int i = 0; i < src.dim; ++i) {
		newShape[i] = src.shape[i] * reps[i];
		size *= newShape[i];
	}
	ts::Tensor<T> result(src.dim, newShape);
	int *newIdx = new int[src.dim];
	for (int i = 0; i < size; ++i) {
		result.data[i] = src.getVal(newIdx);
		++newIdx[src.dim - 1];
		for (int j = src.dim - 1; j > 0; --j) {
			if (newIdx[j] == newShape[j]) {
				newIdx[j] = 0;
				++newIdx[j - 1];
			}
			else {
				break;
			}
		}
	}
	delete[] newIdx;
	delete[] newShape;
	return result;
}

template <typename T>
void ts::modify(ts::Tensor<T> src, std::pair<int,int> range[], ts::Tensor<T> val) {
	if (range == nullptr) {
		throw std::invalid_argument("range is a NULL pointer.");
	}
	for (int i = 0; i < src.dim; ++i) {
		if (range[i].second == -1) {
			// -1 as a label of ignoring this dimension
			range[i].second = range[i].first + 1;
		}
		if (range[i].first < 0 || range[i].first >= src.shape[i]) {
			char message[50];
			sprintf(message, "Left bound of dimension %d out of range.", i);
			throw std::invalid_argument(message);
		}
		if (range[i].second <= range[i].first || range[i].second > src.shape[i]) {
			char message[50];
			sprintf(message, "Right bound of dimension %d out of range.", i);
			throw std::invalid_argument(message);
		}
	}
	int *oldIdx = new int[src.dim];
	int newDim = 0;
	for (int i = 0; i < src.dim; ++i) {
		oldIdx[i] = range[i].first;
		if (range[i].second - range[i].first > 1) {
			++newDim;
		}
	}
	if (newDim != val.dim) {
		throw std::runtime_error("Fail to modify (different dimention).");
	}
	int *newShape = new int[newDim];
	newDim = 0;
	int newSize = 1;
	for (int i = 0; i < src.dim; ++i) {
		if (range[i].second - range[i].first > 1) {
			newShape[newDim] = range[i].second - range[i].first;
			newSize *= newShape[i];
			++newDim;
		}
	}
	for (int i = 0; i < newDim; ++i) {
		if (newShape[i] != val.dim[i]) {
			throw std::runtime_error("Fail to modify (different shape).");
		}
	}
	int* *roiPtr = new int*[newSize];
	int newPos = 0;
	for (int j = 0; j < newSize; ++j) {
		int oldPos = 0, weight = 1;
		for (int i = src.dim - 1; i >= 0; --i) {
			oldPos += oldIdx[i] * weight;
			weight *= src.shape[i];
		}
		roiPtr[newPos] = &src.data[oldPos];
		++newPos;
		++oldIdx[src.dim - 1];
		for (int i = src.dim - 1; i > 0; ++i) {
			if (oldIdx[i] == range[i].second) {
				oldIdx[i] = range[i].first;
				++oldIdx[i - 1];
			}
		}
	}
	for (int i = 0; i < newSize; ++i) {
		*roiPtr[i] = val.data[i];
	}
	delete[] oldIdx;
	delete[] newShape;
	delete[] roiPtr;
	return;
}

template <typename T>
ts::Tensor<T> ts::transpose(ts::Tensor<T> src, int src_dim1, int src_dim2) {
	if (src_dim1 < 0 || src_dim1 >= src.dim) {
		throw std::invalid_argument("src_dim1 is out of range.");
	}
	if (src_dim2 < 0 || src_dim2 >= src.dim) {
		throw std::invalid_argument("src_dim2 is out of range.");
	}
	ts::Tensor<T> result(src.dim, src.shape);
	swap(result.shape[src_dim1], result.shape[src_dim2]);
	int size = 1, mod1 = 1, mod2 = 1;
	for (int i = src.dim - 1; i >= 0; --i) {
		if (i == src_dim1) {
			mod1 = size;
		}
		if (i == src_dim2) {
			mod2 = size;
		}
		size *= result.shape[i];
	}
	for (int i = 0; i < size; ++i) {
		int idx1 = (i / mod1) % result.shape[src_dim1];
		int idx2 = (i / mod2) % result.shape[src_dim2];
		result.data[
			i - idx1 * mod1 - idx2 * mod2 + idx1 * mod2 + idx2 * mod1
		] = src.data[i];
	}
	return result;
}

template <typename T>
ts::Tensor<T> ts::permute(ts::Tensor<T> src, int src_order[]) {
	if (src_order == nullptr) {
		throw std::invalid_argument("src_order is a NULL pointer.");
	}
	bool *mk = new bool[src.dim];
	for (int i = 0; i < src.dim; ++i) {
		mk[i] = false;
	}
	for (int i = 0; i < src.dim; ++i) {
		if (src_order[i] < 0 || src_order[i] >= src.dim) {
			throw std::invalid_argument("src_order is out of range.");
		}
		if (mk[src_order[i]]) {
			delete[] mk;
			throw std::invalid_argument("src_order is not a permutation.");
		}
	}
	delete[] mk;
	ts::Tensor<T> result(src.dim, src.shape);
	for (int i = 0; i < src.dim; ++i) {
		result.shape[i] = src.shape[src_order[i]];
	}
	int *oldIdx = new int[src.dim];
	int *weight = new int[src.dim];
	weight[src.dim - 1] = 1;
	for (int i = src.dim - 1; i >= 0; --i) {
		oldIdx[i] = 0;
		weight[i - 1] = weight[i] * result.shape[i];
	}
	oldIdx[0] = 0;
	int size = weight[0] * result.shape[0];
	for (int i = 0; i < size; ++i) {
		int pos = 0;
		for (int j = 0; j < src.dim; ++j) {
			pos += oldIdx[src_order[j]] * weight[j];
		}
		result.data[pos] = src.data[i];
		++oldIdx[src.dim - 1];
		for (int j = src.dim - 1; j > 0; --j) {
			if (oldIdx[j] == src.shape[j]) {
				oldIdx[j] = 0;
				++oldIdx[j - 1];
			}
			else {
				break;
			}
		}
	}
	delete[] oldIdx;
	delete[] weight;
	return result;
}

template <typename T>
void ts::print(ts::Tensor<T> src,int newDim, int newShape[]) {
	int oldSize = 1, newSize = 1;
	for (int i = 0; i < src.dim; ++i) {
		oldSize *= src.shape[i];
	}
	for (int i = 0; i < newDim; ++i) {
		newSize *= newShape[i];
	}
	if (oldSize != newSize) {
		throw std::runtime_error("Fail to print tensor (different size).");
	}
	newSize /= newShape[newDim - 1];
	int *newIdx = new int[newDim];
	for (int i = 0, j; i < newSize; ++i) {
		printf("%s", i ? "       " : "tensor(");
		for (j = src.dim - 2; j >= 0; --j) {
			if (newIdx[j] != 0) {
				break;
			}
		}
		for (int k = 0; k <= j; ++k) {
			printf(" ");
		}
		for (int k = j + 1; k < src.dim; ++k) {
			printf("[");
		}
		for (int k = 0; k < newShape[newDim - 1]; ++k) {
			printf("%.5g%s",
				(double)(src.data[i * newShape[newDim - 1] + k]),
				k == newShape[newDim - 1] - 1 ? "]" : ", "
			);
		}
		int carryCnt = 1;
		++newIdx[newDim - 2];
		for (int k = newDim - 2; k >= 0; --k) {
			if (newIdx[k] == newShape[k]) {
				newIdx[k] = 0;
				++newIdx[k - 1];
				++carryCnt;
			}
		}
		for (int k = 0; k < carryCnt; ++k) {
			printf("]");
		}
		if (i != newSize - 1) {
			printf(",");
			for (int k = 0; k < carryCnt; ++k) {
				puts("");
			}
		}
		else {
			puts(")");
		}
	}
	delete[] newIdx;
	return;
}

template <typename T>
void ts::print(ts::Tensor<T> src) {
	return ts::print(src, src.dim, src.shape);
}

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