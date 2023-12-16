#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <set>

#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define threadsPerBlock 256

// CUDA Kernels begin
template <typename T>
__global__ void addKernel(T* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] + src2[i];
	}
	return;
}

template <typename T>
__global__ void addKernel(T* result, T* src1, T src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] + src2;
	}
	return;
}

template <typename T>
__global__ void subKernel(T* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] - src2[i];
	}
	return;
}

template <typename T>
__global__ void subKernel(T* result, T* src1, T src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] - src2;
	}
	return;
}

template <typename T>
__global__ void mulKernel(T* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] * src2[i];
	}
	return;
}

template <typename T>
__global__ void mulKernel(T* result, T* src1, T src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] * src2;
	}
	return;
}

template <typename T>
__global__ void divKernel(T* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] / src2[i];
	}
	return;
}

template <typename T>
__global__ void divKernel(T* result, T* src1, T src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = src1[i] / src2;
	}
	return;
}

template <typename T>
__global__ void logKernel(T* result, T* src, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = std::log((double)src[i]);
	}
	return;
}

template <typename T>
__global__ void sumKernel(T* result, T* src, int size, int shape0, int shape1) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int idx0 = i / shape0 / shape1;
		// int idx1 = i / shape1 % shape0;
		int idx2 = i % shape1;
		atomicAdd(&result[idx0 * shape1 + idx2], src[i]);
	}
	return;
}

template <typename T>
__global__ void meanKernel(T* result, T* src, int size) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		sdata[tid] = src[i];
	}
	else {
		sdata[tid] = 0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
	if (!tid) result[blockIdx.x] = sdata[0];
	return;
}

template <typename T>
__global__ void minKernel(T* result, T* src, int size) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		sdata[tid] = src[i];
	}
	else {
		sdata[tid] = src[0];
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s && sdata[tid] > sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
	if (!tid) result[blockIdx.x] = sdata[0];
	return;
}

template <typename T>
__global__ void maxKernel(T* result, T* src, int size) {
	extern __shared__ T sdata[];
	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		sdata[tid] = src[i];
	}
	else {
		sdata[tid] = src[0];
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s && sdata[tid] < sdata[tid + s]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
	if (!tid) result[blockIdx.x] = sdata[0];
	return;
}

template <typename T>
__global__ void eqKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] == src2[i]);
	}
	return;
}

template <typename T>
__global__ void neKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] != src2[i]);
	}
	return;
}

template <typename T>
__global__ void gtKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] > src2[i]);
	}
	return;
}

template <typename T>
__global__ void geKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] >= src2[i]);
	}
	return;
}

template <typename T>
__global__ void ltKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] < src2[i]);
	}
	return;
}

template <typename T>
__global__ void leKernel(bool* result, T* src1, T* src2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		result[i] = (src1[i] <= src2[i]);
	}
	return;
}

// CUDA Kernels end

namespace ts {
	template <typename T>
	class Tensor {
		public:
			int dim;
			int *shape;
			T *data;
			std::string outputBuf;

		public:
			Tensor(int src_dim, int src_shape[]) {
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
				outputBuf = "";
			}

			Tensor(int src_dim, int src_shape[], T *src_data) {
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
				outputBuf = "";
			}

			Tensor(const Tensor<T>& src) {
				dim = src.dim;
				shape = new int[dim];
				int size = 1;
				for (int i = 0; i < dim; ++i) {
					shape[i] = src.shape[i];
					size *= shape[i];
				}
				data = new T[size];
				for (int i = 0; i < size; ++i) {
					data[i] = src.data[i];
				}
				outputBuf = "";
			}

			Tensor(FILE* input) {
				if (input == nullptr) {
					throw std::invalid_argument("input is a NULL pointer.");
				}
				// read the whole file
				std::vector<char> buffer;
				char ch;
				while ((ch = fgetc(input)) != EOF) {
					if (ch != '\n' && ch != '\r' && ch != '\t' && ch != ' ') {
						buffer.push_back(ch);
					}
				}
				buffer.push_back('\0');
				std::string str(buffer.begin(), buffer.end());
				if (str.substr(0, 7) != "tensor(") {
					throw std::invalid_argument("Please input a tensor.");
				}
				// get dim
				dim = 0;
				while (str[dim + 7] == '[') {
					++dim;
				}
				// get shape
				int cntComma = 0;
				shape = new int[dim]{};
				for (int i = 8; i < str.size(); ++i) {
					int conseq = 0;
					while (str[i - conseq] == ']') {
						++conseq;
					}
					if (!shape[conseq - 1]) {
						shape[conseq - 1] = cntComma + 1;
					}
					if (str[i] == ',') {
						++cntComma;
					}
				}
				int size = shape[dim - 1];
				for (int i = dim - 1; i; --i) {
					shape[i] /= shape[i - 1];
				}
				// get data
				int pos = 0;
				data = new T[size]{};
				for (int i = 0; i < size; ++i) {
					if (str[pos] < '0' || str[pos] > '9') {
						++pos;
						--i;
						continue;
					}
					data[i] = 0;
					while (str[pos] >= '0' && str[pos] <= '9') {
						data[i] = data[i] * 10 + str[pos] - '0';
						++pos;
					}
					if (str[pos] == '.') {
						++pos;
						double weight = 0.1;
						while (str[pos] >= '0' && str[pos] <= '9') {
							data[i] += weight * (str[pos] - '0');
							weight *= 0.1;
							++pos;
						}
					}
				}
				outputBuf = "";
			}

			Tensor<T> load(std::string filename) {
				FILE* input = fopen(filename.c_str(), "r");
				if (input == nullptr) {
					throw std::invalid_argument("Cannot open file.");
				}
				Tensor<T> result(input);
				fclose(input);
				return result;
			}

			~Tensor() {
				dim = -1;
				delete[] shape;
				delete[] data;
			}

			Tensor<T> size() {
				return Tensor<int>(1, new int[1]{dim}, shape);
			}

			std::string type() {
				return std::string(typeid(T).name()) + "\n";
			}

			std::string data_ptr() {
				char *addr = new char[20];
				sprintf(addr, "%p\n", data);
				std::string str(addr);
				delete[] addr;
				return str;
			}

			T getVal(int idx[]) {
				int pos = 0, size = 1;
				for (int i = dim - 1; i >= 0; --i) {
					pos += (idx[i] % shape[i]) * size;
				}
				return data[pos];
			}
			
			Tensor<T> slice(std::pair<int,int> range[]) {
				if (range == nullptr) {
					throw std::invalid_argument("range is a NULL pointer.");
				}
				for (int i = 0; i < this->dim; ++i) {
					if (range[i].second == -1) {
						// -1 as a label of ignoring this dimension
						range[i].second = range[i].first + 1;
					}
					if (range[i].first < 0 || range[i].first >= this->shape[i]) {
						char message[50] = "Left bound out of range.";
						//sprintf(message, "Left bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
					if (range[i].second <= range[i].first || range[i].second > this->shape[i]) {
						char message[50] = "Right bound out of range.";
						//sprintf(message, "Right bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
				}
				int *oldIdx = new int[this->dim];
				int newDim = 0;
				for (int i = 0; i < this->dim; ++i) {
					oldIdx[i] = range[i].first;
					if (range[i].second - range[i].first > 1) {
						++newDim;
					}
				}
				int *newShape = new int[newDim];
				newDim = 0;
				for (int i = 0; i < this->dim; ++i) {
					if (range[i].second - range[i].first > 1) {
						newShape[newDim] = range[i].second - range[i].first;
						++newDim;
					}
				}
				Tensor<T> result(newDim, newShape);
				delete[] newShape;
				int newPos = 0;
				while (true) {
					int oldPos = 0, weight = 1;
					for (int i = this->dim - 1; i >= 0; --i) {
						oldPos += oldIdx[i] * weight;
						weight *= this->shape[i];
					}
					result.data[newPos] = this->data[oldPos];
					++newPos;
					++oldIdx[this->dim - 1];
					for (int i = this->dim - 1; i > 0; --i) {
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

			Tensor<T> tile(int reps[]) {
				for (int i = 0; i < this->dim; ++i) {
					if (reps[i] <= 0) {
						throw std::invalid_argument("Non-positive reps.");
					}
				}
				int *newShape = new int[this->dim];
				int size = 1;
				for (int i = 0; i < this->dim; ++i) {
					newShape[i] = this->shape[i] * reps[i];
					size *= newShape[i];
				}
				Tensor<T> result(this->dim, newShape);
				int *newIdx = new int[this->dim];
				for (int i = 0; i < size; ++i) {
					result.data[i] = this->getVal(newIdx);
					++newIdx[this->dim - 1];
					for (int j = this->dim - 1; j > 0; --j) {
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

			Tensor<T> concat(Tensor<T> src2, int dim) {
				if (this->dim != src2.dim) {
					throw std::invalid_argument("Unconcatenatable sources (different dimension).");
				}
				for (int i = 0; i < this->dim; ++i) {
					if (i != dim && this->shape[i] != src2.shape[i]) {
						throw std::invalid_argument("Unconcatenatable sources (shape error).");
					}
				}
				int *newShape = new int[this->dim];
				for (int i = 0; i < this->dim; ++i) {
					if (i != dim) {
						newShape[i] = this->shape[i];
					}
					else {
						newShape[i] = this->shape[i] + src2.shape[i];
					}
				}
				Tensor<T> result(this->dim, newShape);
				int weight1 = 1, weight2 = 1;
				for (int i = this->dim - 1; i > dim; --i) {
					weight1 *= this->shape[i];
				}
				weight1 *= this->shape[dim];
				weight2 *= src2.shape[dim];
				int blkCnt = 1;
				for (int i = dim - 1; i >= 0; --i) {
					blkCnt *= this->shape[i];
				}
				for (int i = 0; i < blkCnt; ++i) {
					for (int j = 0; j < weight1; ++j) {
						result.data[i * (weight1 + weight2) + j] = this->data[i * weight1 + j];
					}
					for (int j = 0; j < weight1; ++j) {
						result.data[i * (weight1 + weight2) + weight1 + j] = src2.data[i * weight2 + j];
					}
				}
				return result;
			}

			void modify(std::pair<int,int> range[], Tensor<T> val) {
				if (range == nullptr) {
					throw std::invalid_argument("range is a NULL pointer.");
				}
				for (int i = 0; i < this->dim; ++i) {
					if (range[i].second == -1) {
						// -1 as a label of ignoring this dimension
						range[i].second = range[i].first + 1;
					}
					if (range[i].first < 0 || range[i].first >= this->shape[i]) {
						char message[50] = "Left bound out of range.";;
						//sprintf(message, "Left bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
					if (range[i].second <= range[i].first || range[i].second > this->shape[i]) {
						char message[50] = "Right bound out of range.";
						//sprintf(message, "Right bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
				}
				int *oldIdx = new int[this->dim];
				int newDim = 0;
				for (int i = 0; i < this->dim; ++i) {
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
				for (int i = 0; i < this->dim; ++i) {
					if (range[i].second - range[i].first > 1) {
						newShape[newDim] = range[i].second - range[i].first;
						newSize *= newShape[newDim];
						++newDim;
					}
				}
				for (int i = 0; i < newDim; ++i) {
					if (newShape[i] != val.shape[i]) {
						throw std::runtime_error("Fail to modify (different shape).");
					}
				}
				int* *roiPtr = new int*[newSize];
				int newPos = 0;
				for (int j = 0; j < newSize; ++j) {
					int oldPos = 0, weight = 1;
					for (int i = this->dim - 1; i >= 0; --i) {
						oldPos += oldIdx[i] * weight;
						weight *= this->shape[i];
					}
					roiPtr[newPos] = &this->data[oldPos];
					++newPos;
					++oldIdx[this->dim - 1];
					for (int i = this->dim - 1; i > 0; --i) {
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

			Tensor<T> permute(int src_order[]) {
				if (src_order == nullptr) {
					throw std::invalid_argument("src_order is a NULL pointer.");
				}
				bool *mk = new bool[this->dim];
				for (int i = 0; i < this->dim; ++i) {
					mk[i] = false;
				}
				for (int i = 0; i < this->dim; ++i) {
					if (src_order[i] < 0 || src_order[i] >= this->dim) {
						throw std::invalid_argument("src_order is out of range.");
					}
					if (mk[src_order[i]]) {
						delete[] mk;
						throw std::invalid_argument("src_order is not a permutation.");
					}
				}
				delete[] mk;
				Tensor<T> result(this->dim, this->shape);
				for (int i = 0; i < this->dim; ++i) {
					result.shape[i] = this->shape[src_order[i]];
				}
				int *oldIdx = new int[this->dim];
				int *weight = new int[this->dim];
				weight[this->dim - 1] = 1;
				for (int i = this->dim - 1; i >= 0; --i) {
					oldIdx[i] = 0;
					if (i > 0) {
						weight[i - 1] = weight[i] * result.shape[i];
					}
				}
				oldIdx[0] = 0;
				int size = weight[0] * result.shape[0];
				for (int i = 0; i < size; ++i) {
					int pos = 0;
					for (int j = 0; j < this->dim; ++j) {
						pos += oldIdx[src_order[j]] * weight[j];
					}
					result.data[pos] = this->data[i];
					++oldIdx[this->dim - 1];
					for (int j = this->dim - 1; j > 0; --j) {
						if (oldIdx[j] == this->shape[j]) {
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

			Tensor<T> transpose(int src_dim1, int src_dim2) {
				if (src_dim1 < 0 || src_dim1 >= this->dim) {
					throw std::invalid_argument("src_dim1 is out of range.");
				}
				if (src_dim2 < 0 || src_dim2 >= this->dim) {
					throw std::invalid_argument("src_dim2 is out of range.");
				}
				int *order = new int[this->dim];
				for (int i = 0; i < this->dim; ++i) {
					order[i] = i;
				}
				std::swap(order[src_dim1], order[src_dim2]);
				return this->permute(order);
			}

			std::ostream& appendBuffer(std::ostream& ost, std::string str) {
				outputBuf += str;
				if (outputBuf.size() > 1000000000) {
					ost << outputBuf;
					outputBuf = "";
				}
				return ost;
			}

			std::ostream& flushBuffer(std::ostream& ost) {
				ost << outputBuf;
				outputBuf = "";
				return ost;
			}

			std::ostream& print(int newDim, int newShape[], std::ostream& ost = std::cout, bool printShape = false) {
				int oldSize = 1, newSize = 1;
				static char tmp[100000000];
				for (int i = 0; i < this->dim; ++i) {
					oldSize *= this->shape[i];
				}
				for (int i = 0; i < newDim; ++i) {
					newSize *= newShape[i];
				}
				if (oldSize != newSize) {
					throw std::runtime_error("Fail to print tensor (different size).");
				}
				if (newDim == 1) {
					appendBuffer(ost, "tensor([");
					for (int i = 0; i < newShape[0]; ++i) {
						if (printShape && newShape[0] > 7 && i == 3) {
							appendBuffer(ost, " ... , ");
							i = newShape[0] - 3;
							continue;
						}
						if (typeid(T) == typeid(bool)) {
							sprintf(tmp, "%s%s", this->data[i] ? " True" : "False", i == newShape[0] - 1 ? "])" : ", ");
							appendBuffer(ost, tmp);
						} else {
							sprintf(tmp, "%5g%s", (double)(this->data[i]), i == newShape[0] - 1 ? "])" : ", ");
							appendBuffer(ost, tmp);
						}
					}
					appendBuffer(ost, "\n");
					flushBuffer(ost);
					return ost;
				}
				newSize /= newShape[newDim - 1];
				int *newIdx = new int[newDim];
				for (int i = 0; i < newDim; ++i) {
					newIdx[i] = 0;
				}
				std::string cdots = "\n       ";
				bool needCdots = false;
				for (int i = 0; i <= dim; ++i) {
					cdots += " ";
				}
				cdots += " ...  \n\n";
				for (int i = 0, j; i < newSize; ++i) {
					sprintf(tmp, "%s", i ? "       " : "tensor(");
					appendBuffer(ost, tmp);
					for (j = this->dim - 2; j >= 0; --j) {
						if (newIdx[j] != 0) {
							break;
						}
					}
					for (int k = 0; k <= j; ++k) {
						appendBuffer(ost, " ");
					}
					for (int k = j + 1; k < this->dim; ++k) {
						appendBuffer(ost, "[");
					}
					for (int k = 0; k < newShape[newDim - 1]; ++k) {
						if (printShape && newShape[newDim - 1] > 7 && k == 3) {
							appendBuffer(ost, " ... , ");
							k = newShape[newDim - 1] - 3;
							continue;
						}
						if (typeid(T) == typeid(bool)) {
							sprintf(tmp, "%s%s",
								this->data[i * newShape[newDim - 1] + k] ? " True" : "False",
								k == newShape[newDim - 1] - 1 ? "]" : ", "
							);
							appendBuffer(ost, tmp);
						} else {
							sprintf(tmp, "%5g%s",
								(double)(this->data[i * newShape[newDim - 1] + k]),
								k == newShape[newDim - 1] - 1 ? "]" : ", "
							);
							appendBuffer(ost, tmp);
						}
					}
					int carryCnt = 1;
					if (printShape && newShape[newDim - 2] > 7 && newIdx[newDim - 2] == 2) {
						i += newShape[newDim - 2] - 5;
						newIdx[newDim - 2] = newShape[newDim - 2] - 2;
						// printf("%d\n", newShape[newDim - 1]);
						needCdots = true;
					}
					else {
						++newIdx[newDim - 2];
					}
					for (int k = newDim - 2; k > 0; --k) {
						if (newIdx[k] == newShape[k]) {
							newIdx[k] = 0;
							if (printShape && newShape[k - 1] > 5 && newIdx[k - 1] == 2) {
								needCdots = true;
								newIdx[k - 1] = newShape[k - 1] - 2;
							}
							else {
								++newIdx[k - 1];
							}
							++carryCnt;
						}
					}
					for (int k = 1; k < carryCnt; ++k) {
						appendBuffer(ost, "]");
					}
					if (i != newSize - 1) {
						appendBuffer(ost, ",");
						for (int k = 0; k < carryCnt; ++k) {
							appendBuffer(ost, "\n");
						}
					}
					else {
						appendBuffer(ost, "])\n");
					}
					if (needCdots) {
						appendBuffer(ost, cdots);
						needCdots = false;
					}
				}
				delete[] newIdx;
				flushBuffer(ost);
				return ost;
			}
			
			std::ostream& print(std::ostream& ost = std::cout, bool printShape = false) {
				return this->print(this->dim, this->shape, ost, printShape);
			}

			void save(std::string filename) {
				std::ofstream fout(filename, std::ios::out);
				if (!fout) {
					throw std::invalid_argument("Cannot open file.");
				}
				std::ostream& ost = fout;
				print(ost, false);
				return;
			}

			friend std::ostream& operator<<(std::ostream& ost, Tensor<T> src) {
				return src.print(ost, false);
			}

			Tensor<T> add(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				addKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_src, data_ths, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> add(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				addKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> sub(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				subKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_src, data_ths, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> sub(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				subKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> mul(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				mulKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_src, data_ths, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> mul(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				mulKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> Div(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				divKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_src, data_ths, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> Div(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				divKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> Log() {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				logKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<T> sum(int dim) {
				if (dim < 0 || dim >= this->dim) {
					throw std::invalid_argument("Error!");
				}
				int shape0 = 1, shape1 = this->shape[dim], shape2 = 1, size = 1;
				for (int i = dim - 1; i > dim; --i) shape0 *= this->shape[i];
				shape2 = shape0;
				for (int i = dim; ~i; --i) shape0 *= this->shape[i];
				size = shape0;
				shape0 = shape0 / shape1 / shape2;
				int *newShape = new int[this->dim - 1];
				for (int i = 0; i < dim; ++i) newShape[i] = this->shape[i];
				for (int i = dim + 1; i < this->dim; ++i) newShape[i - 1] = this->shape[i];
				Tensor<T> result(this->dim - 1, newShape);
				delete[] newShape;

				T *data_ths, *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size / shape1 * sizeof(T)));

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				sumKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, size, shape1, shape2);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size / shape1 * sizeof(T), cudaMemcpyDeviceToHost));

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));

				return result;
			}

			T mean() {
				int size = 1; 
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res, *partial_sum;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));

				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				size_t sharedMemSize = threadsPerBlock * sizeof(T);
				CUDA_CALL(cudaMalloc(&data_res, blocksPerGrid * sizeof(T)));
				meanKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_res, data_ths, size);
				CUDA_CALL(cudaDeviceSynchronize());
				partial_sum = new T[blocksPerGrid];
				CUDA_CALL(cudaMemcpy(partial_sum, data_res, blocksPerGrid * sizeof(T), cudaMemcpyDeviceToHost));
				T total_sum = (T)0;
				for (int i = 0; i < blocksPerGrid; ++i) {
					total_sum += partial_sum[i];
				}

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				delete[] partial_sum;
				
				return total_sum / size;
			}

			T Min() {
				int size = 1; 
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res, *block_ans;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));

				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				size_t sharedMemSize = threadsPerBlock * sizeof(T);
				CUDA_CALL(cudaMalloc(&data_res, blocksPerGrid * sizeof(T)));
				minKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_res, data_ths, size);
				CUDA_CALL(cudaDeviceSynchronize());
				block_ans = new T[blocksPerGrid];
				CUDA_CALL(cudaMemcpy(block_ans, data_res, blocksPerGrid * sizeof(T), cudaMemcpyDeviceToHost));
				T ans = block_ans[0];
				for (int i = 0; i < blocksPerGrid; ++i) {
					ans = std::min(ans, block_ans[i]);
				}

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				delete[] block_ans;
				
				return ans;
			}

			T Max() {
				int size = 1; 
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];

				T *data_ths, *data_res, *block_ans;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));

				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				size_t sharedMemSize = threadsPerBlock * sizeof(T);
				CUDA_CALL(cudaMalloc(&data_res, blocksPerGrid * sizeof(T)));
				maxKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_res, data_ths, size);
				CUDA_CALL(cudaDeviceSynchronize());
				block_ans = new T[blocksPerGrid];
				CUDA_CALL(cudaMemcpy(block_ans, data_res, blocksPerGrid * sizeof(T), cudaMemcpyDeviceToHost));
				T ans = block_ans[0];
				for (int i = 0; i < blocksPerGrid; ++i) {
					ans = std::max(ans, block_ans[i]);
				}

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_res));
				delete[] block_ans;
				
				return ans;
			}

			Tensor<bool> eq(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				eqKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<bool> ne(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				neKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<bool> gt(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				gtKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<bool> ge(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				geKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<bool> lt(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				ltKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor<bool> le(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("Error!");
				}
				for (int i = 0; i < src.dim; ++i) {
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("Error!");
					}
				}
				Tensor<bool> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];

				T *data_ths, *data_src;
				bool *data_res;
				CUDA_CALL(cudaMalloc(&data_ths, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_src, size * sizeof(T)));
				CUDA_CALL(cudaMalloc(&data_res, size)); // sizeof(bool) = 1

				CUDA_CALL(cudaMemcpy(data_ths, this->data, size * sizeof(T), cudaMemcpyHostToDevice));
				CUDA_CALL(cudaMemcpy(data_src, src.data, size * sizeof(T), cudaMemcpyHostToDevice));
				
				int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
				leKernel<<<blocksPerGrid, threadsPerBlock>>>(data_res, data_ths, data_src, size);
				CUDA_CALL(cudaGetLastError());

				CUDA_CALL(cudaMemcpy(result.data, data_res, size, cudaMemcpyDeviceToHost)); // sizeof(bool) = 1

				CUDA_CALL(cudaFree(data_ths));
				CUDA_CALL(cudaFree(data_src));
				CUDA_CALL(cudaFree(data_res));
				
				return result;
			}

			Tensor& operator=(const Tensor<T>& src) {
				if (this == &src) {
					return *this;
				}
				if (this->dim != src.dim) {
					throw std::runtime_error("Fail to assign (different dimension).");
				}
				int size = 1;
				for (int i = 0; i < this->dim; ++i) {
					if (this->shape[i] != src.shape[i]) {
						throw std::runtime_error("Fail to assign (different shape).");
					}
					size *= this->shape[i];
				}
				for (int i = 0; i < size; ++i) {
					this->data[i] = src.data[i];
				}
				return *this;
			}
	};

// Part 1: creation and initialization

	template <typename T>
	Tensor<T> zeros(int src_dim, int src_shape[]) {
		// initialize with zeros with given shape
		if (src_dim <= 0) {
			throw std::invalid_argument("Non-positive dimention.");
		}
		if (src_shape == nullptr) {
			throw std::invalid_argument("src_shape is a NULL pointer.");
		}
		Tensor<T> result(src_dim, src_shape);
		int size = 1;
		for (int i = 0; i < src_dim; ++i) {
			size *= src_shape[i];
		}
		for (int i = 0; i < size; ++i) {
			result.data[i] = 0;
		}
		return result;
	}

	template <typename T, typename G>
	Tensor<T> zeros_like(Tensor<G> src) {
		// initialize with zeros with the same shape as src
		return zeros<T>(src.dim, src.shape);
	}

	template <typename T>
	Tensor<T> ones(int src_dim, int src_shape[]) {
		// initialize with ones with given shape
		if (src_dim <= 0) {
			throw std::invalid_argument("Non-positive dimention.");
		}
		if (src_shape == nullptr) {
			throw std::invalid_argument("src_shape is a NULL pointer.");
		}
		Tensor<T> result(src_dim, src_shape);
		int size = 1;
		for (int i = 0; i < src_dim; ++i) {
			size *= src_shape[i];
		}
		for (int i = 0; i < size; ++i) {
			result.data[i] = 1;
		}
		return result;
	}

	template <typename T, typename G>
	Tensor<T> ones_like(Tensor<G> src) {
		// initialize with ones with the same shape as src
		return ones<T>(src.dim, src.shape);
	}

	template <typename T>
	Tensor<T> eye(int shape) {
		// initialize a 2D tensor with shape * shape size
		Tensor<T> result(2, new int[2]{shape, shape});
		for (int i = 0; i < shape; ++i) {
			result.data[i * shape + i] = 1;
		}
		return result;
	}

	template <typename T>
	Tensor<T> rand(int src_dim, int src_shape[]) {
		// randomly initialize the tensor with given shape
		if (src_dim <= 0) {
			throw std::invalid_argument("Non-positive dimention.");
		}
		if (src_shape == nullptr) {
			throw std::invalid_argument("src_shape is a NULL pointer.");
		}
		Tensor<T> result(src_dim, src_shape);
		int size = 1;
		srand(time(NULL));
		for (int i = 0; i < src_dim; ++i) {
			size *= src_shape[i];
		}
		for (int i = 0; i < size; ++i) {
			result.data[i] = 1.0 * std::rand() / (T)RAND_MAX;
		}
		return result;
	}

	template <typename T, typename G>
	Tensor<T> rand_like(Tensor<G> src) {
		// randomly initialize the tensor with the same shape as src
		return rand<T>(src.dim, src.shape);
	}

	template <typename T>
	Tensor<T> full(int src_dim, int src_shape[], T src_val) {
		// create a tensor with given shape and fill it with src_val
		if (src_dim <= 0) {
			throw std::invalid_argument("Non-positive dimention.");
		}
		if (src_shape == nullptr) {
			throw std::invalid_argument("src_shape is a NULL pointer.");
		}
		Tensor<T> result(src_dim, src_shape);
		int size = 1;
		for (int i = 0; i < src_dim; ++i) {
			size *= src_shape[i];
		}
		for (int i = 0; i < size; ++i) {
			result.data[i] = src_val;
		}
		return result;
	}

	template <typename T, typename G>
	Tensor<T> full_like(Tensor<G> src, T src_val) {
		// create a tensor with the same shape as src and fill it with src_val
		return full<T>(src.dim, src.shape, src_val);
	}
	
	template <typename T>
	Tensor<T> excrange(T start, T end, T step) {
		// create a 1D tensor from start to end (exclusive) with common difference step
		if (step == 0) {
			throw std::invalid_argument("step cannot be zero.");
		}
		int size = ceil((double)(end - start) / step);
		Tensor<T> result(1, new int[1]{size});
		for (int i = 0; i < size; ++i) {
			result.data[i] = start + i * step;
		}
		return result;
	}

	template <typename T>
	Tensor<T> incrange(T start, T end, T step) {
		// create a 1D tensor from start to end (inclusive) with common difference step
		if (step == 0) {
			throw std::invalid_argument("step cannot be zero.");
		}
		int size = floor((double)(end - start) / step) + 1;
		Tensor<T> result(1, new int[1]{size});
		for (int i = 0; i < size; ++i) {
			result.data[i] = start + i * step;
		}
		return result;
	}

// Part 2: tensor operations

	template <typename T>
	Tensor<T> slice(Tensor<T> src, std::pair<int,int> range[]) {
		// get the slice of the tensor from first to second (exclusive)
		return src.slice(range);
	}

	template <typename T>
	Tensor<T> concat(Tensor<T> src1, Tensor<T> src2, int dim) {
		// concatenate two tensors along the given dimension
		return src1.concat(src2, dim);
	}

	template <typename T>
	Tensor<T> tile(Tensor<T> src, int reps[]) {
		// repeat the tensor along each dimension by reps times
		return src.tile(reps);
	}

	template <typename T>
	void modify(Tensor<T> src, std::pair<int,int> range[], Tensor<T> val) {
		// modify the tensor src from src_start to src_end (exclusive) with the data from tensor val
		return src.modify(range, val);
	}

	template <typename T>
	Tensor<T> transpose(Tensor<T> src, int src_dim1, int src_dim2) {
		// transpose the tensor along the given dimensions
		return src.transpose(src_dim1, src_dim2);
	}

	template <typename T>
	Tensor<T> permute(Tensor<T> src, int src_order[]) {
		// permute the tensor with the given order
		return src.permute(src_order);
	}

	template <typename T>
	void load(Tensor<T> src, const char* filename) {
		// load the tensor from the given file
		return src.load(filename);
	}

	template <typename T>
	void print(Tensor<T> src, int newDim, int newShape[], FILE* out = stdout, bool printShape = false) {
		// print a tensor in given shape
		return src.print(newDim, newShape, out, printShape);
	}

	template <typename T>
	void print(Tensor<T> src, FILE* out = stdout, bool printShape = false) {
		// print a tensor in original shape
		return src.print(out, printShape);
	}

//	Part 3: Mathematical Operations
//  Part 3.1

	template <typename T>
	Tensor<T> operator + (Tensor<T> src1, Tensor<T> src2) {
		return  src1.add(src2);
	}
	template <typename T>
	Tensor<T> add(Tensor<T> src1, Tensor<T> src2) {
		return src1.add(src2);
	}
	// Inplementation of Add() between 2 Tensor

	template <typename T>
	Tensor<T> operator + (Tensor<T> src1, T src2) {
		return src1.add(src2);
	}
	template <typename T>
	Tensor<T> add(Tensor<T> src1, T src2) {
		return src1.add(src2);
	}
	// Inplementation of Add() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator - (Tensor<T> src1, Tensor<T> src2) {
		return  src1.sub(src2);
	}
	template <typename T>
	Tensor<T> sub(Tensor<T> src1, Tensor<T> src2) {
		return src1.sub(src2);
	}
	// Inplementation of Sub() between 2 Tensor

	template <typename T>
	Tensor<T> operator - (Tensor<T> src1, T src2) {
		return src1.sub(src2);
	}
	template <typename T>
	Tensor<T> sub(Tensor<T> src1, T src2) {
		return src1.sub(src2);
	}
	// Inplementation of Sub() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator * (Tensor<T> src1, Tensor<T> src2) {
		return src1.mul(src2);
	}
	template <typename T>
	Tensor<T> mul(Tensor<T> src1, Tensor<T> src2) {
		return src1.mul(src2);
	}
	// Inplementation of Sub() between 2 Tensor

	template <typename T>
	Tensor<T> operator * (Tensor<T> src1, T src2) {
		return src1.mul(src2);
	}
	template <typename T>
	Tensor<T> mul(Tensor<T> src1, T src2) {
		return src1.mul(src2);
	}
	// Inplementation of Sub() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator / (Tensor<T> src1, Tensor<T> src2) {
		return  src1.Div(src2);
	}
	template <typename T>
	Tensor<T> Div(Tensor<T> src1, Tensor<T> src2) {
		return src1.Div(src2);
	}
	// Inplementation of Div() between 2 Tensor

	template <typename T>
	Tensor<T> operator / (Tensor<T> src1, T src2) {
		return src1.Div(src2);
	}
	template <typename T>
	Tensor<T> Div(Tensor<T> src1, T src2) {
		return src1.Div(src2);
	}
	// Inplementation of Div() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> Log(Tensor<T> src) {
		return src.Log();
	}
	// Inplementation of Log() with base of e

//  Part 3.2

	template <typename T>
	Tensor<T> sum(Tensor<T> src, int dim) {
		return src.sum(dim);
	}
	// Inplementation of Sum()

	template <typename T>
	T mean(Tensor<T> src) {
		return src.mean();
	}
	// Inplementation of Mean()

	template <typename T>
	T Min(Tensor<T> src) {
		return src.Min();
	}
	// Inplementation of Min()

	template <typename T>
	T Max(Tensor<T> src) {
		return src.Max();
	}
	// Inplementation of Max()

//  Part 3.3

	template <typename T>
	Tensor<bool> operator == (Tensor<T> src1, Tensor<T> src2) {
		return src1.eq(src2);
	}
	// Inplementation of eq()

	template <typename T>
	Tensor<bool> operator != (Tensor<T> src1, Tensor<T> src2) {
		return src1.ne(src2);
	}
	// Inplementation of ne()
	
	template <typename T>
	Tensor<bool> operator > (Tensor<T> src1, Tensor<T> src2) {
		return src1.gt(src2);
	}
	// Inplementation of gt()

	template <typename T>
	Tensor<bool> operator >= (Tensor<T> src1, Tensor<T> src2) {
		return src1.ge(src2);
	}
	// Inplementation of ge()

	template <typename T>
	Tensor<bool> operator < (Tensor<T> src1, Tensor<T> src2) {
		return src1.lt(src2);
	}
	// Inplementation of lt()

	template <typename T>
	Tensor<bool> operator <= (Tensor<T> src1, Tensor<T> src2) {
		return src1.le(src2);
	}
	// Inplementation of le()
}
