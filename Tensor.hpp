#include <iostream>
#include <cmath>

namespace ts {
	template <typename T>
	class Tensor {
		public:
			int dim;
			int *shape;
			T *data;

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
			}

			~Tensor() {
				dim = -1;
				delete[] shape;
				delete[] data;
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
						char message[50];
						//sprintf(message, "Left bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
					if (range[i].second <= range[i].first || range[i].second > this->shape[i]) {
						char message[50];
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
						char message[50];
						//sprintf(message, "Left bound of dimension %d out of range.", i);
						throw std::invalid_argument(message);
					}
					if (range[i].second <= range[i].first || range[i].second > this->shape[i]) {
						char message[50];
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

			void print(int newDim, int newShape[]) {
				int oldSize = 1, newSize = 1;
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
					printf("tensor([");
					for (int i = 0; i < newShape[0]; ++i) {
						printf("%5g%s",
							(double)(this->data[i]), i == newShape[0] - 1 ? "])" : ", "
						);
					}
					puts("");
					return;
				}
				newSize /= newShape[newDim - 1];
				int *newIdx = new int[newDim];
				for (int i = 0; i < newDim; ++i) {
					newIdx[i] = 0;
				}
				for (int i = 0, j; i < newSize; ++i) {
					printf("%s", i ? "       " : "tensor(");
					for (j = this->dim - 2; j >= 0; --j) {
						if (newIdx[j] != 0) {
							break;
						}
					}
					for (int k = 0; k <= j; ++k) {
						printf(" ");
					}
					for (int k = j + 1; k < this->dim; ++k) {
						printf("[");
					}
					for (int k = 0; k < newShape[newDim - 1]; ++k) {
						printf("%5g%s",
							(double)(this->data[i * newShape[newDim - 1] + k]),
							k == newShape[newDim - 1] - 1 ? "]" : ", "
						);
					}
					int carryCnt = 1;
					++newIdx[newDim - 2];
					for (int k = newDim - 2; k > 0; --k) {
						if (newIdx[k] == newShape[k]) {
							newIdx[k] = 0;
							++newIdx[k - 1];
							++carryCnt;
						}
					}
					for (int k = 1; k < carryCnt; ++k) {
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

			void print() {
				return this->print(this->dim, this->shape);
			}

			Tensor<T> add(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("step cannot be zero.");
				}
				for (int i = 0; i < src.dim; ++i) 
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("step cannot be zero.");
					}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
				for (int i = 0; i < size; ++i) {
					result.data[i] = src.data[i] + this->data[i];
				}
				return result;
			}

			Tensor<T> add(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				for (int i = 0; i < size; ++i) result.data[i] = this->data[i] + src;
				return result;
			}

			Tensor<T> sub(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("step cannot be zero.");
				}
				for (int i = 0; i < src.dim; ++i) 
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("step cannot be zero.");
					}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
				for (int i = 0; i < size; ++i) {
					result.data[i] = src.data[i] - this->data[i];
				}
				return result;
			}

			Tensor<T> sub(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				for (int i = 0; i < size; ++i) result.data[i] = this->data[i] - src;
				return result;
			}

			Tensor<T> mul(Tensor src) {
				if (this->dim != src.dim) throw std::invalid_argument("step cannot be zero.");
				for (int i = 0; i < this->dim - 2; ++i) 
					if (this->shape[i] != src.shape[i]) throw std::invalid_argument("step cannot be zero.");
				if (this->shape[this->dim - 1] != src.shape[src.dim - 2] || this->shape[this->dim - 2] != src.shape[src.dim - 1]) {
					throw std::invalid_argument("step cannot be zero.");
				}
				int *tmp_shape = new int[this->dim];
				for (int i = 0; i < this->dim - 1; ++i) tmp_shape[i] = this->shape[i]; 
				tmp_shape[this->dim - 1] = src.shape[src.dim - 1];
				Tensor<T> result(this->dim, tmp_shape);
				int base = this->shape[this->dim - 2] * src.shape[src.dim - 1];
				int size = 1, row = this->shape[this->dim - 2], col = src.shape[src.dim - 1];
				int base1 = this->shape[this->dim - 1] * this->shape[this->dim - 2];
				int base2 = src.shape[src.dim - 1] * src.shape[src.dim - 2];
				for (int i = 0; i < this->dim - 2; ++i) size = size * this->shape[i];
				for (int i = 0; i < size; ++i) { 
					for (int j = 0; j < row; ++j) {
						for (int k = 0; k < col; ++k) {
							T cur = (T)0;
							for (int l = 0; l < this->shape[this->dim - 1]; ++l) 
								cur += this->data[i * base1 + j * this->shape[this->dim - 1] + l] * src.data[i * base2 + l * src.shape[src.dim - 1] + k];
							result.data[i * base + j * col + k] = cur;
						}
					}
				}
				return result;
			}

			Tensor<T> mul(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				for (int i = 0; i < size; ++i) result.data[i] = this->data[i] * src;
				return result;
			}

			Tensor<T> Div(Tensor<T> src) {
				if (src.dim != this->dim) {
					throw std::invalid_argument("step cannot be zero.");
				}
				for (int i = 0; i < src.dim; ++i) 
					if (src.shape[i] != this->shape[i]) {
						throw std::invalid_argument("step cannot be zero.");
					}
				Tensor<T> result(src.dim, src.shape);
				int size = 1;
				for (int i = 0; i < src.dim; ++i) size *= src.shape[i];
				for (int i = 0; i < size; ++i) {
					result.data[i] = src.data[i] / this->data[i];
				}
				return result;
			}

			Tensor<T> Div(T src) {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				for (int i = 0; i < size; ++i) result.data[i] = this->data[i] / src;
				return result;
			}

			Tensor<T> Log() {
				Tensor<T> result(this->dim, this->shape);
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				for (int i = 0; i < size; ++i) result.data[i] = log(this->data[i]);
				return result;
			}

			Tensor<T> sum(int dim) {
				if (dim < 0 || dim >= this->dim) {
					throw std::invalid_argument("step cannot be zero.");
				} else 
				if (dim == 0) {
					int size = 1;
					for (int i = 1; i < this->dim; ++i) {
						size *= this->shape[i];
					}
					int *tmp_shape = new int[size];
					for (int i = 1; i < this->dim; ++i) {
						tmp_shape[i - 1] = this->shape[i];
					}
					Tensor<T> result(this->dim - 1, tmp_shape);
					for (int i = 0; i < size; ++i) {
						T cur = (T)0;
						for (int j = 0; j < this->shape[0]; ++j) cur += this->data[i + j * size];
						result.data[i] = cur; 
					}
					return result;
				} else 
				if (dim == this->dim - 1) {
					int size = 1;
					for (int i = 0; i < this->dim - 1; ++i) {
						size *= this->shape[i];
					}
					int *tmp_shape = new int[size];
					for (int i = 0; i < this->dim - 1; ++i) {
						tmp_shape[i] = this->shape[i];
					}
					Tensor<T> result(this->dim - 1, tmp_shape);
					for (int i = 0; i < size; ++i) {
						T cur = (T)0; 
						for (int j = 0; j < this->shape[this->dim - 1]; ++j) cur += this->data[i * this->shape[this->dim - 1] + j];
						result.data[i] = cur;
					}
					return result;
				} else {
					int suf_size = 1, size = 1, pre_size = 1;
					for (int i = dim + 1; i < this->dim; ++i) suf_size = suf_size * this->shape[i];
					for (int i = 0; i < dim; ++i) pre_size = pre_size * this->shape[i];
					for (int i = 0; i < this->dim; ++i) 
						if (i != dim) size = size * this->shape[i];
					int *tmp_shape = new int[size];
					for (int i = 0; i < dim; ++i) tmp_shape[i] = this->shape[i];
					for (int i = dim + 1; i < this->dim; ++i) tmp_shape[i - 1] = this->shape[i];
					Tensor<T> result(this->dim - 1, tmp_shape);
					for (int i = 0; i < pre_size; ++i) {
						for (int j = 0; j < suf_size; ++j) {
							T cur = (T)0;
							for (int k = 0; k < this->shape[dim]; ++k) 
								cur += this->data[i * suf_size * this->shape[dim] + k * suf_size + j];
							result.data[i * suf_size + j] = cur;
						}
					}
					return result;
				}
			}

			T mean() {
				int size = 1; 
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				T cur = (T)0;
				for (int i = 0; i < size; ++i) cur = cur + this->data[i];
				return cur / size;
			}

			T Min() {
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				T cur = this->shape[0];
				for (int i = 1; i < size; ++i) if (this->data[i] < cur) cur = this->data[i];
				return cur;
			}

			T Max() {
				int size = 1;
				for (int i = 0; i < this->dim; ++i) size *= this->shape[i];
				T cur = this->shape[0];
				for (int i = 1; i < size; ++i) if (this->data[i] > cur) cur = this->data[i];
				return cur;
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
	void print(Tensor<T> src, int newDim, int newShape[]) {
		// print a tensor in given shape
		return src.print(newDim, newShape);
	}

	template <typename T>
	void print(Tensor<T> src) {
		// print a tensor in original shape
		return src.print();
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
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] == src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of eq()

	template <typename T>
	Tensor<bool> operator != (Tensor<T> src1, Tensor<T> src2) {
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] != src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of ne()
	
	template <typename T>
	Tensor<bool> operator > (Tensor<T> src1, Tensor<T> src2) {
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] > src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of gt()

	template <typename T>
	Tensor<bool> operator >= (Tensor<T> src1, Tensor<T> src2) {
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] >= src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of ge()

	template <typename T>
	Tensor<bool> operator < (Tensor<T> src1, Tensor<T> src2) {
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] < src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of lt()

	template <typename T>
	Tensor<bool> operator <= (Tensor<T> src1, Tensor<T> src2) {
		if (src1.dim != src2.dim) throw std::invalid_argument("step cannot be zero.");
		for (int i = 0; i < src1.dim; ++i) 
			if (src1.shape[i] != src2.shape[i]) throw std::invalid_argument("step cannot be zero.");
		Tensor<bool> result(src1.dim, src1.shape);
		int size = 1;
		for (int i = 0; i < src1.dim; ++i) size *= src1.shape[i];
		for (int i = 0; i < size; ++i) if (src1.data[i] <= src2.data[i]) result.data[i] = true; else result.data[i] = false;
		return result;
	}
	// Inplementation of le()
}