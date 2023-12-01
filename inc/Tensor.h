#include <iostream>

namespace ts {
	template <typename T>
	class Tensor {
		public:
			int dim;
			int *shape;
			T *data;

		public:
			Tensor(int src_dim, int src_shape[]);
			Tensor(int src_dim, int src_shape[], T *src_data);
			~Tensor();
			T getVal(int idx[]);
			Tensor<T> transpose(int src_dim1, int src_dim2);
			Tensor<T> permute(int src_order[]);
	};

// Part 1: creation and initialization

	template <typename T>
	Tensor<T> zeros(int src_dim, int src_shape[]);
	// initialize with zeros with given shape

	template <typename T>
	Tensor<T> zeros_like(Tensor<T> src);
	// initialize with zeros with the same shape as src

	template <typename T>
	Tensor<T> ones(int src_dim, int src_shape[]);
	// initialize with ones with given shape

	template <typename T>
	Tensor<T> ones_like(Tensor<T> src);
	// initialize with ones with the same shape as src

	template <typename T>
	Tensor<T> eye(int src_dim);
	// initialize a 2D tensor with src_dim * src_dim size

	template <typename T>
	Tensor<T> rand(int src_dim, int src_shape[]);
	// randomly initialize the tensor with given shape

	template <typename T>
	Tensor<T> rand_like(Tensor<T> src);
	// randomly initialize the tensor with the same shape as src

	template <typename T>
	Tensor<T> full(int src_dim, int src_shape[], T src_val);
	// create a tensor with given shape and fill it with src_val

	template <typename T>
	Tensor<T> full_like(Tensor<T> src, T src_val);
	// create a tensor with the same shape as src and fill it with src_val

	template <typename T>
	Tensor<T> excrange(T start, T end, T step);
	// create a 1D tensor from start to end (exclusive)
	// with common difference step

	template <typename T>
	Tensor<T> incrange(T start, T end, T step);
	// create a 1D tensor from start to end (inclusive)
	// with common difference step

// Part 2: tensor operations

	template <typename T>
	Tensor<T> slice(Tensor<T> src, std::pair<int,int> range[]);
	// get the slice of the tensor from first to second (exclusive)

	template <typename T>
	Tensor<T> concat(Tensor<T> src1, Tensor<T> src2, int dim);
	// concatenate two tensors along the given dimension

	template <typename T>
	Tensor<T> tile(Tensor<T> src, int reps[]);
	// repeat the tensor along each dimension by reps times

	template <typename T>
	Tensor<T> modify(int src_dim, int src_start, int src_end, T src_val);
	// modify the tensor from src_start to src_end (exclusive)
	// along dimension src_dim data from with src_val

	template <typename T>
	Tensor<T> modify(Tensor<T> src, int src_start[], int src_end[], Tensor<T> val);
	// modify the tensor src from src_start to src_end (exclusive)
	// with the data from tensor val

	template <typename T>
	Tensor<T> transpose(Tensor<T> src, int src_dim1, int src_dim2);
	// transpose the tensor along the given dimensions

	template <typename T>
	Tensor<T> permute(Tensor<T> src, int src_order[]);
	// permute the tensor with the given order

//	Part 3: Math Operations
//3.1------------------------------------------------------------------------------------------------
	template <typename T>
	Tensor<T> operator + (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of Add() between 2 Tensor

	template <typename T>
	Tensor<T> operator + (Tensor<T> src1, T src2);
	// Inplementation of Add() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator - (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of Sub() between 2 Tensor

	template <typename T>
	Tensor<T> operator - (Tensor<T> src1, T src2);
	// Inplementation of Sub() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator * (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of Sub() between 2 Tensor

	template <typename T>
	Tensor<T> operator * (Tensor<T> src1, T src2);
	// Inplementation of Sub() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> operator / (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of Div() between 2 Tensor

	template <typename T>
	Tensor<T> operator / (Tensor<T> src1, T src2);
	// Inplementation of Div() between a Tensor and a single prototype(int, float...)

	template <typename T>
	Tensor<T> log(Tensor<T> src);
	// Inplementation of Log() with base of e
//------------------------------------------------------------------------------------------------------
//3.2---------------------------------------------------------------------------------------------------
	template <typename T>
	Tensor<T> sum(Tensor<T> src, int dim);
	// Inplementation of Sum()

	template <typename T>
	T mean(Tensor<T> src);
	// Inplementation of Mean()

	template <typename T>
	T min(Tensor<T> src);
	// Inplementation of Min()

	template <typename T>
	T max(Tensor<T> src);
	// Inplementation of Max()
//------------------------------------------------------------------------------------------------------
//3.3---------------------------------------------------------------------------------------------------
	template <typename T>
	Tensor<bool> operator == (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of eq()

	template <typename T>
	Tensor<bool> operator != (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of ne()
	
	template <typename T>
	Tensor<bool> operator > (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of gt()

	template <typename T>
	Tensor<bool> operator >= (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of ge()

	template <typename T>
	Tensor<bool> operator < (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of lt()

	template <typename T>
	Tensor<bool> operator <= (Tensor<T> src1, Tensor<T> src2);
	// Inplementation of le()
}