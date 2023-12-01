namespace ts {
	template <typename T>
	class Tensor {
		private:
			int dim;
			int *shape;
			T *data;

		public:
			Tensor(int src_dim, int src_shape[]);
			Tensor(int src_dim, int src_shape[], T *src_data);
			~Tensor();
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
	Tensor<T> arange(T start, T end, T step);
	// create a 1D tensor from start to end (exclusive)
	// with common difference step

	template <typename T>
	Tensor<T> range(T start, T end, T step);
	// create a 1D tensor from start to end (inclusive)
	// with common difference step

	// Part 2: tensor operations

	template <typename T>
	Tensor<T> index(Tensor<T> src, int src_index[]);
	// get the value of the tensor at the given index

	template <typename T>
	Tensor<T> slice(Tensor<T> src, int src_dim, int src_start, int src_end);
	// get the slice of the tensor from src_start to src_end (exclusive)
	// along dimension src_dim

	template <typename T>
	Tensor<T> slice(Tensor<T> src, int src_start[], int src_end[]);
	// get the slice of the tensor from src_start to src_end (exclusive)

	template <typename T>
	Tensor<T> concat(Tensor<T> src1, Tensor<T> src2, int src_dim);
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
}