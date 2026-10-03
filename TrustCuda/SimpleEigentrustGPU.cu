#include "SimpleEigentrustGPU.h"
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <cusparse.h>

#define CHECK_CUDA(call) do { cudaError_t s = (call); if (s != cudaSuccess) { \
	std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(s)); std::exit(EXIT_FAILURE); } } while (0)
#define CHECK_CUSPARSE(call) do { cusparseStatus_t s = (call); if (s != CUSPARSE_STATUS_SUCCESS) { \
	std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cusparseGetErrorString(s)); std::exit(EXIT_FAILURE); } } while (0)

bool SimpleEigentrustGPU::hasConverged(double * trust_vec_next, double * trust_vec_orig)
{
	std::vector<Peer>::size_type m = getPeers().size();
	thrust::device_vector<double> d_v(m);
	square<double>        unary_op;
	thrust::plus<double> binary_op;
	double init = 0;

	thrust::device_ptr<double> trust_vec_next_ptr = thrust::device_pointer_cast(trust_vec_next);
	thrust::device_ptr<double> trust_vec_orig_ptr = thrust::device_pointer_cast(trust_vec_orig);
	thrust::transform(trust_vec_next_ptr, trust_vec_next_ptr + m, trust_vec_orig_ptr, d_v.begin(), thrust::minus<double>());
	double norm = std::sqrt(thrust::transform_reduce(d_v.begin(), d_v.end(), unary_op, init, binary_op));
	return norm < getError() ? true : false;
}

void SimpleEigentrustGPU::computeEigentrust(const TrustMatrix & C)
{
	const int m = static_cast<int>(getPeers().size());
	const int nnz = static_cast<int>(C.values.size());
	const double a = getDamping();

	thrust::device_vector<int> d_row_ptr = C.row_ptr;
	thrust::device_vector<int> d_col_idx = C.col_idx;
	thrust::device_vector<double> d_values = C.values;
	thrust::device_vector<double> d_dangling = C.dangling;
	thrust::device_vector<double> d_e(m, 1 / static_cast<double>(m));
	thrust::device_vector<double> d_y(m);
	double * e = thrust::raw_pointer_cast(d_e.data());
	double * y = thrust::raw_pointer_cast(d_y.data());

	cusparseHandle_t handle;
	cusparseSpMatDescr_t mat = nullptr;
	cusparseDnVecDescr_t vec_e, vec_y;
	void * buffer = nullptr;
	double alpha = 1 - a;
	double beta = 1;
	CHECK_CUSPARSE(cusparseCreate(&handle));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vec_e, m, e, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y, m, y, CUDA_R_64F));
	if (nnz > 0)
	{
		CHECK_CUSPARSE(cusparseCreateCsr(&mat, m, m, nnz,
			thrust::raw_pointer_cast(d_row_ptr.data()), thrust::raw_pointer_cast(d_col_idx.data()), thrust::raw_pointer_cast(d_values.data()),
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		size_t buffer_size = 0;
		CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_e, &beta, vec_y,
			CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
		CHECK_CUDA(cudaMalloc(&buffer, buffer_size));
	}

	// t(k+1) = (1 - a) * C^T * t(k) + a * p, with p the uniform pre-trust vector.
	// Dangling rows of C are uniform, so they add (1 - a) * (dangling mass) / m to every peer.
	do{
		thrust::device_ptr<double> e_ptr = thrust::device_pointer_cast(e);
		thrust::device_ptr<double> y_ptr = thrust::device_pointer_cast(y);
		double dangling_mass = thrust::inner_product(d_dangling.begin(), d_dangling.end(), e_ptr, 0.0);
		thrust::fill(y_ptr, y_ptr + m, ((1 - a) * dangling_mass + a) / m);
		if (nnz > 0)
		{
			CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_e, &beta, vec_y,
				CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
		}
		std::swap(e, y);
		std::swap(vec_e, vec_y);
	} while (!hasConverged(e, y));

	setTrustValues(e);

	CHECK_CUDA(cudaFree(buffer));
	if (mat)
		CHECK_CUSPARSE(cusparseDestroySpMat(mat));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vec_e));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vec_y));
	CHECK_CUSPARSE(cusparseDestroy(handle));
}

void SimpleEigentrustGPU::setTrustValues(double * dev_trust_vector)
{
	thrust::device_ptr<double> dev_trust_vec_ptr = thrust::device_pointer_cast(dev_trust_vector);
	thrust::device_vector<double> d_trust_vector(dev_trust_vec_ptr, dev_trust_vec_ptr + getPeers().size());
	thrust::host_vector<double> host_trust_vector = d_trust_vector;
	for (auto i = getPeers().begin(); i != getPeers().end(); i++)
	{
		i->setTrustValue(*(host_trust_vector.begin() + i->getId()));
	}
}
