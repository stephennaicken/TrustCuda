#include "SimpleEigentrustGPU.h"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <cublas_v2.h>

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
	return norm < Eigentrust::getError() ? true : false;
}

void SimpleEigentrustGPU::computeEigentrust(double * C, double * e, double * y)
{
	std::vector<Peer>::size_type m = getPeers().size();
	double alpha = 1;
	double beta = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	do{
		cublasDgemv(handle, CUBLAS_OP_T, m, m, &alpha, C, m, e, 1, &beta, y, 1);
		double * tmp = e;
		e = y;
		y = tmp;
	} while (!hasConverged(e, y));
	cublasDestroy(handle);
	setTrustValues(e);
}

void SimpleEigentrustGPU::setTrustValues(double * dev_trust_vector)
{
	thrust::device_ptr<double> dev_trust_vec_ptr = thrust::device_pointer_cast(dev_trust_vector);
	thrust::device_vector<double> d_trust_vector(dev_trust_vec_ptr, dev_trust_vec_ptr + getPeers().size());
	thrust::host_vector<double> host_trust_vector = d_trust_vector;
	for (int i = 0; i < getPeers().size(); i++)
	{
		getPeers().at(i).setTrustValue(*(host_trust_vector.begin() + i));
	}
}