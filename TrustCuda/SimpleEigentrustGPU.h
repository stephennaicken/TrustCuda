#pragma once
#include "Eigentrust.h"

// square<T>: f(x) -> x*x
template <typename T>
struct square
{
	__host__ __device__
		T operator()(const T& x) const {
		return x * x;
	}
};

class SimpleEigentrustGPU final :
	public Eigentrust
{
private:
	void setTrustValues(double *);
protected:
	bool hasConverged(double * trust_vec_next,  double * trust_vec_orig) override;
public:
	SimpleEigentrustGPU(std::vector<Peer>& peers, double error) : Eigentrust(peers, error){}
	~SimpleEigentrustGPU(){};
	void computeEigentrust(double * C, double * e, double * y) override;
};

