#include <array>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Peer.h"
#include "SimpleEigentrustGPU.h"

int main(void)
{
	const int m = 10000;
	const double error = 0.0001;
	const unsigned int num_transactions = 200000;

	if (m == 0)
		std::abort();

	std::vector<Peer> peers;
	for (int i = 0; i < m; i++)
	{
		peers.push_back(Peer());
	}

	SimpleEigentrustGPU eigentrust(peers, error);
	Peer::generateInteractions(eigentrust.getPeers(), num_transactions);

	thrust::host_vector<double> C(pow(m, 2));
	eigentrust.computeMatrix(C.begin(), C.end(), peers.begin(), peers.end());

	thrust::device_vector<double> d_C = C;
	thrust::device_vector<double> d_e(m, 1 / static_cast<double>(m));
	thrust::device_vector<double> d_y(m);
	eigentrust.computeEigentrust(thrust::raw_pointer_cast(&d_C[0]), thrust::raw_pointer_cast(&d_e[0]), thrust::raw_pointer_cast(&d_y[0]));

	thrust::host_vector<double> e = d_e;

	std::cout << "Peer ID:\t Trust Value" << std::endl;
	for (auto i = peers.begin(); i != peers.end(); i++)
	{
		std::cout << i->getId() << ":\t" << i->getTrustValue() << std::endl;
	}
	std::cin.get();

	return 0;
}