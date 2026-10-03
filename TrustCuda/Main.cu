#include <cstdlib>
#include <iostream>
#include "Peer.h"
#include "SimpleEigentrustGPU.h"

int main(void)
{
	const int m = 10000;
	const double error = 0.0001;
	const double damping = 0.15;
	const unsigned int num_transactions = 200000;

	if (m < 2)
		std::abort();

	std::vector<Peer> peers;
	for (int i = 0; i < m; i++)
	{
		peers.push_back(Peer());
	}

	SimpleEigentrustGPU eigentrust(peers, error, damping);
	Peer::generateInteractions(eigentrust.getPeers(), num_transactions);

	eigentrust.computeEigentrust(eigentrust.computeMatrix());

	std::cout << "Peer ID:\t Trust Value" << std::endl;
	for (auto i = peers.begin(); i != peers.end(); i++)
	{
		std::cout << i->getId() << ":\t" << i->getTrustValue() << std::endl;
	}
	std::cin.get();

	return 0;
}