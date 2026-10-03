#include "Peer.h"
#include <stdexcept>
#include <time.h>

const signed int Peer::positive_transaction = 1;
const signed int Peer::negative_transaction = -1;

const double Peer::mean = 0;
const double Peer::std_dev = 1;
std::default_random_engine Peer::rnd_engine = std::default_random_engine();
std::normal_distribution<> Peer::distribution(mean, std_dev);

unsigned int Peer::id_count = 0;

void Peer::interact(Peer & j, bool init){
	double n = distribution(rnd_engine);
	signed int trans_val = n >= -std_dev && n <= std_dev ? positive_transaction : negative_transaction;
	try{
		signed int & val = transactions.at(&j);
		val = val + trans_val;
	}
	catch (const std::out_of_range & oor)
	{
		transactions.insert(std::pair<Peer*, signed int>(&j, trans_val));
	}
	if (init)
		j.interact(*this, false);
}

void Peer::generateInteractions(std::vector<Peer>& peers, const unsigned int num_transactions)
{
	if (peers.size() < 2)
		return;
	srand(time(nullptr));
	for (unsigned int n = 0; n < num_transactions; n++)
	{
		size_t i = rand() % peers.size();
		size_t j;
		do {
			j = rand() % peers.size();
		} while (j == i);
		peers.at(i).interact(peers.at(j), true);
	}
}