#pragma once

#include <random>
#include <map>

class Peer
{
private:
	static const signed int positive_transaction;
	static const signed int negative_transaction;

	static unsigned int id_count;

	static const double mean;
	static const double std_dev;
	static std::default_random_engine rnd_engine;
	static std::normal_distribution<> distribution;

	const unsigned int id;
	std::map<Peer*, signed int> transactions;
	double trust_value;
public:
	Peer() : id(id_count++), transactions(), trust_value(0){}
	~Peer(){};
	void interact(Peer & j, bool init);
	const std::map<Peer*, signed int> & getTransactions() const
	{
		return transactions;
	}
	unsigned int getId() const { return id; }
	void setTrustValue(const double trust_value) { this->trust_value = trust_value; }
	double getTrustValue() const { return trust_value; }
	static void generateInteractions(std::vector<Peer>& peers, const unsigned int num_transactions);
	Peer & operator=(const Peer &);
};

