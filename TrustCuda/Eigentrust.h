#pragma once
#include <algorithm>
#include "ReputationSystem.h"

// Transposed local trust matrix C^T in CSR format, so that row j holds the
// normalized trust every peer i places in peer j. Peers without any positive
// transactions ("dangling" peers) have no entries; their row of C is the
// uniform distribution, which is applied implicitly via the dangling vector.
struct TrustMatrix
{
	std::vector<int> row_ptr;
	std::vector<int> col_idx;
	std::vector<double> values;
	std::vector<double> dangling;	// 1 for dangling peers, 0 otherwise
};

class Eigentrust :
	public ReputationSystem
{
private:
	double error;
	double damping;
protected:
	virtual bool hasConverged(double * trust_vec_next, double * trust_vec_orig) = 0;
public:
	Eigentrust(std::vector<Peer>& peers, double err, double a) : ReputationSystem(peers), error(err), damping(a){};
	virtual ~Eigentrust(){};

	TrustMatrix computeMatrix() const
	{
		const std::vector<Peer> & peers = getPeers();
		const size_t m = peers.size();
		TrustMatrix C;
		C.row_ptr.assign(m + 1, 0);
		C.dangling.assign(m, 0);

		// Normalization factor of each row of C and the number of entries in each row of C^T
		std::vector<unsigned int> sums(m, 0);
		for (auto i = peers.begin(); i != peers.end(); i++)
		{
			const std::map<Peer*, signed int> & map = i->getTransactions();
			for (auto t = map.begin(); t != map.end(); t++)
			{
				if (t->second > 0)
				{
					sums[i->getId()] += t->second;
					C.row_ptr[t->first->getId() + 1]++;
				}
			}
			if (sums[i->getId()] == 0)
				C.dangling[i->getId()] = 1;
		}
		for (size_t j = 0; j < m; j++)
			C.row_ptr[j + 1] += C.row_ptr[j];

		C.col_idx.resize(C.row_ptr[m]);
		C.values.resize(C.row_ptr[m]);
		std::vector<int> next(C.row_ptr.begin(), C.row_ptr.end() - 1);
		// Peers are visited in id order, so column indices end up sorted within each row
		for (auto i = peers.begin(); i != peers.end(); i++)
		{
			const std::map<Peer*, signed int> & map = i->getTransactions();
			for (auto t = map.begin(); t != map.end(); t++)
			{
				if (t->second > 0)
				{
					int pos = next[t->first->getId()]++;
					C.col_idx[pos] = i->getId();
					C.values[pos] = t->second / (double)sums[i->getId()];
				}
			}
		}
		return C;
	}
	virtual void computeEigentrust(const TrustMatrix & C) = 0;
	double getError() const { return error; }
	double getDamping() const { return damping; }
};
