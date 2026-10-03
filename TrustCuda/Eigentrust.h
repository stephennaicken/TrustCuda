#pragma once
#include <algorithm>
#include "ReputationSystem.h"

class Eigentrust :
	public ReputationSystem
{
private:
	double error;
	double damping;
protected:
	inline unsigned int idx2c(unsigned int i, unsigned int j, unsigned int ld) const
	{
		return (j*ld) + i;
	};
	virtual bool hasConverged(double * trust_vec_next, double * trust_vec_orig) = 0;
public:
	Eigentrust(std::vector<Peer>& peers, double err, double a) : ReputationSystem(peers), error(err), damping(a){};
	virtual ~Eigentrust(){};
	// Builds the normalized local trust matrix C (column-major, m x m), where row i
	// holds peer i's trust in every other peer. Rows of peers without any positive
	// transactions fall back to the uniform distribution so that C stays row-stochastic.
	// The range [hv_begin, hv_end) must be zero-initialized.
	template<class MatrixVectorIterator, class PeerIterator>
	void computeMatrix(MatrixVectorIterator hv_begin, MatrixVectorIterator hv_end, PeerIterator begin, PeerIterator end){
		const unsigned int m = getPeers().size();
		while (begin != end)
		{
			unsigned int id = begin->getId();
			const std::map<Peer*, signed int> & map = begin->getTransactions();
			unsigned int sum = 0;
			for (auto i = map.begin(); i != map.end(); i++)
			{
				unsigned int j = i->first->getId();
				unsigned int matrix_pos = idx2c(id, j, m);
				sum += *(hv_begin + matrix_pos) = std::max(i->second, 0);
			}

			if (sum > 0)
			{
				for (auto i = map.begin(); i != map.end(); i++)
				{
					unsigned int j = i->first->getId();
					unsigned int matrix_pos = idx2c(id, j, m);
					*(hv_begin + matrix_pos) = *(hv_begin + matrix_pos) / (double)sum;
				}
			}
			else
			{
				for (unsigned int j = 0; j < m; j++)
				{
					*(hv_begin + idx2c(id, j, m)) = 1 / (double)m;
				}
			}
			begin++;
		}
	}
	virtual void computeEigentrust(double * C, double * e, double * y) = 0;
	double getError() const { return error; }
	double getDamping() const { return damping; }
};

