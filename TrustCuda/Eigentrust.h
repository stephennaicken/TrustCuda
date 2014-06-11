#pragma once
#include "ReputationSystem.h"

class Eigentrust :
	public ReputationSystem
{
private:
	double error;
protected:
	inline unsigned int idx2c(unsigned int i, unsigned int j, unsigned int ld) const
	{
		return (j*ld) + i;
	};
	virtual bool hasConverged(double * trust_vec_next, double * trust_vec_orig) = 0;
public:
	Eigentrust(std::vector<Peer>& peers, double err) : ReputationSystem(peers), error(err){};
	virtual ~Eigentrust(){};
	template<class MatrixVectorIterator, class PeerIterator>
	void computeMatrix(MatrixVectorIterator hv_begin, MatrixVectorIterator hv_end, PeerIterator begin, PeerIterator end){
		while (begin != end)
		{
			unsigned int id = begin->getId();
			const std::map<Peer*, signed int> & map = begin->getTransactions();
			size_t size = begin->getTransactions().size();
			if (size > 0)
			{
				unsigned int sum = 0;
				for (auto i = map.begin(); i != map.end(); i++)
				{
					unsigned int j = i->first->getId();
					unsigned int matrix_pos = idx2c(id, j, getPeers().size());
					sum += *(hv_begin + matrix_pos) = std::max(i->second, 0);
				}

				for (auto i = map.begin(); i != map.end(); i++)
				{
					unsigned int j = i->first->getId();
					unsigned int matrix_pos = idx2c(id, j, getPeers().size());
					*(hv_begin + matrix_pos) = *(hv_begin + matrix_pos) / (double)sum;
				}
			}
			begin++;
		}
	}
	virtual void computeEigentrust(double * C, double * e, double * y) = 0;
	double getError() const { return error; }
};

