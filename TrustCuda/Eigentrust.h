#pragma once
#include "ReputationSystem.h"

class Eigentrust :
	public ReputationSystem
{
private:
	const double error;
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
			{
				while (begin != end)
				{
					Peer p = ((Peer)*begin);
					unsigned int id = p.getId();
					const std::map<Peer*, signed int> & map = p.getTransactions();
					size_t size = p.getTransactions().size();
					if (size > 0)
					{
						unsigned int sum = 0;
						for (std::map<Peer*, signed int>::const_iterator i = map.begin(); i != map.end(); i++)
						{
							unsigned int j = i->first->getId();
							sum += *(hv_begin + idx2c(id, j, ReputationSystem::getPeers().size())) = std::max(i->second, 0);
						}

						for (auto i = map.begin(); i != map.end(); i++)
						{
							unsigned int j = i->first->getId();
							*(hv_begin + idx2c(id, j, ReputationSystem::getPeers().size())) = *(hv_begin + idx2c(id, j, ReputationSystem::getPeers().size())) / (double)sum;;
						}
					}
					begin++;
				}
			}
	}
	virtual void computeEigentrust(double * C, double * e, double * y) = 0;
	double getError() const { return error; }
};

