#pragma once

#include <vector>
#include "Peer.h"

class ReputationSystem
{
private:
	std::vector<Peer> & peers;
public:
	ReputationSystem(std::vector<Peer> & peer_list) : peers(peer_list){}
	virtual ~ReputationSystem() {};
	std::vector<Peer> & getPeers() const { return peers; }
};

