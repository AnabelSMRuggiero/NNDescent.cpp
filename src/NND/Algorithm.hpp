#ifndef NND_ALGORITHM_HPP
#define NND_ALGORITHM_HPP
#include <algorithm>
#include "GraphStructures.hpp"

namespace nnd{

//Placeholder until proper initialiation strat is implemented
//Simply queues up all possible local joins for each point
template<typename DataType>
void PopulateInitialQueueStates(const Graph<DataType>& graphState, std::vector<ComparisonQueue>& queues){
    for(const auto& vertex : graphState){
        for(const auto& neighbor : vertex.neighbors){
            for(const auto& nextNeighbor : graphState[neighbor.first].neighbors){
                queues[vertex.dataIndex].PushQueue(std::pair<size_t, size_t>(vertex.dataIndex, nextNeighbor.first));
            }    
        }
    }
}

/*
Computes an entire iteration of NND
*/
template<typename DataType>
int ComputeLocalJoins(const MNISTData& dataSource,
                       Graph<DataType>& graphState, 
                       std::vector<ComparisonQueue>& joinQueues, 
                       std::vector<ComparisonQueue>& cmpQueues, 
                       SpaceMetric<std::valarray<unsigned char>> distanceFunctor){

    NeighborSearchFunctor searchFunctor;
    int neighborListChanges(0);
    for (auto& joinQueue : joinQueues){
        for(auto& joinTarget : joinQueue.queue){
            bool pushToCmp = false;
            double distance = 0;
            searchFunctor.searchValue = joinTarget.first;
            //Check to see if A is already a neighbor of B, if so, bingo
            auto result = std::find_if(graphState[joinTarget.second].neighbors.begin(), graphState[joinTarget.second].neighbors.end(), searchFunctor);
            if (result != graphState[joinTarget.second].neighbors.end()){
                distance = result->second;
            } else{
            //I'm like 99% sure the first target cannot have the second in it's list of neighbors here unless I screwed up queuing the
            //join
                distance = distanceFunctor(
                    graphState[joinTarget.first].dataReference,
                    graphState[joinTarget.second].dataReference);
                
                if (distance < graphState[joinTarget.second].neighbors[0].second){
                    graphState[joinTarget.second].PushNeigbor(std::pair<size_t, double>(joinTarget.first, distance));
                    neighborListChanges++;
                    pushToCmp = true;
                }
            }
            if (distance < graphState[joinTarget.first].neighbors[0].second){
                graphState[joinTarget.first].PushNeigbor(std::pair<size_t, double>(joinTarget.second, distance));
                neighborListChanges++;
                pushToCmp = true;
            }
            if(pushToCmp){
                cmpQueues[joinTarget.first].PushQueue(joinTarget);
                cmpQueues[joinTarget.second].PushQueue(joinTarget);
            }
        }
        joinQueue.FlushQueue();

    }   
    return neighborListChanges;
}


template<typename DataType>
void PopulateJoinQueueStates(const Graph<DataType>& graphState, std::vector<ComparisonQueue>& cmpQueues, std::vector<ComparisonQueue>& joinQueues){
    NeighborSearchFunctor searchFunctor;
    for(auto& cmpQueue : cmpQueues){
        for(const auto& cmpTarget : cmpQueue.queue){
            //Parse through the first Vertex's neighbors for comparisons with 
            //Todo: Remove this label/matching goto
            debug:
            for(const auto& neighbor : graphState[cmpTarget.first].neighbors){
                searchFunctor.searchValue = neighbor.first;
                auto result = std::find_if(graphState[cmpTarget.second].neighbors.begin(), graphState[cmpTarget.second].neighbors.end(), searchFunctor);
                if (result != graphState[cmpTarget.second].neighbors.end()) continue;

                joinQueues[cmpTarget.second].PushQueue(std::pair<size_t, size_t>(cmpTarget.second, neighbor.first));
            }
            for(const auto& neighbor : graphState[cmpTarget.second].neighbors){
                searchFunctor.searchValue = neighbor.first;
                auto result = std::find_if(graphState[cmpTarget.first].neighbors.begin(), graphState[cmpTarget.first].neighbors.end(), searchFunctor);
                if (result != graphState[cmpTarget.first].neighbors.end()) continue;

                joinQueues[cmpTarget.first].PushQueue(std::pair<size_t, size_t>(cmpTarget.first, neighbor.first));
            }
            /*
            if((joinQueues[cmpTarget.first].queue.size() == 0) || (joinQueues[cmpTarget.second].queue.size() == 0)){
                goto debug;
            };
            */
        }
        cmpQueue.FlushQueue();
    }
}

//Mainly For Debugging to make sure I didn't screw up my graph state.
template<typename DataType>
void VerifyGraphState(const Graph<DataType>& currentGraph){
    for (const auto& vertex : currentGraph){
        for (const auto& neighbor : vertex.neighbors){
            if (neighbor.first == vertex.dataIndex) throw("Vertex is own neighbor");
            for (const auto& neighbor1 : vertex.neighbors){
                if (&neighbor == &neighbor1) continue;
                if (neighbor.first == neighbor.second) throw("Duplicate neighbor in heap");
            }
        }
    }
}


}
#endif //NND_ALGORITHM_HPP