/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_ALGORITHM_HPP
#define NND_ALGORITHM_HPP
#include <algorithm>
#include "GraphStructures.hpp"

namespace nnd{

// This needs to be updated before use

//Placeholder until proper initialiation strat is implemented
//Simply queues up all possible local joins for each point
/*
template<typename DataType, TriviallyCopyable IndexType, typename FloatType>
void PopulateInitialQueueStates(const Graph<IndexType, FloatType>& graphState, std::vector<ComparisonQueue<std::pair<size_t,size_t>>>& queues){
    for(const auto& vertex : graphState){
        for(const auto& neighbor : vertex.neighbors){
            for(const auto& nextNeighbor : graphState[neighbor.first].neighbors){
                queues[vertex.dataIndex].PushQueue(std::pair<IndexType, IndexType>(vertex.dataIndex, nextNeighbor.first));
            }    
        }
    }
}
*/
/*
Computes an entire iteration of NND
*/
template<typename DataType, TriviallyCopyable IndexType, typename FloatType>
int ComputeLocalJoins(const MNISTData& dataSource,
                       Graph<IndexType, FloatType>& graphState, 
                       std::vector<ComparisonQueue<std::pair<size_t,size_t>>>& joinQueues, 
                       std::vector<ComparisonQueue<std::pair<size_t,size_t>>>& cmpQueues, 
                       SpaceMetric<std::valarray<unsigned char>> distanceFunctor){

    NeighborSearchFunctor<IndexType, FloatType> searchFunctor;
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
                    graphState[joinTarget.second].PushNeigbor(std::pair<IndexType, double>(joinTarget.first, distance));
                    neighborListChanges++;
                    pushToCmp = true;
                }
            }
            if (distance < graphState[joinTarget.first].neighbors[0].second){
                graphState[joinTarget.first].PushNeigbor(std::pair<IndexType, double>(joinTarget.second, distance));
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


template<typename FloatType>
void PopulateJoinQueueStates(const Graph<size_t, FloatType>& graphState, std::vector<ComparisonQueue<std::pair<size_t,size_t>>>& cmpQueues, std::vector<ComparisonQueue<std::pair<size_t,size_t>>>& joinQueues){
    NeighborSearchFunctor<size_t, FloatType> searchFunctor;
    for(auto& cmpQueue : cmpQueues){
        for(const auto& cmpTarget : cmpQueue.queue){
            //Parse through the first Vertex's neighbors for comparisons with 
            
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
            
        }
        cmpQueue.FlushQueue();
    }
}

//Mainly For Debugging to make sure I didn't screw up my graph state.
template<typename DataType, typename FloatType>
void VerifyGraphState(const Graph<size_t, FloatType>& currentGraph){
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