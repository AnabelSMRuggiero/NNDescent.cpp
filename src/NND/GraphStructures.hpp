#ifndef NND_GRAPHSTRUCTURES_HPP
#define NND_GRAPHSTRUCTURES_HPP
#include <vector>
#include <valarray>
#include <algorithm>
#include <functional>
#include <ranges>

#include "UtilityFunctions.hpp"
#include "MNISTData.hpp"
#include "SpaceMetrics.hpp"
#include "RNG.hpp"

#include "Utilities/DataDeserialization.hpp"

namespace nnd{

template<TriviallyCopyable IndexType, typename FloatType>
struct GraphVertex{

    // I mainly use the dataIndex here for debuging/verification and in a placeholder
    // I should be able to refactor this point in a later pass
    //const size_t dataIndex;
    
    //I shouldn't even have this here. This is a reference into a vector currently!
    //const std::valarray<NumericType>& dataReference;
    //The first in the pair is the index, the second is the distance
    std::vector<std::pair<IndexType, FloatType>> neighbors;
    //std::vector<size_t> reverseNeighbor;

    GraphVertex() : neighbors(0){};

    GraphVertex(size_t numNeighbors):neighbors(0) {
        this->neighbors.reserve(numNeighbors + 1);
    };

    GraphVertex(GraphVertex&& rval): neighbors(std::forward<std::vector<std::pair<IndexType, FloatType>>>(rval.neighbors)){};

    /*
    template<typename DataType>
    std::valarray<DataType> VertexLocation(const std::valarray<DataType>& dataSource){
        return dataSource[dataReference];
    }
    */

    void PushNeigbor(std::pair<IndexType, FloatType> newNeighbor){
        neighbors.push_back(newNeighbor);
        std::pop_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison<IndexType, FloatType>);
        neighbors.pop_back();
    };

    //private:
    
};

template<TriviallyCopyable IndexType, typename FloatType>
using Graph = std::vector<GraphVertex<IndexType, FloatType>>;

//Operator() of rngFunctor must return a random size_t in [0, data.size())
template<TriviallyCopyable IndexType, typename FloatType>
Graph<IndexType, FloatType> ConstructInitialGraph(size_t numVerticies, size_t numNeighbors){
    Graph<IndexType, FloatType> retGraph(0);
    retGraph.reserve(numVerticies);

    for (size_t i = 0; i<numVerticies; i+=1){
        //std::slice vertexSlice(0, dataSource.vectorLength, 1);

        retGraph.push_back(GraphVertex<IndexType, FloatType>(numNeighbors));
        
    }
    /*
    //Temporary method for initialization of graph neighbors. This will likely be replaced by rp-tree construction
    for (size_t i = 0; i<retGraph.size(); i+=1){
        for (size_t j = 0; j<numNeighbors; j += 1){
            size_t randomIndex(rngFunctor());
            while(true){
                if (randomIndex == i) randomIndex += (i == 0) ? 1 : -1;
                searchFunctor.searchValue = randomIndex;
                //Check to see if A is already a neighbor of B, if so, bingo
                auto result = std::find_if(retGraph[randomIndex].neighbors.begin(), retGraph[randomIndex].neighbors.end(), searchFunctor);
                if (result == retGraph[randomIndex].neighbors.end()) break;
                randomIndex = rngFunctor();
            }
            double distance = distanceFunctor(
                    retGraph[i].dataReference,
                    retGraph[randomIndex].dataReference);

            retGraph[i].neighbors.emplace_back(randomIndex, distance);

            //std::pair<size_t, double> ()
        }
        std::make_heap(retGraph[i].neighbors.begin(), retGraph[i].neighbors.end(), NeighborDistanceComparison);
    }
    */
    return retGraph;
};

template<typename DataType, typename FloatType>
void BruteForceBlock(Graph<BlockIndex, FloatType>& uninitGraph, size_t numNeighbors, const DataBlock<DataType>& dataBlock, SpaceMetric<DataType, FloatType> distanceFunctor){
    
    // I can make this branchless. Check to see if /O2 or /O3 can make this branchless (I really doubt it)
    for (size_t i = 0; i < dataBlock.blockData.size(); i += 1){
        for (size_t j = i+1; j < dataBlock.blockData.size(); j += 1){
            FloatType distance = distanceFunctor(dataBlock.blockData[i], dataBlock.blockData[j]);
            if (uninitGraph[i].neighbors.size() < numNeighbors){
                uninitGraph[i].neighbors.push_back(std::pair<BlockIndex, FloatType>(BlockIndex(dataBlock.blockNumber, j), distance));
                if (uninitGraph[i].neighbors.size() == numNeighbors){
                    std::make_heap(uninitGraph[i].neighbors.begin(), uninitGraph[i].neighbors.end(), NeighborDistanceComparison<BlockIndex, FloatType>);
                }
            } else if (distance < uninitGraph[i].neighbors[0].second){
                uninitGraph[i].PushNeigbor(std::pair<BlockIndex, FloatType>(BlockIndex(dataBlock.blockNumber, j), distance));
            }
            if (uninitGraph[j].neighbors.size() < numNeighbors){
                uninitGraph[j].neighbors.push_back(std::pair<BlockIndex, FloatType>(BlockIndex(dataBlock.blockNumber, i), distance));
                if (uninitGraph[j].neighbors.size() == numNeighbors){
                    std::make_heap(uninitGraph[j].neighbors.begin(), uninitGraph[j].neighbors.end(), NeighborDistanceComparison<BlockIndex, FloatType>);
                }
            } else if (distance < uninitGraph[j].neighbors[0].second){
                uninitGraph[j].PushNeigbor(std::pair<BlockIndex, FloatType>(BlockIndex(dataBlock.blockNumber, i), distance));
            }
        }
    }
}

/*
Queue that accepts up to a maxmimum number of elements
Once the max is reached, new elements are added based on the number of total elements
That have been attempted to be added since the last time the queue was flushed
Currently, storing the index of the pointA neighbor is redundant, but the plan is to
associate a queue for blocks of data, not individual vertecies.
*/
struct ComparisonQueue{

    std::vector<std::pair<size_t, size_t>> queue;
    size_t queueMaxLength;
    size_t queueWeight;
    std::vector<std::pair<size_t, size_t>>::iterator ringIterator;
    StlRngFunctor<std::mt19937_64, std::uniform_real_distribution, float> rngFunctor;

    ComparisonQueue():
        queue(0),
        queueMaxLength(0),
        queueWeight(0),
        rngFunctor(std::mt19937_64(0), std::uniform_real_distribution<float>()){
            ringIterator = queue.begin();
        }

    ComparisonQueue(size_t maxLength):
        queue(0),
        queueMaxLength(maxLength),
        queueWeight(0),
        rngFunctor(std::mt19937_64(0), std::uniform_real_distribution<float>()){
            queue.reserve(maxLength);
            ringIterator = queue.begin();
        }
    
    ComparisonQueue(ComparisonQueue&& rhsQueue):
        queue(std::move(rhsQueue.queue)),
        queueMaxLength(rhsQueue.queueMaxLength),
        queueWeight(rhsQueue.queueWeight),
        rngFunctor(std::move(rhsQueue.rngFunctor)),
        ringIterator(std::move(rhsQueue.ringIterator)){};

    //ComparisonQueue& operator=(const ComparisonQueue& rhs)

    void PushQueue(const std::pair<size_t, size_t>& indecies){
        queueWeight+=1;
        if (queue.size() < queueMaxLength){
            queue.push_back(indecies);
            return;
        } else {
            /*
            size_t rand = size_t(rngFunctor()*queueWeight);
            if (rand<queueMaxLength) queue[rand] = indecies;
            return;
            */
            *ringIterator = indecies;
            ringIterator++;
            if (ringIterator == queue.end()) ringIterator = queue.begin();
            return;
        }
    }

    void FlushQueue(){
        queue.clear();
        queueWeight = 0;
        ringIterator = queue.begin();
    }

};


std::vector<ComparisonQueue> ConstructQueues(size_t numQueues, size_t queueMax){

    std::vector<ComparisonQueue> retQueues(0);
    retQueues.reserve(numQueues);
    //ComparisonQueue copyQueue(queueMax);
    //retQueues.insert(retQueues.begin(), {queueMax});
    //retQueues.push_back(std::move(copyQueue));
    for (size_t i = 0; i < numQueues; i +=1){
        retQueues.emplace_back(queueMax);
    }
    return retQueues;
}


}
#endif //DATASTRUCTURES