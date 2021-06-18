#ifndef NND_GRAPHSTRUCTURES_HPP
#define NND_GRAPHSTRUCTURES_HPP
#include <vector>
#include <valarray>
#include <algorithm>
#include <functional>
#include "UtilityFunctions.hpp"
#include "MNISTData.hpp"
#include "SpaceMetrics.hpp"
#include "RNG.hpp"

namespace nnd{

template<typename DataType>
struct GraphVertex{

    const size_t dataIndex;
    
    const std::valarray<DataType>& dataReference;
    //The first in the pair is the index, the second is the distance
    std::vector<std::pair<size_t,double>> neighbors;
    //std::vector<size_t> reverseNeighbor;

    GraphVertex() : dataIndex(-1), dataReference(std::valarray<unsigned char>()), neighbors(0){};

    GraphVertex(size_t sourceIndex, const std::valarray<unsigned char>& sourceData, size_t numNeighbors):
        dataIndex(sourceIndex), dataReference(sourceData), neighbors(0) {
            this->neighbors.reserve(numNeighbors + 1);
        };

    GraphVertex(GraphVertex&& rval):
        dataIndex(rval.dataIndex), dataReference(rval.dataReference), neighbors(rval.neighbors){}

    /*
    template<typename DataType>
    std::valarray<DataType> VertexLocation(const std::valarray<DataType>& dataSource){
        return dataSource[dataReference];
    }
    */

    void PushNeigbor(std::pair<size_t,double>& newNeighbor){
        neighbors.push_back(newNeighbor);
        std::push_heap(neighbors.begin(), neighbors.end(), NeighborDistanceComparison);
        neighbors.pop_back();
    }

    //private:
    
};

template<typename DataType>
using Graph = std::vector<GraphVertex<DataType>>;

//Operator() of rngFunctor must return a random size_t in [0, data.size())
template<typename DataType>
Graph<DataType> ConstructInitialGraph(const MNISTData& dataSource, size_t numNeighbors, std::function<size_t()> rngFunctor, SpaceMetric<std::valarray<unsigned char>> distanceFunctor){
    NeighborSearchFunctor searchFunctor;
    Graph<DataType> retGraph(0);
    retGraph.reserve(dataSource.numberOfSamples);

    for (size_t i = 0; i<dataSource.numberOfSamples; i+=1){
        //std::slice vertexSlice(0, dataSource.vectorLength, 1);

        retGraph.push_back(GraphVertex<DataType>(i, dataSource.samples[i], numNeighbors));
        
    }

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

    return std::move(retGraph);
};


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

    void PushQueue(std::pair<size_t, size_t>& indecies){
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
    return std::move(retQueues);
}


}
#endif //DATASTRUCTURES