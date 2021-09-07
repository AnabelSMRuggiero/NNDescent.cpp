/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <bit>
#include <memory>
#include <memory_resource>
#include <execution>
#include <functional>

#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"

#include "Parallelization/ThreadPool.hpp"

#include "NND/RNG.hpp"

#include "RPTrees/Forest.hpp"
#include "RPTrees/SplittingScheme.hpp"
#include "NND/MetaGraph.hpp"

using namespace nnd;


//Copying this here as the orginal is in a header that includes RPTrees
template<typename DataEntry>
std::pair<IndexMaps<size_t>, std::vector<DataBlock<DataEntry>>> PartitionData(const RandomProjectionForest& treeToMap, const DataSet<DataEntry>& sourceData){
    //There might be a more elegant way to do this with templates. I tried.
    auto boundContructor = [](const DataSet<DataEntry>& dataSource, std::span<const size_t> dataPoints, size_t blockNumber){ 
        return DataBlock<DataEntry>(dataSource, dataPoints, blockNumber);
    };
    
    DataMapper<DataEntry, DataBlock<DataEntry>, decltype(boundContructor)> dataMapper(sourceData, boundContructor);
    CrawlTerminalLeaves(treeToMap, dataMapper);
    
    std::vector<DataBlock<DataEntry>> retBlocks = std::move(dataMapper.dataBlocks);
    IndexMaps<size_t> retMaps = {
        std::move(dataMapper.splitToBlockNum),
        std::move(dataMapper.blockIndexToSource),
        std::move(dataMapper.sourceToBlockIndex),
        std::move(dataMapper.sourceToSplitIndex)
    };


    
    return {retMaps, std::move(retBlocks)};
}

template<typename Sinkee>
void Sink(Sinkee&& objToSink){
    Sinkee consume(std::move(objToSink));
};

int main(int argc, char *argv[]){

    static const std::endian dataEndianness = std::endian::big;

    static const size_t numThreads = 12;

    SplittingHeurisitcs splitParams= {16, 205, 123, 287};

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);

    // rngEngine(0);
    //std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    RngFunctor rngFunctor(0, mnistFashionTrain.size()-1);
    
    
    
    ParallelEuclidianScheme<AlignedArray<float>, AlignedArray<float>> splittingScheme(mnistFashionTrain);

    std::unique_ptr<size_t[]> indecies = std::make_unique<size_t[]>(mnistFashionTrain.size());
    std::iota(indecies.get(), indecies.get()+mnistFashionTrain.size(), 0);


    ForestBuilder builder{std::move(rngFunctor), splitParams, splittingScheme};

    /*
    std::execution::parallel_unsequenced_policy,
                                      std::unique_ptr<size_t[]>&& indecies, 
                                      const size_t numIndecies, 
                                      ThreadPool<void>& threadPool,
                                      std::pmr::memory_resource* upstream = std::pmr::get_default_resource()
    */

    std::pmr::synchronized_pool_resource memory;
    ThreadPool<TreeRef> pool(numThreads);
    pool.StartThreads();
    RandomProjectionForest rpTrees = builder(std::execution::par_unseq,
                                             std::move(indecies), 
                                             mnistFashionTrain.size(),
                                             pool,
                                             &memory);

    pool.StopThreads();


    //auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTrees, mnistFashionTrain);
    
    //Sink(std::move(rpTrees));

    return 0;
}