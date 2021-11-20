/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/


#include <filesystem>
#include <unordered_map>
#include <unordered_set>

#include "Utilities/DelayConstruct.hpp"
#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"
#include "Utilities/Metrics/Euclidean.hpp"

#include "NND/MetaGraph.hpp"
#include "NND/Type.hpp"
#include "NND/SubGraphQuerying.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"
#include "NND/Search.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

using namespace nnd;


struct FragmentMetaData{
    size_t numBlocks;
};




template<typename DistType>
std::vector<DataBlock<DistType>> OpenDataBlocks(std::filesystem::path fragmentDirectory){
    std::ifstream metaDataFile{fragmentDirectory / "MetaData.bin", std::ios_base::binary};
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<DataBlock<DistType>> dataBlocks;
    dataBlocks.reserve(metadata.numBlocks);

    for (size_t i = 0; i<metadata.numBlocks; i+=1){
        std::filesystem::path dataBlockPath = fragmentDirectory / ("DataBlock-" + std::to_string(i) + ".bin");
        std::ifstream dataBlockFile{dataBlockPath, std::ios_base::binary};
        auto extractor = [&](){
            return Extract<DataBlock<DistType>>(dataBlockFile, i);
        };
        dataBlocks.emplace_back(DelayConstruct<DataBlock<DistType>>(extractor));
    }
    return dataBlocks;
}

template<typename IndexType, typename DistType>
std::vector<QueryContext<IndexType, DistType>> OpenQueryContexts(std::filesystem::path fragmentDirectory){
    std::ifstream metaDataFile{fragmentDirectory / "MetaData.bin", std::ios_base::binary};
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<QueryContext<IndexType, DistType>> queryContexts;
    queryContexts.reserve(metadata.numBlocks);

    for (size_t i = 0; i<metadata.numBlocks; i+=1){
        std::filesystem::path dataBlockPath = fragmentDirectory / ("QueryContext-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{dataBlockPath, std::ios_base::binary};
        auto extractor = [&](){
            return Extract<QueryContext<IndexType, DistType>>(contextFile);
        };
        queryContexts.emplace_back(DelayConstruct<QueryContext<IndexType, DistType>>(extractor));
        //queryContexts.emplace_back(QueryContext<IndexType, DistType>(contextFile));
    }
    return queryContexts;
}


std::vector<IndexBlock> OpenIndexBlocks(std::filesystem::path fragmentDirectory){
    std::ifstream metaDataFile{fragmentDirectory / "MetaData.bin", std::ios_base::binary};
    FragmentMetaData metadata = Extract<FragmentMetaData>(metaDataFile);

    std::vector<IndexBlock> queryContexts;
    queryContexts.reserve(metadata.numBlocks);

    for (size_t i = 0; i<metadata.numBlocks; i+=1){
        std::filesystem::path dataBlockPath = fragmentDirectory / ("IndexBlock-" + std::to_string(i) + ".bin");
        std::ifstream contextFile{dataBlockPath, std::ios_base::binary};
        auto extractor = [&](){
            return Extract<IndexBlock>(contextFile);
        };
        queryContexts.emplace_back(DelayConstruct<IndexBlock>(extractor));
    }
    return queryContexts;
}

int main(int argc, char *argv[]){

    constexpr size_t numThreads = 12;

    //IndexParamters indexParams{12, 40, 35, 6};
    IndexParameters indexParams{12, 20, 15, 6};

    size_t numBlockGraphNeighbors = 12;
    //size_t numCOMNeighbors = 40;
    //size_t maxNearestNodes = 35;
    size_t numCOMNeighbors = 20;
    size_t maxNearestNodes = 15;
    size_t queryDepth = 6;

    SearchParameters searchParams{10, 6, 5};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 6;
    size_t maxNewSearches = 10;

    SplittingHeurisitcs splitParams= {2500, 1500, 3500, 0.0f};
    //SplittingHeurisitcs splitParams= {205, 123, 287, 0.0f};

    //SplittingHeurisitcs splitParams= {20, 12, 28, 0.0f};

    size_t additionalInitSearches = 8;

    HyperParameterValues parameters{splitParams, indexParams, searchParams};
    
    bool parallelIndexBuild = true;
    bool parallelSearch = true;

    /*
    std::filesystem::path indexLocation("./Saved-Indecies/MNIST-Fashion");


    std::string testDataFilePath("./TestData/MNIST-Fashion-Test.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<float> mnistFashionTest(testDataFilePath, 28*28, 10'000);
    DataSet<uint32_t, alignof(uint32_t)> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000);
    */

    std::filesystem::path indexLocation("./Saved-Indecies/SIFT");
    
    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<float> mnistFashionTest(testDataFilePath, 128, 10'000);
    DataSet<uint32_t, alignof(uint32_t)> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000);
    

    auto dataBlocks = OpenDataBlocks<float>(indexLocation);

    auto queryContexts = OpenQueryContexts<DataIndex_t, float>(indexLocation);

    auto indexBlocks = OpenIndexBlocks(indexLocation);
    
    auto splittingVectors = [&](){
        std::filesystem::path splittingVectorsPath = indexLocation / "SplittingVectors.bin";
        std::ifstream vecFile(splittingVectorsPath, std::ios_base::binary);
        return extract<std::unordered_map<size_t, std::pair<AlignedArray<float>, float>>>(vecFile);
    }();

    auto splitToBlockNum = [&](){
        std::ifstream mappingFile(indexLocation / "SplittingIndexToBlockNumber.bin", std::ios_base::binary);
        return extract<std::unordered_map<size_t, size_t>>(mappingFile);
    }();


    auto blockIndexToSource = [&](){
        std::ifstream mappingFile(indexLocation / "BlockIndexToSourceIndex.bin", std::ios_base::binary);
        return extract<std::vector<std::vector<size_t>>>(mappingFile);
    }();
    std::unordered_set<size_t> splittingIndicies;
    for (const auto& [key, value] : splittingVectors) splittingIndicies.insert(key);

    EuclidianScheme<float, AlignedArray<float>> transformingScheme(mnistFashionTest);

    transformingScheme.splittingVectors = std::move(splittingVectors);

    for (size_t i = 0; i<10; i+=1){
        std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

        

        
    
        RandomProjectionForest rpTreesTest = (parallelSearch) ?
                                             RPTransformData(std::execution::par_unseq ,mnistFashionTest, splittingIndicies, transformingScheme, numThreads):
                                             RPTransformData(mnistFashionTest, splittingIndicies, transformingScheme) ;
                                         
    

        size_t numberSearchBlocks = dataBlocks.size();


        for (auto& context: queryContexts){
            context.querySearchDepth = searchQueryDepth;
            context.querySize = numberSearchNeighbors;
        }

        SearchFunctor<float, DataSet<float>, EuclideanMetricPair> searchDist(dataBlocks, mnistFashionTest);
        SinglePointFunctor<float> searchFunctor(searchDist);

    
    
        std::vector<std::vector<size_t>> results(mnistFashionTest.size());
        IndexMaps<size_t> testMappings;

    

        if(parallelSearch){

        

            ThreadPool<SinglePointFunctor<float>> searchPool(numThreads, searchDist);

            DataMapper<float, void, void> testMapper(mnistFashionTest);
            std::vector<ParallelContextBlock<float>> searchContexts;

            auto searcherConstructor = [&, &splitToBlockNum=splitToBlockNum](size_t splittingIndex, std::span<const size_t> indicies)->void{
            
                ParallelContextBlock<float> retBlock{splitToBlockNum[splittingIndex], std::vector<ParallelSearchContext<float>>(indicies.size())};
            
                for(size_t i = 0; size_t index: indicies){
                    //const DataView searchPoint, const std::vector<DataBlock<DataEntry>>& blocks
                    ParallelSearchContext<float>* contextPtr = &retBlock.second[i];
                    contextPtr->~ParallelSearchContext<float>();
                    new (contextPtr) ParallelSearchContext<float>(numberSearchNeighbors, numberSearchBlocks, index);

                    //retVec.emplace_back(numberSearchNeighbors, numberSearchBlocks, index);
                    i++;
                }
                searchContexts.push_back(std::move(retBlock));
                testMapper(splittingIndex, indicies);
            };
            //DataMapper<AlignedArray<float>, std::vector<ParallelSearchContext<float>>, decltype(searcherConstructor)> testMapper(mnistFashionTest, searcherConstructor);
            CrawlTerminalLeaves(rpTreesTest, searcherConstructor);

            //auto searchContexts = std::move(testMapper.dataBlocks);

            testMappings = {
                std::move(testMapper.splitToBlockNum),
                std::move(testMapper.blockIndexToSource),
                std::move(testMapper.sourceToBlockIndex),
                std::move(testMapper.sourceToSplitIndex)
            };
    
    
            //auto blocksToSearch = BlocksToSearch(searchContexts, metaGraph, additionalInitSearches);
            //auto blocksToSearch = BlocksToSearch(metaGraph, additionalInitSearches);

            std::span<const IndexBlock> indexView{indexBlocks.data(), indexBlocks.size()};
            InitialSearchTask<float> searchGenerator = { queryContexts,
                indexView,
                //std::move(blocksToSearch),
                maxNewSearches,
                AsyncQueue<std::pair<BlockIndecies, SearchSet>>()};

            searchPool.StartThreads();
            SearchQueue searchHints = ParaFirstBlockSearch(searchGenerator, searchContexts, searchPool);
    
            QueueView hintView = {searchHints.data(), searchHints.size()};
        
            ParaSearchLoop(searchPool, hintView, searchContexts, std::span{queryContexts}, indexView, maxNewSearches, mnistFashionTest.size());
            searchPool.StopThreads();
            std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
            //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " << std::endl;
            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;

            std::span<std::vector<size_t>> indexMappingView(blockIndexToSource.data(), blockIndexToSource.size());

            for (size_t i = 0; auto& [blockNum, testBlock]: searchContexts){
                for (size_t j = 0; auto& context: testBlock){
                    GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
                    //size_t testIndex = testMappings.blockIndexToSource[i][j];
                    size_t testIndex = context.dataIndex;
                    //std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
                    for (const auto& neighbor: result){
                        results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
                    }
                    j++;
                }
                i++;
            }
        } else{
        
            DataMapper<float, void, void> testMapper(mnistFashionTest);
            std::vector<ContextBlock<float>> searchContexts;
            auto searcherConstructor = [&, &splitToBlockNum=splitToBlockNum](size_t splittingIndex, std::span<const size_t> indicies)->void{
            
                ContextBlock<float> retBlock;
                retBlock.first = splitToBlockNum[splittingIndex];
                retBlock.second.reserve(indicies.size());
                for(size_t index: indicies){
                    //const DataView searchPoint, const std::vector<DataBlock<DataEntry>>& blocks
                
                    retBlock.second.push_back({numberSearchNeighbors, numberSearchBlocks, index});
                }
                searchContexts.push_back(std::move(retBlock));
                testMapper(splittingIndex, indicies);
            };


        
            CrawlTerminalLeaves(rpTreesTest, searcherConstructor);

            //auto searchContexts = std::move(testMapper.dataBlocks);

            testMappings = {
                std::move(testMapper.splitToBlockNum),
                std::move(testMapper.blockIndexToSource),
                std::move(testMapper.sourceToBlockIndex),
                std::move(testMapper.sourceToSplitIndex)
            };
        
        
            //auto blocksToSearch = BlocksToSearch(searchContexts, metaGraph, additionalInitSearches);
        
            //OffsetSpan<const IndexBlock> indexSpan(index.data(), index.size(), metaGraph.GetBlockOffset());
            std::span<const IndexBlock> indexView{indexBlocks.data(), indexBlocks.size()};
            SearchQueue searchHints = FirstBlockSearch(searchContexts, searchFunctor, std::span{queryContexts}, indexView, maxNewSearches);
            //std::vector<IndexBlock> index = IndexFinalization(blockUpdateContexts)

            QueueView hintView = {searchHints.data(), searchHints.size()};

            SearchLoop(searchFunctor, hintView, searchContexts, std::span{queryContexts}, indexView, maxNewSearches, mnistFashionTest.size());
        

        

            std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
            //std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << "s test set search " << std::endl;
            std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;

            std::span<std::vector<size_t>> indexMappingView(blockIndexToSource.data(), blockIndexToSource.size());
        
            for (size_t i = 0; auto& [ignore, testBlock]: searchContexts){
                for (size_t j = 0; auto& context: testBlock){
                    GraphVertex<BlockIndecies, float>& result = context.currentNeighbors;
                    size_t testIndex = context.dataIndex;
                    //std::sort_heap(result.begin(), result.end(), NeighborDistanceComparison<BlockIndecies, float>);
                    for (const auto& neighbor: result){
                        results[testIndex].push_back(indexMappingView[neighbor.first.blockNumber][neighbor.first.dataIndex]);
                    }
                    j++;
                }
                i++;
            }
        
       
        }
    

   
    
    
    


        size_t numNeighborsCorrect(0);
        std::vector<size_t> correctNeighborsPerIndex(results.size());
        for(size_t i = 0; const auto& result: results){
            for(size_t j = 0; const auto& neighbor: result){
                auto findItr = std::find(std::begin(mnistFashionTestNeighbors[i]), std::begin(mnistFashionTestNeighbors[i]) + 10, neighbor);
                if (findItr != (std::begin(mnistFashionTestNeighbors[i]) + 10)){
                    numNeighborsCorrect++;
                    correctNeighborsPerIndex[i]++;
                }
                j++;
            }
            i++;
        }

        double recall = double(numNeighborsCorrect)/ double(10*mnistFashionTestNeighbors.size());
        std::cout << (recall * 100) << std::endl;
    }
    return 0;
}