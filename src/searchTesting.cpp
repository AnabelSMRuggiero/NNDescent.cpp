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
#include <utility>
#include <vector>

#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/Data.hpp"
#include "ann/DataDeserialization.hpp"
#include "ann/DelayConstruct.hpp"
#include "ann/Metrics/Euclidean.hpp"
#include "ann/Metrics/SpaceMetrics.hpp"
#include "ann/Type.hpp"

#include "NND/FunctorErasure.hpp"
#include "NND/Index.hpp"
#include "NND/MetaGraph.hpp"
#include "NND/MetricHelpers.hpp"
#include "NND/Parallel-Algorithm/FreeFunctions.hpp"
#include "NND/Search.hpp"
#include "NND/SubGraphQuerying.hpp"
#include "NND/Type.hpp"

#include "RPTrees/Forest.hpp"
#include "RPTrees/SplittingScheme.hpp"

using namespace nnd;




namespace nnd {






} // namespace nnd

int main(int argc, char* argv[]) {

    constexpr size_t numThreads = 12;



    SearchParameters search_parameters{ 10, 6, 10 };


    bool parallelSearch = true;


    /*
    std::filesystem::path indexLocation("./Saved-Indecies/SIFT");

    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<float> test_data_set(testDataFilePath, 128, 10'000);
    DataSet<std::uint32_t, ann::align_val_of<std::uint32_t>> test_neighbors(testNeighborsFilePath, 100, 10'000);
    using metric = euclidean_metric_pair;
    */
    
    std::filesystem::path indexLocation("./Saved-Indecies/NYTimes");
    
    std::string testDataFilePath("./TestData/NYTimes-Angular-Test.bin");
    std::string testNeighborsFilePath("./TestData/NYTimes-Angular-Neighbors.bin");
    DataSet<float> test_data_set(testDataFilePath, 128, 10'000);
    DataSet<std::uint32_t, ann::align_val_of<std::uint32_t>> test_neighbors(testNeighborsFilePath, 100, 10'000);
    using metric = inner_product_pair;
    

    nnd::index<float> index = open_index<float>(indexLocation);
    index.search_parameters = search_parameters;
    fixed_block_binder searchDist(metric{}, test_data_set, std::span<const DataBlock<float>>{ std::as_const(index.data_points) });
    erased_unary_binder<float> searchFunctor(searchDist);

    for (auto& context : index.query_contexts) {
        context.querySearchDepth = index.search_parameters.searchDepth;
        context.querySize = index.search_parameters.searchNeighbors;
    }

    index.distance_metric = searchFunctor;


    for (size_t i = 0; i < 10; i += 1) {
        std::chrono::time_point<std::chrono::steady_clock> runStart2 = std::chrono::steady_clock::now();

        auto results = parallelSearch 
                        ? nnd::search(test_data_set, index, numThreads)
                        : nnd::search(test_data_set, index);
        

        std::chrono::time_point<std::chrono::steady_clock> runEnd2 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::duration<float>>(runEnd2 - runStart2).count() << std::endl;

        size_t numNeighborsCorrect(0);
        std::vector<size_t> correctNeighborsPerIndex(results.size());
        for (size_t i = 0; const auto& result : results) {
            for (size_t j = 0; const auto& neighbor : result) {
                auto findItr = std::find(std::begin(test_neighbors[i]), std::begin(test_neighbors[i]) + 10, neighbor);
                if (findItr != (std::begin(test_neighbors[i]) + 10)) {
                    numNeighborsCorrect++;
                    correctNeighborsPerIndex[i]++;
                }
                j++;
            }
            i++;
        }

        double recall = double(numNeighborsCorrect) / double(10 * test_neighbors.size());
        std::cout << (recall * 100) << std::endl;
    }
    return 0;
}