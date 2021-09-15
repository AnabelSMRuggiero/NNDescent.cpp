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

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"

using namespace nnd;




template<typename Sinkee>
void Sink(Sinkee&& objToSink){
    Sinkee consume(std::move(objToSink));
};

int main(int argc, char *argv[]){

    SplittingHeurisitcs firstSplitParams= {125'000, 75'000, 175'000, 0.4f};


    static const std::endian dataEndianness = std::endian::native;

    std::string trainDataFilePath("./TestData/SIFT-Train.bin");
    DataSet<AlignedArray<float>> trainData(trainDataFilePath, 128, 1'000'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);


    std::string testDataFilePath("./TestData/SIFT-Test.bin");
    std::string testNeighborsFilePath("./TestData/SIFT-Neighbors.bin");
    DataSet<AlignedArray<float>> testData(testDataFilePath, 128, 10'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);
    DataSet<AlignedArray<uint32_t>> testNeighbors(testNeighborsFilePath, 100, 10'000, &ExtractNumericArray<AlignedArray<uint32_t>,dataEndianness>);

    auto [forest, splittingVectors] = BuildRPForest<EuclidianScheme<AlignedArray<float>, AlignedArray<float>>>(std::execution::seq, trainData, firstSplitParams);

    //
    std::vector<std::unique_ptr<size_t[]>> subSections;
    auto indexArrayMaker = [&subSections](const size_t, std::span<const size_t> indecies){
        std::unique_ptr<size_t[]>subSection = std::make_unique<size_t[]>(indecies.size());
        std::copy(indecies.begin(), indecies.end(), subSection.get());
        subSections.push_back(std::move(subSection));
    };

    CrawlTerminalLeaves(forest, indexArrayMaker);

    return 0;
}