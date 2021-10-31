/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/


#include <filesystem>

#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Type.hpp"
#include "Utilities/Data.hpp"

#include "NND/Type.hpp"
#include "NND/SubGraphQuerying.hpp"
using namespace nnd;


struct FragmentMetaData{
    size_t numBlocks;
};

template<typename DelayType, typename Functor>
struct DelayConstructHelper{
    
    Functor&& func;
    
    

    operator DelayType() noexcept(noexcept(func())){
        return func();
    }
};

template<typename DelayType, std::invocable<> Functor>
    requires std::is_constructible_v<DelayType, std::invoke_result_t<Functor>>
auto DelayConstruct(Functor&& func) noexcept {
    return DelayConstructHelper<DelayType, Functor>{std::forward<Functor>(func)};
}

template<typename DelayType, typename... Ts>
    requires std::is_constructible_v<DelayType, Ts...>
auto DelayConstruct(Ts&&... ts) noexcept {
    auto constructor = [&](){
        return DelayType(std::forward<Ts>(ts)...);
    };
    return DelayConstruct<DelayType>(constructor);
}



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

constexpr bool test = ExtractableClass<GraphVertex<unsigned int, float>, std::ifstream&>;

int main(int argc, char *argv[]){

    std::filesystem::path indexLocation("./Saved-Indecies/MNIST-Fashion");


    std::string testDataFilePath("./TestData/MNIST-Fashion-Test.bin");
    std::string testNeighborsFilePath("./TestData/MNIST-Fashion-Neighbors.bin");
    DataSet<float> mnistFashionTest(testDataFilePath, 28*28, 10'000);
    DataSet<uint32_t, alignof(uint32_t)> mnistFashionTestNeighbors(testNeighborsFilePath, 100, 10'000);
    
    auto dataBlocks = OpenDataBlocks<float>(indexLocation);

    auto queryContexts = OpenQueryContexts<DataIndex_t, float>(indexLocation);

    auto indexBlocks = OpenIndexBlocks(indexLocation);

    return 0;
}