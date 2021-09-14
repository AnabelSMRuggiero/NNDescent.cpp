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



using namespace nnd;




template<typename Sinkee>
void Sink(Sinkee&& objToSink){
    Sinkee consume(std::move(objToSink));
};

int main(int argc, char *argv[]){

    static const std::endian dataEndianness = std::endian::native;

    std::string trainDataFilePath("./TestData/NYTimes-Angular-Train.bin");
    DataSet<AlignedArray<float>> nytimesTrain(trainDataFilePath, 256, 290'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);

    NormalizeDataSet(nytimesTrain);

    return 0;
}