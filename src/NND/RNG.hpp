/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_RNG_HPP
#define NND_RNG_HPP
#include <random>

namespace nnd{

//The mersene twister engine from the stl is like 5kb (without optimizations)
//I kinda want something more lightweight, even if not as statistcally good.

//Just something simple and fast
//Generates an int [begin, end) by progressing through a bit permutation
/*
struct RNGFunctor{

    uint64_t seed;
    uint32_t rangeBegin;
    uint32_t rangeEnd;
    uint32_t rangeLength;

    RNGFunctor(int32_t begin, int32_t end):
        seed(chrono::steady_clock::now().time_since_epoch().count()),
        rangeBegin(begin),
        rangeEnd(end),
        rangeLength(end-begin) {
            if (begin >= end) throw(std::string("End must be greater than begin"));
        };

    int32_t operator()(){
        constexpr uint64_t topHalfMask = (4'294'967'296-1) << 32;
        constexpr uint64_t botHalfMask = (4'294'967'296-1);
        
        uint32_t seedTop = ((this->seed)&topHalfMask)>>32;
        uint32_t seedBot = (this->seed)&botHalfMask;


    }

};
*/




//template<typename Engine, typename Distribution, typename RetType = size_t>
struct RngFunctor{
    std::mt19937_64 functorEngine;
    std::uniform_int_distribution<size_t> functorDistribution;
    // (size_t{0}, mnistFashionTrain.size()-1)
   
    RngFunctor(size_t min, size_t max): functorEngine(0), functorDistribution(min, max){};
    //    functorEngine(std::move(engine)), functorDistribution(std::move(distribution)){};

    RngFunctor(const RngFunctor&) = default;
    
    void SetRange(size_t min, size_t max){
        functorDistribution.param(std::uniform_int_distribution<size_t>::param_type{min, max});
    }

    size_t operator()(){
        return functorDistribution(functorEngine);
    }

    
};

}

#endif //NND_RNG_HPP