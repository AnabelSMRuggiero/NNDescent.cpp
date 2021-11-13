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