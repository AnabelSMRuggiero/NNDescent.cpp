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




template<typename Engine = std::mt19937_64, template<typename> typename Distribution = std::uniform_int_distribution, typename RetType = size_t>
struct StlRngFunctor{
    Engine functorEngine;
    Distribution<RetType> functorDistribution;

    //Takes ownership of engine and distribution
    StlRngFunctor(Engine&& engine = std::mt19937_64(0), Distribution<RetType>&& distribution = std::uniform_int_distribution<RetType>(0)):
        functorEngine(std::move(engine)), functorDistribution(std::move(distribution)){};
        
    RetType operator()(){
        return functorDistribution(functorEngine);
    }

    
};

}

#endif //NND_RNG_HPP