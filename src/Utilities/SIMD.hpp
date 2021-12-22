/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_POINTERMANIPULATION_HPP
#define NND_POINTERMANIPULATION_HPP

namespace nnd {

enum struct InstructionSet{
    //X64
    avx,
    avx2,
    fma,
    avx512,
    //ARM... maybe at some point.
    neon,
    //Catch all
    unknown
};

consteval InstructionSet DetectInstructionSet(){
    #ifdef __AVX512__
        return InstructionSet::avx512;
    #elif defined __FMA__
        return InstructionSet::fma;
    #elif defined __AVX2__
        return InstructionSet::avx2;
    #elif __AVX__
        return InstructionSet::avx;
    #else
        return InstructionSet::unknown;
    #endif
}

using enum InstructionSet;

constexpr InstructionSet defaultInstructionSet = DetectInstructionSet();

template<InstructionSet set>
constexpr size_t vectorWidth = 0;

template<>
constexpr size_t vectorWidth<avx512> = 64;

template<>
constexpr size_t vectorWidth<fma> = 32;

template<>
constexpr size_t vectorWidth<avx2> = 32;

template<>
constexpr size_t vectorWidth<avx> = 32;

template<typename Derived>
struct DataVectorBase{
    using DerivedClass = Derived;
};


    
}

#endif