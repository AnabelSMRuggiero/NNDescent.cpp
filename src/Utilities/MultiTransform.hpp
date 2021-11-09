/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_MULTITRANSFORM_HPP
#define NND_MULTITRANSFORM_HPP

#include <ranges>

namespace nnd{

template<typename T>
consteval size_t RangeRank(){
    return 0;
}

template<typename Range>
constexpr bool isNestedRange = std::ranges::range<std::ranges::range_value_t<Range>>;

template<std::ranges::range Range>
consteval size_t RangeRank(){
    return RangeRank<std::ranges::range_value_t<Range>>() + 1;
}

template<typename Range>
constexpr size_t rangeRank = RangeRank<Range>();

template<std::ranges::range Range>
void MultiTransform(){
    
}

}

#endif