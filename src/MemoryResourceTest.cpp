/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <cstddef>
#include <vector>
#include "Utilities/MemoryResources.hpp"

using namespace nnd;

int main(){

    {
        ChatterResource squaker;
        MemoryCache testCache(&squaker);

        

        {
            std::pmr::vector<std::byte> testVec1(19, &testCache);

            std::pmr::vector<size_t> testVec2(20, &testCache);

            std::pmr::vector<float> testVec3(15, &testCache);

            
        }

            std::pmr::vector<int> testVec4(13, &testCache);

    }


    return 0;
}