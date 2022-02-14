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
#include <numeric>
#include "ann/MemoryResources.hpp"
#include "ann/AlignedMemory/AlignedAllocator.hpp"
#include "Parallelization/ThreadPool.hpp"


using namespace nnd;

int main(){

    {
        ChatterResource squaker;
        UnboundedCache testCache(&squaker);

        {
            std::pmr::vector<double> moreTest(45, &testCache);
            std::pmr::vector<int> evenMoreTest(49, &testCache);
        }
        

        {
            std::pmr::vector<std::byte> testVec1(19, &testCache);

            std::pmr::vector<size_t> testVec2(20, &testCache);

            std::pmr::vector<float> testVec3(15, &testCache);

            std::pmr::vector<std::byte> testVec4(666, &testCache);
            std::pmr::vector<unsigned int> testVec5(31, &testCache);
            
        }

        std::pmr::vector<int> testVec6(13, &testCache);

        {
            std::pmr::vector<float> testVec7(5, &testCache);
        }

    }

    {
        using namespace ann::udl;
        std::vector<double, ann::aligned_allocator<double, 32_a>> test1(32);
        std::vector<double, ann::aligned_allocator<double, 32_a>> test2(32);

        std::iota(test1.begin(), test1.end(), double{0.0});

        



    }

    return 0;
}