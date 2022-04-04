/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <atomic>
#include <cstddef>
#include <memory_resource>
#include <random>
#include <thread>
#include <vector>
#include <numeric>
#include "ann/MemoryResources.hpp"
#include "ann/AlignedMemory/AlignedAllocator.hpp"
#include "Parallelization/ThreadPool.hpp"

#include "NND/RNG.hpp"


using namespace nnd;

int main(){

    {
        ann::ChatterResource squaker;
        ann::ShallowCache testCache(&squaker);

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

    {   
        std::size_t numthreads = 12;
        ThreadPool<RngFunctor> pool(numthreads, std::size_t{4}, std::size_t{14});
        //ThreadPool<RngFunctor> pool(numthreads, std::size_t{15}, std::size_t{16});
        
        std::pmr::monotonic_buffer_resource memoryIn(std::pmr::get_default_resource());
        ann::threaded_multipool here_goes_nothing{&memoryIn};

        std::atomic<std::size_t> side_effect = 0;

        std::atomic<bool> done = false;
        auto task = [&, resource = std::pmr::new_delete_resource()](RngFunctor& rng){
            RngFunctor pick_size{1, 124};
            std::seed_seq seed{};
            std::mt19937_64 random_bits{seed};
            while(!done){

                std::pmr::vector<std::pmr::vector<std::byte>> accumulator(resource);

                std::size_t number_iterations = pick_size();
                for (std::size_t i = 0; i < number_iterations; ++i){
                    auto largest_bit = std::size_t{1}<<rng();
                    auto mask = largest_bit - 1;

                    accumulator.emplace_back(largest_bit + mask & random_bits());
                    std::fill(accumulator.back().begin(), accumulator.back().end(), std::byte{4});
                    std::size_t temp = 0;
                    for (auto& byte : accumulator.back()){
                        temp += static_cast<std::size_t>(byte);
                    }
                    side_effect.fetch_add(temp, std::memory_order_relaxed);
                }
            }
        };

        auto big_task = [&, resource = std::pmr::new_delete_resource()](RngFunctor& rng){
            RngFunctor pick_size{64, 128*64};
            std::seed_seq seed{};
            std::mt19937_64 random_bits{seed};
            while(!done){

                std::pmr::vector<std::pmr::vector<std::byte>> accumulator(resource);

                std::size_t number_iterations = pick_size();
                for (std::size_t i = 0; i < number_iterations; ++i){
                    auto largest_bit = std::size_t{1}<<rng();
                    auto mask = largest_bit - 1;

                    accumulator.emplace_back(largest_bit + mask & random_bits());
                    std::fill(accumulator.back().begin(), accumulator.back().end(), std::byte{4});
                    std::size_t temp = 0;
                    for (auto& byte : accumulator.back()){
                        temp += static_cast<std::size_t>(byte);
                    }
                    side_effect.fetch_add(temp, std::memory_order_relaxed);
                }
            }
        };

        auto greedy_task = [&, resource = &here_goes_nothing](RngFunctor& rng){
            std::pmr::vector<std::pmr::vector<std::byte>> accumulator(resource);

            while(!done){

                accumulator.emplace_back(std::size_t{1}<<rng());
                std::fill(accumulator.back().begin(), accumulator.back().end(), std::byte{4});
                std::size_t temp = 0;
                for (auto& byte : accumulator.back()){
                    temp += static_cast<std::size_t>(byte);
                }
                side_effect.fetch_add(temp, std::memory_order_relaxed);
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(1ms);
            }
        };

        pool.StartThreads();
        /*
        for (std::size_t i = 0; i<numthreads; ++i){
            pool.DelegateTask(task);
        }
        */
        
        for (std::size_t i = 0; i<(numthreads-1); ++i){
            pool.DelegateTask(task);
        }

        pool.DelegateTask(big_task);
        
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100s);
        done = true;
        pool.Latch();
        pool.StopThreads();
        std::cout<< side_effect << "\n";
    }

    return 0;
}