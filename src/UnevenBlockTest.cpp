#include "./ann/Data.hpp"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace nnd;

int main(int argc, char* argv[]){
    //UnevenBlock<long double> testBlock = UninitUnevenBlock<long double>(13, 430);

    std::mt19937_64 engine;
    std::uniform_int_distribution<size_t> distro(1, 85);

    std::vector<size_t> arraySizes;
    size_t numArrays = 12;
    for (size_t i = 0; i<numArrays; i+=1){
        arraySizes.push_back(distro(engine));
    }

    size_t totalElements = std::accumulate(arraySizes.begin(), arraySizes.end(), 0);

    UnevenBlock<long double> testBlock = UninitUnevenBlock<long double>(arraySizes.size(), totalElements);

    size_t* headerStart = static_cast<size_t*>(static_cast<void*>(testBlock.get()));

    std::inclusive_scan(arraySizes.begin(), arraySizes.end(), headerStart+1, std::plus<size_t>{}, 0);
    
    for (auto&& entry: testBlock){
        std::iota(entry.begin(), entry.end(), 0.0l);
    }

    for (auto&& entry: testBlock){
        for (auto&& element: entry){
            std::cout << element << std::endl;
        }
    }

    return 0;
}