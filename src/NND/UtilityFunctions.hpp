#ifndef NND_UTILITYFUNCTIONS_H
#define NND_UTILITYFUNCTIONS_H
#include <utility>

namespace nnd{

bool const NeighborDistanceComparison(const std::pair<size_t,double>& neighborA, const std::pair<size_t,double>& neighborB){
    return neighborA.second < neighborB.second;
};

bool const NeighborIdentityCheck(const std::pair<size_t,double>& neighborA, const std::pair<size_t,double>& neighborB){
    return neighborA.first == neighborB.first;
};

//Todo: template this so it can handle arbitrary float types
struct NeighborSearchFunctor{
    
    size_t searchValue;

    NeighborSearchFunctor() : searchValue(0){};

    bool operator()(std::pair<size_t, double> currentValue){
        return currentValue.first == searchValue;
    }

};



}

#endif //NND_UTILITYFUNCTIONS_H