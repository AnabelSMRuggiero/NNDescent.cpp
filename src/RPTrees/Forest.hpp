#ifndef RPT_FOREST_HPP
#define RPT_FOREST_HPP

#include <utility>
#include <vector>
#include <functional>
#include <numeric>

#include "../NND/RNG.hpp"
#include "SplittingScheme.hpp"

namespace nnd{

/*
This seems out of place here. The utility functions header under NND is mainly for functions
related to NND, and didn't want to tuck this there. This is a super general function though.
*/
template<typename Iterator, typename rIterator, typename SplittingFunction>
int Split(Iterator fromBegin, Iterator fromEnd, Iterator toBegin, rIterator toRev, SplittingFunction splitter){
    int numTrue = 0;
    for ( ; fromBegin != fromEnd; fromBegin++){
        if (splitter(*fromBegin)){
            *toBegin = *fromBegin;
            toBegin++;
            numTrue++;
        } else {
            *toRev = *fromBegin;
            toRev++;
        }
    }
    return numTrue;
}

//TODO: add some way to track best splits
struct RandomProjectionForest{

    std::vector<size_t> indexArray;
    int numberOfSplits;


    struct TreeLeaf{
    
        //This is for the split that produced the children
        size_t childrenSplittingIndex;
        std::pair<size_t,size_t> splitRange;
        //Pretty sure the Leaves can be refs. Either way, non of these pointers own the underlying objects
        //All of the leaves of the tree are owned by the enclosing Vector.
        //Wait. Using pointers like this means the tree can't easily be copied at all...
        //I don't even know if this needs to be copied
        std::pair<TreeLeaf*, TreeLeaf*> children;
        TreeLeaf* parent;
        std::vector<TreeLeaf>* enclosingVector;
        /*
        This structure probably can be condensed back down to just a tuple of {bool, size_t, size_t}
        (or two arrays, bool[] and std::pair/std::range[]).
        That'd remove 4 pointers (4/7s) of the struct at this point, add a literal bit, and require
        alocating enough memory for the entire theoretical max tree.
        That assumption may not be a good one in the parallel case (although I am already reserving an agressive amount of space.)
        I'll leave as is for now in case this makes some other piece easier to work on
        
        */
        TreeLeaf() : splitRange(0,0), childrenSplittingIndex(-1), children(nullptr, nullptr), parent(nullptr), enclosingVector(nullptr){};

        TreeLeaf(size_t index1, size_t index2, size_t splittingIndex, std::vector<TreeLeaf>* enclose) : splitRange(index1, index2),
                                                                                childrenSplittingIndex(splittingIndex), 
                                                                                children(nullptr, nullptr),
                                                                                parent(nullptr), 
                                                                                enclosingVector(enclose){};

        TreeLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex, std::vector<TreeLeaf>* enclose) : splitRange(indecies),
                                                                                    childrenSplittingIndex(splittingIndex), 
                                                                                    children(nullptr, nullptr),
                                                                                    parent(nullptr), 
                                                                                    enclosingVector(enclose){};

        TreeLeaf& AddLeftLeaf(size_t index1, size_t index2, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(index1, index2, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.first = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddLeftLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(indecies, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.first = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddRightLeaf(size_t index1, size_t index2, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(index1, index2, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.second = &newLeaf;
            return newLeaf;
        };

        TreeLeaf& AddRightLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
            TreeLeaf& newLeaf = enclosingVector->emplace_back(indecies, splittingIndex, enclosingVector);
            newLeaf.parent = this;
            this->children.second = &newLeaf;
            return newLeaf;
        };

    };
    
    std::vector<TreeLeaf> treeLeaves;


    RandomProjectionForest(size_t numberOfSamples, StlRngFunctor<> rngFunctor, SplittingScheme getSplitComponents, int splits = 8, int splitThreshold = 255) : 
        numberOfSplits(splits), treeLeaves(0){

        //splittingVectors.reserve((1<<numberOfSplits) - 1);
        treeLeaves.reserve(1<<(numberOfSplits+1));
        treeLeaves.emplace_back(std::pair<size_t, size_t>(0, numberOfSamples), 0, &treeLeaves);

        std::vector<TreeLeaf*> splitQueue1(1, &treeLeaves[0]);
        std::vector<TreeLeaf*> splitQueue2(0);

        std::vector<size_t> indexVector1(numberOfSamples);
        std::iota(indexVector1.begin(), indexVector1.end(), 0);
        std::vector<size_t> indexVector2(numberOfSamples);

        // splittingIndex is for the case of 
        //size_t splittingIndex(0);
        size_t rangeIndexOffset(0);
        std::function<bool(size_t)> splittingFunction;

        //size_t beginIndex(0), endIndex(numberOfSamples - 1);

        for (size_t i = 0; i<numberOfSplits; i+=1){
            
            while(splitQueue1.size() > 0){
                
                TreeLeaf& currentSplit = *(splitQueue1.back());

                
                //This is bootleg af, need to refactor how I do rng.
                decltype(rngFunctor.functorDistribution)::param_type newRange(currentSplit.splitRange.first, currentSplit.splitRange.second - 1);
                rngFunctor.functorDistribution.param(newRange);

                size_t index1(rngFunctor());
                size_t index2(rngFunctor());
                while (index2 == index1){
                    index2 = rngFunctor();
                }
                index1 = indexVector1[index1];
                index2 = indexVector1[index2];

                // Get the splitting vector, this can be fed into this function in the parallel/distributed case.
                splittingFunction = getSplitComponents(currentSplit.childrenSplittingIndex, std::pair<size_t, size_t>(index1, index2));


                auto beginIt = indexVector1.begin();
                std::advance(beginIt, currentSplit.splitRange.first);
                auto endIt = indexVector1.end();
                std::advance(endIt, currentSplit.splitRange.second - numberOfSamples);
                auto toBegin = indexVector2.begin();
                std::advance(toBegin, currentSplit.splitRange.first);
                auto toRev = indexVector2.rbegin();
                std::advance(toRev, numberOfSamples - currentSplit.splitRange.second);

                int numSplit = Split(beginIt, endIt, toBegin, toRev, splittingFunction);


                TreeLeaf* leftSplit = &(currentSplit.AddLeftLeaf(std::pair<size_t, size_t>(currentSplit.splitRange.first, currentSplit.splitRange.first + numSplit),
                                                                 currentSplit.childrenSplittingIndex * 2 + 1));
                TreeLeaf* rightSplit = &(currentSplit.AddRightLeaf(std::pair<size_t, size_t>(currentSplit.splitRange.first + numSplit, currentSplit.splitRange.second),
                                                                   currentSplit.childrenSplittingIndex * 2 + 2));

                if (leftSplit->splitRange.second - leftSplit->splitRange.first > splitThreshold) splitQueue2.push_back(leftSplit);
                if (rightSplit->splitRange.second - rightSplit->splitRange.first > splitThreshold) splitQueue2.push_back(rightSplit);
                splitQueue1.pop_back();

            } //end while
            std::swap(indexVector1, indexVector2);
            std::swap(splitQueue1, splitQueue2);
        } //end for
        indexArray = std::move(indexVector1);
    } //end constructor
    

};



}
#endif //RPT_FOREST_HPP