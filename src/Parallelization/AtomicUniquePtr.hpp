/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_ATOMICUNIQUEPTR_HPP
#define NND_ATOMICUNIQUEPTR_HPP

#include <memory>

namespace nnd{

//libstdc++/libc++ don't support atomic shared ptrs yet


//Only supports stateless deleters atm
template<typename PointedAt, typename Deleter = std::default_delete<PointedAt>>
struct AtomicUniquePtr{

    AtomicUniquePtr(): ptr(nullptr), deleter() {};

    AtomicUniquePtr(PointedAt* ptr): ptr(ptr), deleter(){};

    AtomicUniquePtr(PointedAt* ptr, const Deleter&): ptr(ptr), deleter(){};
    AtomicUniquePtr(PointedAt* ptr, Deleter&&): ptr(ptr), deleter(){};

    AtomicUniquePtr(const AtomicUniquePtr&) = delete;
    AtomicUniquePtr(AtomicUniquePtr&&) = delete;

    ~AtomicUniquePtr(){
        PointedAt* dyingPtr = ptr.load();
        if(dyingPtr != nullptr) Deleter()(dyingPtr);
    }

    operator bool(){
        return ptr.load() != nullptr;
    }



    void operator=(std::unique_ptr<PointedAt, Deleter>&& livePtr){
        PointedAt* null = nullptr;
        assert(ptr.compare_exchange_strong(null, livePtr.get()));
        livePtr.release();
    }

    [[nodiscard]] std::unique_ptr<PointedAt, Deleter> GetUnique(){
        return std::unique_ptr<PointedAt, Deleter>(ptr.exchange(nullptr));
    }

    [[nodiscard]] const PointedAt* get(){
        return ptr.load();
    }

    private:
    std::atomic<PointedAt*> ptr;
    [[no_unique_address]] Deleter deleter;

};


template<typename ElementType>
using AtomicPtrArr = std::unique_ptr<AtomicUniquePtr<ElementType>[]>;

}

#endif