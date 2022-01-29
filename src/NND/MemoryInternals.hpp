/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_MEMORYINTERALS_HPP
#define NND_MEMORYINTERALS_HPP

#include <memory_resource>
#include <atomic>

namespace nnd{

namespace internal{
//Can't grab the default here because set_default_resource is usually called in main,
//the static and first thread_local one are instantiated before then.
inline thread_local std::pmr::memory_resource* threadDefaultResource = std::pmr::new_delete_resource();

inline void SetThreadResource(std::pmr::memory_resource* resourcePtr){
    threadDefaultResource = resourcePtr;
}

inline std::pmr::memory_resource* GetThreadResource(){
    return threadDefaultResource;
}

inline std::atomic<std::pmr::memory_resource*> internalDefaultResource = std::pmr::new_delete_resource();

inline void SetInternalResource(std::pmr::memory_resource* resourcePtr){
    internalDefaultResource = resourcePtr;
}

inline std::pmr::memory_resource* GetInternalResource(){
    return internalDefaultResource;
}


}

//literally just std::pmr::polymorphic_allocator, but with shadowed default constructor that pulls from the internal resource.
template<typename ValueType>
struct PolymorphicAllocator : std::pmr::polymorphic_allocator<ValueType>{
    using std::pmr::polymorphic_allocator<ValueType>::polymorphic_allocator;
    PolymorphicAllocator() : std::pmr::polymorphic_allocator<ValueType>(internal::GetInternalResource()){}

    template<typename OtherValueType>
    PolymorphicAllocator(const std::pmr::polymorphic_allocator<OtherValueType>& other) : std::pmr::polymorphic_allocator<ValueType>(other) {}

    PolymorphicAllocator select_on_container_copy_construction() const{
        return PolymorphicAllocator();
    }
};


}

#endif