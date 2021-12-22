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

#include <memory_resource>
#include <cstddef>
#include <utility>
#include <new>
#include <array>
#include <optional>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <bit>
#include <memory>

#include "PointerManipulation.hpp"

namespace nnd{

namespace internal{
    //Can't grab the default here because set_default_resource is usually called in main,
    //the static and first thread_local one are instantiated before then.
    thread_local std::pmr::memory_resource* threadDefaultResource = std::pmr::new_delete_resource();

    void SetThreadResource(std::pmr::memory_resource* resourcePtr){
        threadDefaultResource = resourcePtr;
    }

    std::pmr::memory_resource* GetThreadResource(){
        return threadDefaultResource;
    }

    static std::atomic<std::pmr::memory_resource*> internalDefaultResource = std::pmr::new_delete_resource();

    void SetInternalResource(std::pmr::memory_resource* resourcePtr){
        internalDefaultResource = resourcePtr;
    }

    std::pmr::memory_resource* GetInternalResource(){
        return internalDefaultResource;
    }


}


struct FreeListNode{
    std::byte* chunkStart;
    size_t chunkSize;
    size_t chunkAlign;
    FreeListNode* next;
};


struct FreeListResource : std::pmr::memory_resource{

    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{
        auto [newSize, newAlign] = AdjustAllocation(bytes, alignment);
        if (head == nullptr){
            return GetChunk(newSize, newAlign);
        } else {

        }
        //itr through list and return valid chunk size
            //if good relocate list node to new alloc end
            //and return ptr to chunk start

        //else return new chunk
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{
        //find list node location from args
        //push to front of list
        //if list over threshold, return an old chunk upstream
    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override{
        return this == &other;
    };


    private:

    

    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();
    FreeListNode* head = nullptr;
    size_t listSize = 0;

    
    void* FindNodeLocation(std::byte* chunkStart, size_t size, size_t alignment){
        //alignment = std::max(alignment , alignof(FreeListNode));
        size_t excess = size % alignof(FreeListNode);
        if(excess > 0){
            size += alignof(FreeListNode) - excess;
        }
        void* ptrToNode = static_cast<void*>(chunkStart + size); 
        return ptrToNode;
    }

    void ReemplaceNode(FreeListNode* chunkNode, size_t requestedSize, size_t requestedAlignment){
        FreeListNode temp = *chunkNode;
        temp.next = nullptr;
        chunkNode->~FreeListNode();
        void* newLoc = FindNodeLocation(temp.chunkStart, requestedSize, requestedAlignment);
        new (newLoc) FreeListNode(temp);
    }

    FreeListNode* PopFront(){
        FreeListNode* oldHead = head;
        head = head->next;
        listSize -= 1;
        return oldHead;
    }
    
    void PushFront(FreeListNode* newNode){
        
        newNode->next = head;
        head = newNode;
        listSize += 1;
        
    }
    
    void ReturnChunk(FreeListNode* listNode){
        upstream->deallocate(listNode->chunkStart, listNode->chunkSize, listNode->chunkAlign);
    }

    std::pair<size_t, size_t> AdjustAllocation(size_t size, size_t alignment){
        alignment = std::max(alignment, alignof(FreeListNode));
        size_t excess = size % alignof(FreeListNode);
        if(excess > 0){
            size += alignof(FreeListNode) - excess;
        }
        size += sizeof(FreeListNode);
        return {size, alignment};
    }

    void* GetChunk(size_t size, size_t alignment){
        
        //auto [newSize, newAlign] = AdjustAllocation(size, alignment);

        void* chunk = upstream->allocate(size, alignment);
        
        std::byte* byteArray = new (chunk) std::byte[size];
        std::byte* nodeLocation = byteArray + (size - sizeof(FreeListNode));
        new (nodeLocation) FreeListNode{byteArray, size, alignment, nullptr};
        
        return chunk;
    }

};



struct CacheNode{
    std::byte* chunkStart = nullptr;
    size_t chunkSize = 0;
    size_t chunkAlign = 0;
};

//MemoryCache owns the memory it is holding, but gives up ownership of the memory it gives out.
//This way, as long as two MemoryCaches have the same upstream, they could be interchangable (not implemented yet)
template<size_t cacheSlots = 3>
struct MemoryCache : std::pmr::memory_resource{
    using NodeType = CacheNode;

    static constexpr size_t cacheSize = cacheSlots;

    MemoryCache() = default;
    
    MemoryCache(std::pmr::memory_resource* upstream): upstream(upstream) {}

    MemoryCache(const MemoryCache&) = delete;

    MemoryCache(MemoryCache&&) = delete;

    MemoryCache& operator=(const MemoryCache&) = delete;

    MemoryCache& operator=(MemoryCache&&) = delete;

    ~MemoryCache(){
        for (auto itr = cachedMemory.begin(); itr != partitionPoint; itr++){
            ReturnChunk(*(*itr));
        }
    }

    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{
        auto [newSize, newAlign] = AdjustAllocation(bytes, alignment);
        
        for (auto itr = cachedMemory.begin(); itr != partitionPoint; itr++){
            std::optional<NodeType>& node = *itr;
            if ((node->chunkSize >= newSize) && (node->chunkAlign >= newAlign)){
                std::byte* memPtr = node->chunkStart;
                ReemplaceNode(node, bytes, alignment);
                --partitionPoint;
                std::swap(node, *partitionPoint);
                return static_cast<void*>(memPtr);
            }   
        }

        return GetChunk(newSize, newAlign);

        //else return new chunk
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{
        std::byte* byteArray = std::launder(static_cast<std::byte*>(p));
        NodeType* nodeLocation = std::launder(static_cast<NodeType*>(FindNodeLocation(byteArray, bytes, alignment)));

        if (partitionPoint != cachedMemory.end()){
            *partitionPoint = *nodeLocation;
            partitionPoint++;
        } else{
            ReturnChunk(*(cachedMemory.back()));
            cachedMemory.back() = *nodeLocation;
            std::swap(cachedMemory.back(), cachedMemory.front());
        }

        nodeLocation->~NodeType();
        //find list node location from args
        //push to front of list
        //if list over threshold, return an old chunk upstream
    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override {
        return this == &other;
    };


    private:

    std::array<std::optional<NodeType>, cacheSize> cachedMemory;
    typename std::array<std::optional<NodeType>, cacheSize>::iterator partitionPoint = cachedMemory.begin();

    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();


    
    void* FindNodeLocation(std::byte* chunkStart, size_t size, size_t alignment){
        alignment = std::max(alignment, alignof(NodeType));
        size_t excess = size % alignof(NodeType);
        if(excess > 0){
            size += alignof(NodeType) - excess;
        }
        void* ptrToNode = static_cast<void*>(chunkStart + size); 
        return ptrToNode;
    }

    void ReemplaceNode(std::optional<NodeType>& nodeToPlace, size_t requestedSize, size_t requestedAlignment){
        
        void* newLoc = FindNodeLocation(nodeToPlace->chunkStart, requestedSize, requestedAlignment);
        new (newLoc) NodeType(*nodeToPlace);
        nodeToPlace = std::nullopt;
    }

    
    void ReturnChunk(const NodeType& memToReturn){
        upstream->deallocate(memToReturn.chunkStart, memToReturn.chunkSize, memToReturn.chunkAlign);
    }

    std::pair<size_t, size_t> AdjustAllocation(size_t size, size_t alignment){
        alignment = std::max(alignment, alignof(NodeType));
        size_t excess = size % alignof(NodeType);
        if(excess > 0){
            size += alignof(NodeType) - excess;
        }
        size += sizeof(NodeType);
        return {size, alignment};
    }

    void* GetChunk(size_t size, size_t alignment){
        
        //auto [newSize, newAlign] = AdjustAllocation(size, alignment);

        void* chunk = upstream->allocate(size, alignment);
        
        std::byte* byteArray = new (chunk) std::byte[size];
        std::byte* nodeLocation = byteArray + (size - sizeof(NodeType));
        new (nodeLocation) NodeType{byteArray, size, alignment};
        
        return chunk;
    }

};

//For testing, prints allocs and deallocs to cout
struct ChatterResource : std::pmr::memory_resource{
    
    ChatterResource() = default;

    ChatterResource(std::pmr::memory_resource* upstream): upstream(upstream){}

    ChatterResource(const ChatterResource&) = delete;

    ChatterResource(ChatterResource&&) = delete;

    ChatterResource& operator=(const ChatterResource&) = delete;

    ChatterResource& operator=(ChatterResource&&) = delete;


    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{
        
        std::cout << "Allocation - size: " << bytes << ", alignment: " << alignment << std::endl;

        return upstream->allocate(bytes, alignment);
        //else return new chunk
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{

        std::cout << "Deallocation - size: " << bytes << ", alignment: " << alignment << std::endl;

        upstream->deallocate(p, bytes, alignment);
    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override{
        return this == &other;
    };

    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();
};

struct SingleListNode{
    SingleListNode* next;
};
struct MemoryChunk{
    std::byte* start;
    size_t size;
};
//Implementation detail of AtomicMultiPool

struct AtomicPool{
    const size_t chunkSize;
    const size_t restockAmount;
    
    AtomicPool(const size_t chunkSize, const size_t restockAmount, std::pmr::memory_resource* upstream):
        chunkSize{chunkSize},
        restockAmount{restockAmount},
        memoryChunks{upstream} {}

    std::byte* Allocate(const auto& grabFromUpstream){
        SingleListNode* currentHead = stackHead.load();
        while(true){
            if(currentHead == nullptr){
                Restock();
            }

            while(currentHead != nullptr){
                if(stackHead.compare_exchange_strong(currentHead, currentHead->next)){
                    currentHead->~SingleListNode();
                    return PtrCast<std::byte*>(currentHead);
                }
            }
        }
    }

    void Restock(auto& grabFromUpstream, SingleListNode* currentHead = nullptr){
        std::unique_lock restockGuard(restockMutex, std::try_to_lock_t{});
        if (restockGuard){
            do{
                std::byte* memoryChunk = grabFromUpstream(restockAmount*chunkSize, memoryChunks);
                //memoryChunks.push_back(memoryChunk);
                for (size_t i = 0; i<restockAmount; i+=1){
                    SingleListNode* newNode = new (memoryChunk) SingleListNode{stackHead.load()};
                    while(!stackHead.compare_exchange_strong(newChunk->next, newNode)){};
                    memoryChunk += chunkSize;
                    stackHead.notify_one();
                }
            } while((waitingThreads.load() == 0) && (stackHead.load() != nullptr));
        } else{
            waitingThreads.fetch_add(1);
            stackHead.wait(currentHead);
            waitingThreads.fetch_sub(1);
        }
    }

    void Deallocate(std::byte* memoryChunk){
        SingleListNode* returningChunk = new (memoryChunk) SingleListNode{stackHead.load()};
        while(!stackHead.compare_exchange_strong(newChunk->next, returningChunk)){};
    }

    void Release(auto& returnToUpstream){
        returnToUpstream(memoryChunks);
    }

    private:
    std::atomic<SingleListNode*> stackHead{nullptr};
    std::mutex restockMutex;
    std::atomic<size_t> waitingThreads{0};
    std::pmr::vector<MemoryChunk> memoryChunks;
};

struct AtomicMultipool : std::pmr::memory_resource{
    static constexpr size_t startSize = 64;
    static constexpr size_t numberOfPools = 8;
    static constexpr size_t biggestPoolSize = startSize<<(numberOfPools-1); //4k atm

    static constexpr size_t defaultAlignment = 64;
    static_assert(defaultAlignment>=sizeof(std::byte*)); //Implementation assumption
    static constexpr size_t defaultRestock = 10; //can do something smarter later

    AtomicMultipool(std::pmr::memory_resource* upstream): upstreamAlloc{upstream} {
        memoryPools.reserve(numberOfPools);

        auto pullFromUpstream = [&] (const size_t chunkSize, std::pmr::vector<MemoryChunk>& poolChunks)->std::byte*{
            std::unique_lock lockMultipool{this->upstreamLock};
            std::byte* memory = static_cast<std::byte*>(this->upstreamAlloc.allocate_bytes(chunkSize, ));
            //This may cause the vector to reallocate, that reallocation MUST be guarded.
            poolChunks.push_back({memory, chunkSize});
            return memory;
        };

        for (size_t i = 0; i<numberOfPools; i+=1){
            memoryPools.emplace_back(startSize<<i, defaultRestock, upstreamAlloc.resource());
            memoryPools.back().Restock(pullFromUpstream);
        }
    }

    AtomicMultipool(const AtomicMultipool&) = delete;

    AtomicMultipool(AtomicMultipool&&) = delete;

    AtomicMultipool& operator=(const AtomicMultipool&) = delete;

    AtomicMultipool& operator=(AtomicMultipool&&) = delete;

    ~AtomicMultipool(){
        Release();
    }

    private:


    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{

        if ((bytes+alignment)>biggestPoolSize){
            return OversizedAlloc(bytes, alignment);
        }

        auto pullFromUpstream = [&] (const size_t chunkSize, std::pmr::vector<MemoryChunk>& poolChunks)->std::byte*{
            std::unique_lock lockMultipool{this->upstreamLock};
            std::byte* memory = static_cast<std::byte*>(this->upstreamAlloc.allocate_bytes(chunkSize, defaultAlignment));
            //This may cause the vector to reallocate, that reallocation MUST be guarded.
            poolChunks.push_back({memory, chunkSize});
            return memory;
        };

        [[likely]] if (alignment<=defaultAlignment){
            size_t adjustedSize = std::bit_ceil(bytes);
            
            size_t poolIndex = [&]()->size_t{
                if (adjustedSize <= startSize){
                    return 0;
                }
                constexpr size_t indexOffset = std::countl_zero(startSize);
                return std::countl_zero(adjustedSize) - indexOffset;
            }();

            std::byte* chunk = memoryPools[poolIndex].Allocate(pullFromUpstream);

            return static_cast<void*>(chunk);
            
        } else { //guh, who are you, me?
            size_t adjustedSize = std::bit_ceil(bytes+alignment);
            
            size_t poolIndex = [&]()->size_t{
                if (adjustedSize <= startSize){
                    return 0;
                }
                constexpr size_t indexOffset = std::countl_zero(startSize);
                return std::countl_zero(adjustedSize) - indexOffset;
            }();

            std::byte* chunkStart = memoryPools[poolIndex].Allocate(pullFromUpstream);

            void* retPointer = static_cast<void*>(chunkStart);
            std::align(alignment, bytes, retPointer, adjustedSize);

            if ((adjustedSize-bytes) > sizeof(std::byte*)*2){
                new (retPointer + bytes) std::byte{1}; //pointer to head of chunk is on this side
                new (retPointer + bytes + sizeof(std::byte*)) std::byte*{chunkStart};
            } else {
                new (retPointer + bytes) std::byte{0}; //pointer is near beginning of alloc;
                new (retPointer - sizeof(std::byte*)) std::byte*{chunkStart};
            }

            return retPointer;
        }
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{

    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override{
        return this == &other;
    };

    void* OversizedAlloc( std::size_t bytes, std::size_t alignment ){

    }

    Release(){

    }

    std::pmr::polymorphic_allocator<> upstreamAlloc = std::pmr::get_default_resource();
    std::pmr::vector<AtomicPool> memoryPools;
    std::mutex upstreamLock;
    //Add in some sort of flexible memory cache to handle oversized allocs
};



//literally just std::pmr::polymorphic_allocator, but with overwritten default constructor that pulls from the internal resource.
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