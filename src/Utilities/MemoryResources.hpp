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

namespace nnd{


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

namespace internal{
    thread_local std::pmr::memory_resource* threadDefaultResource = std::pmr::get_default_resource();

    void SetThreadResource(std::pmr::memory_resource* resourcePtr){
        threadDefaultResource = resourcePtr;
    }

    std::pmr::memory_resource* GetThreadResource(){
        return threadDefaultResource;
    }
}





}

#endif