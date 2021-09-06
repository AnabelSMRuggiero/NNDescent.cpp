
#include <utility>
#include <memory_resource>
#include <span>
#include "src/Parallelization/AsyncQueue.hpp"


using namespace nnd;

template<typename Element>
struct BinaryTreeNode{
    friend BinaryTreeRef<Element>;
    friend BinaryForest<Element>;
    BinaryTreeNode* parent;
    std::pair<BinaryTreeNode*,BinaryTreeNode*> children;
    size_t nodeIndex;
    Element nodeElement;

    BinaryTreeNode() = default;

    template<typename... Args>
    BinaryTreeNode(std::pair<BinaryTreeNode*, size_t> attach, std::tuple<Args...> args): parent(attach.first), children(nullptr, nullptr), nodeIndex(attach.second), nodeElement(std::make_from_tuple<Element>(args)) {};

    private:
    //BinaryTreeNode(BinaryTreeNode&&)
    //template<typename... Args>
    //BinaryTreeNode(BinaryTreeNode* parent, size_t index, Args... args): parent(parent), children(nullptr, nullptr), nodeIndex(index), nodeElement(args...) {};
};

template<typename Element>
struct BinaryTreeNode<Element[]>{
    BinaryTreeNode* parent;
    std::pair<BinaryTreeNode*,BinaryTreeNode*> children;
    size_t nodeIndex;
    std::span<Element> nodeElement;

    BinaryTreeNode() = default;

    //template<typename... Args>
    BinaryTreeNode(std::pair<BinaryTreeNode*, size_t> attach): parent(attach.first), children(nullptr, nullptr), nodeIndex(attach.second){};

    BinaryTreeNode(const BinaryTreeNode&) = delete;

    BinaryTreeNode(BinaryTreeNode&&) = delete;
    //template<typename... Args>
    //BinaryTreeNode(BinaryTreeNode* parent, size_t index, Args... args): parent(parent), children(nullptr, nullptr), nodeIndex(index), nodeElement(args...) {};
};
//std::pmr::synchronized_pool_resource;

template<typename Element>
struct BinaryForest{

    friend BinaryTreeRef<Element>;

    std::pmr::polymorphic_allocator<std::byte> alloc;
    BinaryTreeNode<Element> topNode;

    BinaryForest() = default;

    template<typename... Args>
    BinaryForest(Args... args): alloc(), topNode({nullptr, 0}, std::forward_as_tuple(args...)){};

    template<typename... Args>
    BinaryForest(std::pmr::memory_resource* upstream, std::tuple<Args...> args): alloc(upstream), topNode({nullptr, 0}, args){};

    template<typename... Args>
    BinaryForest(std::pmr::memory_resource* upstream, size_t startIndex, std::tuple<Args...> args): alloc(upstream), topNode({nullptr, startIndex}, args){};

    ~BinaryForest(){
        std::list<BinaryTreeNode<Element>*, std::pmr::polymorphic_allocator<std::byte>> memToFree = allocedMem.TakeAll();
        for (auto& memory: memToFree){
            alloc.deallocate_object(memory, chunkSize);
        }
    }

    private:
    constexpr static size_t chunkSize = 32;
    AsyncQueue<BinaryTreeNode<Element>*, std::pmr::polymorphic_allocator> allocedMem;
};

template<typename Element>
struct BinaryForest<Element[]>{

    friend BinaryTreeRef<Element[]>;

    std::pmr::polymorphic_allocator<std::byte> alloc;
    BinaryTreeNode<Element[]> topNode;
    const size_t numElements;

    BinaryForest(size_t numElements): numElements(numElements);

    BinaryForest(std::pmr::memory_resource* upstream, size_t numElements): alloc(upstream), numElements(numElements){};

    BinaryForest(std::pmr::memory_resource* upstream, size_t startIndex, const size_t numElements): alloc(upstream), topNode({nullptr, startIndex}), numElements(numElements){};

    //A copy constructor would probably be not to bad to write, but I won't need one.
    BinaryForest(const BinaryForest&) = delete;

    BinaryForest(BinaryForest&& other): alloc(std::move(other.alloc)), topNode(std::move(topNode)), numElements(numElements) {};


    ~BinaryForest(){
        std::list<BinaryTreeNode<Element>*, std::pmr::polymorphic_allocator<std::byte>> memToFree = allocedMem.TakeAll();
        for (auto& memory: memToFree){
            alloc.deallocate_object(memory, chunkSize);
        }
    }

    private:
    constexpr static size_t chunkSize = 32;
    AsyncQueue<BinaryTreeNode<Element>*, std::pmr::polymorphic_allocator> allocedMem;
};


template<typename Element>
struct BinaryTreeRef{
    
    BinaryTreeNode<Element>* refNode;


    TreeRef(RandomProjectionForest& forest): alloc(forest.alloc), refNode(forest.topNode){
        GetChunk();
    };
    
    template<typename... ElementArgs>
    BinaryTreeNode<Element>* AddLeftLeaf(ElementArgs... args){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<BinaryTreeNode<Element>>(buildMemory, {refNode, (refNode->nodeIndex)*2+1}, std::forward_as_tuple(args...));
        refNode->children.first = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    template<typename... ElementArgs>
    BinaryTreeNode<Element>* AddRightLeaf(ElementArgs... args){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<BinaryTreeNode<Element>>(buildMemory, {refNode, (refNode->nodeIndex)*2+2}, std::forward_as_tuple(args...));
        refNode->children.second = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    private:
    std::pmr::polymorphic_allocator<std::byte> alloc;
    void GetChunk(){
        buildMemory = alloc.allocate_object<BinaryTreeNode<Element>>(BinaryForest<Element>::chunkSize);
        buildEnd = buildMemory + BinaryForest<Element>::chunkSize;
    }

    AsyncQueue<BinaryTreeNode<Element>*, std::pmr::polymorphic_allocator<std::byte>>& allocedMem;
    BinaryTreeNode<Element>* buildMemory;
    BinaryTreeNode<Element>* buildEnd;
    //constexpr static size_t chunkSize = 32;

};
