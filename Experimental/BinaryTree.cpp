struct RPLeaf{
    size_t begin;
    size_t end;
    
    RPLeaf() = default;
    
    RPLeaf(std::pair<size_t, size_t> range): begin(range.first), end(range.second) {};
    
};

template<typename Element>
struct BinaryTreeNode{
    BinaryTreeNode* parent;
    std::pair<BinaryTreeNode*,BinaryTreeNode*> children;
    size_t nodeIndex;
    Element nodeElement;

    BinaryTreeNode() = default;

    template<typename... Args>
    BinaryTreeNode(std::pair<BinaryTreeNode*, size_t> attach, std::tuple<Args...> args): parent(attach.first), children(nullptr, nullptr), nodeIndex(attach.second), nodeElement(std::make_from_tuple<Element>(args)) {};

    //template<typename... Args>
    //BinaryTreeNode(BinaryTreeNode* parent, size_t index, Args... args): parent(parent), children(nullptr, nullptr), nodeIndex(index), nodeElement(args...) {};
};

//std::pmr::synchronized_pool_resource;

template<typename Element>
struct BinaryForest{
    std::pmr::polymorphic_allocator<std::byte> alloc;
    BinaryTreeNode<Element> topNode;

    BinaryForest() = default;

    template<typename... Args>
    BinaryForest(std::tuple<Args...> args): alloc(), topNode({nullptr, 0}, args){};

    template<typename... Args>
    BinaryForest(std::pmr::memory_resource* upstream, std::tuple<Args...> args): alloc(upstream), topNode({nullptr, 0}, args){};

    template<typename... Args>
    BinaryForest(std::pmr::memory_resource* upstream, size_t startIndex, std::tuple<Args...> args): alloc(upstream), topNode({nullptr, startIndex}, args){};
};

struct RandomProjectionForest{

    std::pmr::synchronized_pool_resource treeBuffer;
    std::span<size_t> indecies;
    BinaryForest<RPLeaf> tree;
    

    RandomProjectionForest(std::pmr::memory_resource* upstream, std::pair<size_t, size_t> topIndexRange):
        treeBuffer(upstream),
        tree(&treeBuffer, topIndexRange){};

    
};

/*
struct RandomProjectionForest{

    //std::pmr::synchronized_pool_resource treeBuffer;
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf* topNode;
    std::span<size_t> indecies;

    RandomProjectionForest(std::pmr::memory_resource* upstream, std::pair<size_t, size_t> topIndexRange):
        alloc(upstream), topNode(alloc.new_object<TreeLeaf>(topIndexRange, 0, nullptr)){};

    
};
*/
template<typename Element>
struct BinaryTreeRef{
    std::pmr::polymorphic_allocator<std::byte> alloc;
    TreeLeaf* refNode;


    TreeRef(RandomProjectionForest& forest): alloc(forest.alloc), refNode(forest.topNode){
        GetChunk();
    };
    
    TreeLeaf* AddLeftLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<TreeLeaf>(buildMemory, indecies, splittingIndex, refNode);
        refNode->children.first = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    TreeLeaf* AddRightLeaf(std::pair<size_t, size_t> indecies, size_t splittingIndex){
        if (buildMemory == buildEnd) GetChunk();
        alloc.construct<TreeLeaf>(buildMemory, indecies, splittingIndex, refNode);
        refNode->children.second = buildMemory;
        buildMemory++;
        return buildMemory-1;
    }
    
    private:

    void GetChunk(){
        buildMemory = alloc.allocate_object<TreeLeaf>(chunkSize);
        buildEnd = buildMemory + 32;
    }

    TreeLeaf* buildMemory;
    TreeLeaf* buildEnd;
    constexpr static size_t chunkSize = 32;

};
