/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#include <functional>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <bit>
#include <future>
#include <utility>
#include <memory>
#include <vector>
#include <list>
#include <cassert>
#include <type_traits>
#include <iostream>
#include <optional>


#include "Utilities/Type.hpp"
#include "Utilities/DataSerialization.hpp"
#include "Utilities/DataDeserialization.hpp"
#include "Utilities/Data.hpp"
#include "Utilities/Metrics/SpaceMetrics.hpp"
#include "Utilities/Metrics/FunctorErasure.hpp"

#include "NND/GraphStructures.hpp"
#include "NND/GraphInitialization.hpp"

#include "RPTrees/SplittingScheme.hpp"
#include "RPTrees/Forest.hpp"


using namespace nnd;


template<typename DistType, typename COMExtent>
struct ThreadFunctors{
    DispatchFunctor<DistType> dispatchFunctor;
    CachingFunctor<DistType> cache;
    SinglePointFunctor<COMExtent> comDistFunctor;

    ThreadFunctors() = default;

    ThreadFunctors(const ThreadFunctors&) = default;

    ThreadFunctors& operator=(const ThreadFunctors&) = default;

    template<typename DistanceFunctor, typename COMFunctor>
    ThreadFunctors(DistanceFunctor distanceFunctor, COMFunctor comFunctor, size_t maxBlockSize, size_t numNeighbors):
        dispatchFunctor(distanceFunctor),
        cache(dispatchFunctor, maxBlockSize, numNeighbors),
        comDistFunctor(comFunctor) {};

};



template<typename Task>
struct TaskQueue{

    using TaskType = Task;

    TaskQueue& operator=(TaskQueue&& rhs) = default;

    void AddTask(Task&& task){
        std::unique_lock queueLock(queueMutex);
        tasks.push_back(std::move(task));
        taskCounter++;
        
        queueLock.unlock();
        needTasks.notify_one();
    }

    void AddTasks(std::vector<Task>&& newTasks){
        std::unique_lock queueLock(queueMutex);
        for (auto& task: newTasks){
            tasks.push_back(std::move(task));
            taskCounter++;
        }

        queueLock.unlock();
        needTasks.notify_one();
    }

    Task TakeTask(){
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0){
            needTasks.wait(queueLock, [&]{return taskCounter != 0;});
        }
        taskCounter--;
        Task retTask = std::move(tasks.front());
        tasks.pop_front();
        
        return retTask;
    }
    
    std::list<Task> TakeTasks(){
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0){
            needTasks.wait(queueLock, [&]{return taskCounter != 0;});
        }
        taskCounter = 0;
        std::list<Task> retTasks = std::move(tasks);
        tasks = std::list<Task>();
        return retTasks;
    }

    operator bool(){
        return taskCounter != 0;
    }

    private:
    std::list<Task> tasks;
    std::mutex queueMutex;
    std::atomic<size_t> taskCounter; //Leaving this in for now for expanding functionality
    std::condition_variable needTasks;

};

template<typename ThreadState>
struct ThreadPool;

template<typename ThreadState>
struct TaskThread{

    using ThreadTask = std::function<void(ThreadState&)>;

    friend ThreadPool<ThreadState>;

    TaskQueue<ThreadTask> workQueue;

    TaskThread() = default;

    

    TaskThread(TaskThread&& other) = default;


    
    template<typename ...ThreadStateArgs>
    TaskThread(ThreadStateArgs... args): state(args...), workQueue(), running(false) {};

    auto GetThreadLoop(){
        auto threadLoop = [&](std::stop_token stopToken)->void{

            assert(!(this->running));

            auto queueTerminator = [&]() mutable {
                this->workQueue.AddTask(this->GetTerminator());
            };


            std::stop_callback stopQueuer(stopToken, queueTerminator);

            this->running = true;
            while(running){

                ThreadTask taskToDo = this->workQueue.TakeTask();
                taskToDo(state);

            }

        };

        return threadLoop;
    }

    ThreadTask GetTerminator(){
        return [&](ThreadState& threadState){
            running = false;
        };
    }


    private:
    TaskThread(const TaskThread&) = default;
    TaskThread& operator=(const TaskThread&) = default;
    bool running;
    ThreadState state;

};


/*
template<typename ElementType, typename ...ConstructorArgs>
    requires std::is_unbounded_array_v<T>
std::unique_ptr<ArrayType> MakeArray(size_t numEle, ConstructorArgs... args){
    ElementType* retArray = ::operator new[](numEle*sizeof(ElementType));
    size_t constructCount(0);
    try{
        for (size_t i = 0; i<numEle; i+=1){
            ElementType* catchPtr = new(retArray+i) ElementType(args...);
        }
    } catch (...){
        for (size_t i = constructCount; i>=0; i-=1){
            (retArray+i)->~ElementType();
        }
        ::operator delete[](retArray);
        throw;
    }
}
*/
template<typename ThreadState>
struct ThreadPool{


    using ThreadTask = std::function<void(ThreadState&)>;

    ThreadPool(): numThreads(std::jthread::hardware_concurrency()),
                  delegationCounter(0),
                  threadStates(std::make_unique<TaskThread<ThreadState>[]>(numThreads)), 
                  threadHandles(std::make_unique<std::jthread[]>(numThreads)) {};

    ThreadPool(const size_t numThreads): numThreads(numThreads),
                                         delegationCounter(0),
                                         threadStates(std::make_unique<TaskThread<ThreadState>[]>(numThreads)), 
                                         threadHandles(std::make_unique<std::jthread[]>(numThreads)) {};

    template<typename ...ThreadStateArgs>
    ThreadPool(const size_t numThreads, ThreadStateArgs... args): numThreads(numThreads),
                                         delegationCounter(0),
                                         threadStates(std::make_unique<TaskThread<ThreadState>[]>(numThreads)), 
                                         threadHandles(std::make_unique<std::jthread[]>(numThreads)){
        for (size_t i = 0; i<numThreads; i += 1){
            threadStates[i].state = ThreadState(args...);
        }
    }

    void StartThreads(){
        /*
        auto stopTaskGenerator = [&](const size_t threadNumber)->auto{
            ThreadTask stopTask = threadStates[threadNumber].GetTerminator();
            auto queueStop = [&, threadNumber, stopTask]() mutable {
                threadStates[threadNumber].workQueue.AddTask(std::move(stopTask));
            };
            return queueStop;
        };
        */

        for(size_t i = 0; i<numThreads; i+=1){
            threadHandles[i] = std::jthread(threadStates[i].GetThreadLoop());
        }

    };

    void StopThreads(){
        for(size_t i = 0; i<numThreads; i+=1){
            threadHandles[i].request_stop();
        } 
        for(size_t i = 0; i<numThreads; i+=1){
            threadHandles[i].join();
        } 
    };

    void DelegateTask(ThreadTask&& task){
        //implement something smarter here
        //or maybe I won't need to if/when I implement task stealing
        threadStates[delegationCounter].workQueue.AddTask(std::move(task));
        delegationCounter = (delegationCounter+1)%numThreads;

    }

    private:
    
    const size_t numThreads;
    size_t delegationCounter;
    std::unique_ptr<TaskThread<ThreadState>[]> threadStates; 
    std::unique_ptr<std::jthread[]> threadHandles;
    
};

template<typename TaskResult>
using TaskStates = std::unique_ptr<std::pair<std::promise<TaskResult>, std::future<TaskResult>>[]>;

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

    std::unique_ptr<PointedAt, Deleter> load(){
        return std::unique_ptr<PointedAt, Deleter>(ptr.exchange(nullptr));
    }

    private:
    std::atomic<PointedAt*> ptr;
    [[no_unique_address]] Deleter deleter;

};

//template<typename ElementType>
//using AtomicPtrVector = std::vector<std::atomic<std::shared_ptr<ElementType>>>;

template<typename ElementType>
using AtomicPtrArr = std::unique_ptr<AtomicUniquePtr<ElementType>[]>;
/*
template<typename DistType, typename COMExtent>
TaskStates<BlockUpdateContext<DistType>> InitializeBlockContexts(const size_t numBlocks,
                                                          const std::vector<size_t>& blockSizes,
                                                          const HyperParameterValues& params,
                                                          ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    
    TaskStates<BlockUpdateContext<DistType>> retArr = std::make_unique<std::pair<std::promise<Graph<size_t, DistType>>, std::future<Graph<size_t, DistType>>>[]>(blockSizes.size());
    //retArr.reserve(blockSizes.size());

    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{

        std::promise<BlockUpdateContext<DistType>> taskResult;
        std::future<BlockUpdateContext<DistType>> taskFuture = taskResult.get_future();

        retArr[blockNum] = {std::move(taskResult), std::move(taskFuture)};
        auto task = [=, &result = retArr[blockNum].first](ThreadFunctors<DistType, COMExtent>& functors) mutable {
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            functors.comDistFunctor.SetBlock(blockNum);
            Graph<size_t, DistType> blockGraph = BruteForceBlock(params.indexParams.blockGraphNeighbors, blockSize, functors.dispatchFunctor);
            
            GraphVertex<size_t, DistType> queryHint = QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor);
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, numBlocks);
            BlockUpdateContext<DistType> blockContext(blockGraph, std::move(queryContext), numBlocks);
            result.set_value(blockContext);
        };

        return std::function(std::move(task));
    };

    
    //generate tasks, return futures
    for (size_t i = 0; i<numBlocks; i+=1){
        threadPool.DelegateTask(taskGenerator(i, blockSizes[i]));
    }

    return retArr;
}
*/

template<typename DistType, typename COMExtent>
AtomicPtrArr<BlockUpdateContext<DistType>> InitializeBlockContexts(const size_t numBlocks,
                                                          const std::vector<size_t>& blockSizes,
                                                          const HyperParameterValues& params,
                                                          ThreadPool<ThreadFunctors<DistType, COMExtent>>& threadPool){
    
    AtomicPtrArr<BlockUpdateContext<DistType>> retArr = std::make_unique<AtomicUniquePtr<BlockUpdateContext<DistType>>[]>(blockSizes.size());


    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{

        auto task = [=, &atomicPtr = retArr[blockNum]](ThreadFunctors<DistType, COMExtent>& functors) mutable {
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            functors.comDistFunctor.SetBlock(blockNum);
            Graph<size_t, DistType> blockGraph = BruteForceBlock(params.indexParams.blockGraphNeighbors, blockSize, functors.dispatchFunctor);
            
            GraphVertex<size_t, DistType> queryHint = QueryHintFromCOM(blockNum, blockGraph, params.indexParams.blockGraphNeighbors, functors.comDistFunctor);
            QueryContext<DistType> queryContext(blockGraph, std::move(queryHint), params.indexParams.queryDepth, blockNum, numBlocks);
            std::unique_ptr<BlockUpdateContext<DistType>> blockContextPtr = std::make_unique<BlockUpdateContext<DistType>>(blockGraph, std::move(queryContext), numBlocks);
            atomicPtr = std::move(blockContextPtr);
        };

        return std::function(std::move(task));
    };

    
    //generate tasks, return ptrs
    for (size_t i = 0; i<numBlocks; i+=1){
        threadPool.DelegateTask(taskGenerator(i, blockSizes[i]));
    }

    return retArr;
}

template<typename COMExtent>
std::vector<std::optional<ComparisonKey<size_t>>> NearestNodesToDo(const MetaGraph<COMExtent>& metaGraph){

    std::unordered_set<ComparisonKey<size_t>> nearestNodeDistQueue;

    for (size_t i = 0; const auto& vertex: metaGraph.verticies){
        for (const auto& neighbor: vertex){
            nearestNodeDistQueue.insert({i, neighbor.first});
        }
        i++;
    }
    
    std::vector<std::optional<ComparisonKey<size_t>>> distancesToCompute;
    distancesToCompute.reserve(nearestNodeDistQueue.size());
    for (const auto& pair: nearestNodeDistQueue){
        distancesToCompute.push_back(std::make_optional(pair));
    }

    return distancesToCompute;
}

template<typename DistType>
void QueueNearestNodes(std::vector<std::optional<ComparisonKey<size_t>>>& distancesToCompute, const AtomicUniquePtr<BlockUpdateContext<DistType>>* blockUpdateContexts){
    std::vector<std::tuple<size_t, size_t, DistType>> nnDistanceResults(nearestNodeDistQueue.size());
    auto nnDistanceFunctor = [&](const ComparisonKey<size_t> blockNumbers) -> std::tuple<size_t, size_t, DistType>{
        return blockUpdateContexts[blockNumbers.first].queryContext.NearestNodes(blockUpdateContexts[blockNumbers.second].queryContext,
                                                                                 distanceFunctor);
    };
}


/*
    std::vector<Graph<size_t, DistType>> blockGraphs(0);
    blockGraphs.reserve(blockSizes.size());
    for (size_t i =0; i<numBlocks; i+=1){
        distanceFunctor.SetBlocks(i,i);
        blockGraphs.push_back(BruteForceBlock<DistType>(numNeighbors, blockSizes[i], distanceFunctor));
    }

    return blockGraphs;
*/


int main(){

    static const std::endian dataEndianness = std::endian::big;

    std::string trainDataFilePath("./TestData/MNIST-Fashion-Train.bin");
    DataSet<AlignedArray<float>> mnistFashionTrain(trainDataFilePath, 28*28, 60'000, &ExtractNumericArray<AlignedArray<float>,dataEndianness>);


    std::mt19937_64 rngEngine(0);
    std::uniform_int_distribution<size_t> rngDist(size_t(0), mnistFashionTrain.numberOfSamples - 1);
    StlRngFunctor<std::mt19937_64, std::uniform_int_distribution, size_t> rngFunctor(std::move(rngEngine), std::move(rngDist));

    EuclidianTrain<AlignedArray<float>, AlignedArray<float>> splittingScheme(mnistFashionTrain);
    TrainingSplittingScheme splitterFunc(splittingScheme);
    
    IndexParamters indexParams{5, 10, 3, 2};

    size_t numBlockGraphNeighbors = 5;
    size_t numCOMNeighbors = 10;
    size_t maxNearestNodes = 3;
    size_t queryDepth = 2;

    SearchParameters searchParams{10, 10, 10};
    size_t numberSearchNeighbors = 10;
    size_t searchQueryDepth = 10;
    size_t maxNewSearches = 10;

    SplittingHeurisitcs splitParams= {16, 140, 60, 180};

    HyperParameterValues parameters{splitParams, indexParams, searchParams};

    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc, splitParams);



    auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTreesTrain, mnistFashionTrain);

    
    MetaGraph<float> metaGraph(dataBlocks, parameters.indexParams.COMNeighbors, EuclideanMetricPair());

    MetricFunctor<AlignedArray<float>, EuclideanMetricPair> euclideanFunctor(dataBlocks);
    DataComDistance<AlignedArray<float>, float, EuclideanMetricPair> comFunctor(metaGraph, dataBlocks);

    ThreadPool<ThreadFunctors<float, float>> pool(12, euclideanFunctor, comFunctor, splitParams.maxTreeSize, 10);

    std::vector<size_t> sizes;
    sizes.reserve(dataBlocks.size());
    for(const auto& block: dataBlocks){
        sizes.push_back(block.size());
    }
    


    pool.StartThreads();

    auto futures = InitializeBlockContexts(dataBlocks.size(), sizes, parameters, pool);

    //auto testRes = futures[sizes.size()-1].second.get();
    

    pool.StopThreads();

    for (size_t i = 0; i<dataBlocks.size(); i+=1){
        //if (!(futures[i])) 
        assert(futures[i]);
        auto test = futures[i].load();
        std::cout << test->currentGraph.size() << std::endl;
    }
    assert(futures[0] == false);
    return 0;
}