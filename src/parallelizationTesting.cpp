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


template<typename DistType>
struct ThreadFunctors{
    DispatchFunctor<DistType> dispatchFunctor;
    CachingFunctor<DistType> cache;

    template<typename DistanceFunctor>
    ThreadFunctors(DistanceFunctor distanceFunctor, size_t maxBlockSize, size_t numNeighbors): dispatchFunctor(distanceFunctor), cache(dispatchFunctor, maxBlockSize, numNeighbors){}

};

template<typename Task>
struct TaskQueue{

    using TaskType = Task;

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
        std::scoped_lock queueLock(queueMutex);
        if (taskCounter == 0){
            needTasks.wait(queueLock, []{return taskCounter != 0;});
        }
        taskCounter--;
        Task retTask = std::move(tasks.front());
        tasks.pop_front();
        
        return retTask;
    }
    
    std::list<Task> TakeTasks(){
        std::scoped_lock queueLock(queueMutex);
        if (taskCounter == 0){
            needTasks.wait(queueLock, []{return taskCounter != 0;});
        }
        taskCounter = 0;
        std::list<Task> retTasks = std::move(tasks);
        tasks = std::list<Task>();
        return retTasks;
    }

    bool operator bool(){
        return taskCounter != 0;
    }

    private:
    std::list<Task> tasks;
    std::mutex queueMutex;
    std::atomic<size_t> taskCounter; //Leaving this in for now for expanding functionality
    std::condition_variable needTasks;

};

template<typename ThreadState>
struct TaskThread{

    using ThreadTask = std::function<void(ThreadState&)>;

    TaskQueue<ThreadTask> workQueue;

    TaskThread() = default;

    TaskThread(const TaskThread&) = delete;

    template<typename DistanceFunctor>
    TaskThread(DistanceFunctor distanceFunctor, size_t maxBlockSize, size_t numNeighbors): functors(distanceFunctor, maxBlockSize, numNeighbors), workQueue(), running(false) {};

    //template<typename StopCallBack>
    void operator()(std::stop_token stopToken, auto queueStopTask){

        std::stop_callback stopQueuer(stopToken, queueStopTask);

        running = true;
        while(running){

            ThreadTask taskToDo = workQueue.TakeTask();
            taskToDo(functors);

        }
    }

    ThreadTask GetTerminator(){
        return [&running](ThreadFunctors<DistType>& threadState){
            running = false;
        };
    }


    private:
    bool running;
    ThreadState functors;

};

template<typename ThreadState>
struct ThreadPool{

    using ThreadTask = std::function<void(ThreadState&)>;

    ThreadPool(): numThreads(std::jthread::hardware_concurrency()),
                  delegationCounter(0),
                  threadStates(std::make_unique<TaskThread<ThreadState>>(numThreads)), 
                  threadHandles(std::make_unique<std::jthread>(numThreads)) {};

    ThreadPool(const size_t numThreads): numThreads(numThreads),
                                         delegationCounter(0),
                                         threadStates(std::make_unique<TaskThread<ThreadState>>(numThreads)), 
                                         threadHandles(std::make_unique<std::jthread>(numThreads)) {};

    void StartThreads(){
        
        auto stopTaskGenerator = [&](const size_t threadNumber)->auto{
            ThreadTask stopTask = threadStates[threadNumber];
            auto queueStop = [&, threadNumber, stopTask](){
                threadStates[threadNumber].workQueue.AddTask(stopTask);
            };
            return queueStop;
        };
        

        for(size_t i = 0; i<numThreads; i+=1){
            threadHandles[i] = std::jthread(threadStates[i], stopTaskGenerator(i));
        }

    };

    void StopThreads(){
        for(size_t i = 0; i<numThreads; i+=1){
            threadHandles[i].request_stop();
        } 
    };

    void DelegateTask(ThreadTask&& task){
        //implement something smarter here
        //or maybe I won't need to if/when I implement task stealing
        threadStates[delegationCounter].workQueue.AddTask(std::move(task));
        delegationCounter++;
    }

    private:
    const size_t numThreads;
    const size_t delegationCounter;
    std::unique_ptr<TaskThread<ThreadState>[]> threadStates; 
    std::unique_ptr<std::jthread[]> threadHandles;
    
};


template<typename DistType>
std::vector<std::future<Graph<size_t, DistType>>> InitializeBlockGraphs(const size_t numBlocks, const std::vector<size_t>& blockSizes, const size_t numNeighbors, ThreadPool<ThreadFunctors<DistType>>& threadPool){
    
    //lambda that generates a function that executes the relevant work
    auto taskGenerator = [&](size_t blockNum, size_t blockSize)-> auto{
        std::promise<Graph<size_t, DistType>> taskResult;
        
        auto task = [&, blockNum](ThreadFunctors<DistType>& functors){
            functors.dispatchFunctor.SetBlocks(blockNum, blockNum);
            taskResult.set_value(BruteForceBlock(numNeighbors, blockSize, functors.dispatchFunctor))
        };

        return std::make_pair(std::function(task), taskResult.get_future());
    };

    std::vector<std::future<Graph<size_t, DistType>>> retVector;
    //generate tasks, return futures
    for (size_t i = 0; i<numBlocks; i+=1){
        auto [task, future] = taskGenerator(i, blockSizes[i]);
        threadPool.DelegateTask(std::move(task));
        retVector.push_back(std::move(future));
    }

    return retVector;
    
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
    
    SplittingHeurisitcs splitParams= {16, 140, 60, 180};

    RandomProjectionForest rpTreesTrain(size_t(mnistFashionTrain.numberOfSamples), rngFunctor, splitterFunc, splitParams);



    auto [indexMappings, dataBlocks] = PartitionData<AlignedArray<float>>(rpTreesTrain, mnistFashionTrain);

    
    MetricFunctor<AlignedArray<float>, EuclideanMetricPair> testFunctor(dataBlocks);


    //ThreadPool<ThreadFunctors<float>>

    return 0;
}