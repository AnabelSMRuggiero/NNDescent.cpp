/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_THREADPOOL_HPP
#define NND_THREADPOOL_HPP

#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <memory_resource>
#include <new>
#include <ranges>
#include <thread>
#include <type_traits>

#include "ann/MemoryResources.hpp"
#include "ann/AlignedMemory/DynamicArray.hpp"
#include "ann/DelayConstruct.hpp"

#include "NND/MemoryInternals.hpp"

#include "AsyncQueue.hpp"

namespace nnd{

template<typename ThreadState>
struct ThreadPool;

template<typename ThreadState>
struct TaskThread{

    using ThreadTask = std::function<void(ThreadState&)>;

    friend ThreadPool<ThreadState>;

    AsyncQueue<ThreadTask> workQueue;

    TaskThread() requires std::is_default_constructible_v<ThreadState> = default;

    

    TaskThread(TaskThread&& other) = default;


    
    template<typename ...ThreadStateArgs>
    TaskThread(ThreadStateArgs... args): state(args...), workQueue(), running(false) {};

    auto GetThreadLoop(){
        auto threadLoop = [&](std::stop_token stopToken)->void{

            assert(!(this->running));

            auto queueTerminator = [&]() mutable {
                this->workQueue.Put(this->GetTerminator());
            };

            std::stop_callback stopQueuer(stopToken, queueTerminator);

            ann::ShallowCache<5> threadCache;
            internal::SetThreadResource(&threadCache);
            this->running = true;
            while(running){

                ThreadTask taskToDo = this->workQueue.Take();
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

//Not sure why the copy/move constructors of this specialization are generating errors.
//The errors say the defaulted constructors are deleted because of AsyncQueue, but that doesn't affect
//the unspecialized template. /shrug

template<>
struct TaskThread<void>{

    using ThreadTask = std::function<void(void)>;

    friend ThreadPool<void>;

    AsyncQueue<ThreadTask> workQueue;

    TaskThread() = default;

    

    TaskThread(TaskThread<void>&& other) = default;


    
    template<typename ...ThreadStateArgs>
    TaskThread(ThreadStateArgs... args): workQueue(), running(false) {};

    auto GetThreadLoop(){
        auto threadLoop = [&](std::stop_token stopToken)->void{

            assert(!(this->running));

            auto queueTerminator = [&]() mutable {
                this->workQueue.Put(this->GetTerminator());
            };


            std::stop_callback stopQueuer(stopToken, queueTerminator);

            this->running = true;
            while(running){

                ThreadTask taskToDo = this->workQueue.Take();
                taskToDo();

            }

        };

        return threadLoop;
    }

    ThreadTask GetTerminator(){
        return [&](void){
            running = false;
        };
    }


    private:
    TaskThread(const TaskThread<void>&) = default;
    TaskThread<void>& operator=(const TaskThread<void>&) = default;
    bool running = false;

};

template<typename ThreadState>
struct ThreadPool{

    using state = ThreadState;
    using ThreadTask = std::function<void(ThreadState&)>;

    ThreadPool() requires std::is_default_constructible_v<ThreadState>: numThreads(std::jthread::hardware_concurrency()),
                  delegationCounter(0),
                  threadStates(numThreads), 
                  threadHandles(numThreads) {};

    ThreadPool(const size_t numThreads) requires std::is_default_constructible_v<ThreadState>: numThreads{numThreads},
                                         delegationCounter{0},
                                         threadStates{numThreads}, 
                                         threadHandles{numThreads} {};
    
    template<typename... ThreadStateArgs>
        requires std::constructible_from<TaskThread<ThreadState>, ThreadStateArgs...>
    ThreadPool(const size_t numThreads, ThreadStateArgs... args): numThreads{numThreads},
                                         delegationCounter{0},
                                         threadStates([&](){
                                            return TaskThread<ThreadState>{args...};
                                         }, numThreads), 
                                         threadHandles{numThreads}{}

    template<typename ...ThreadStateArgs>
    void RebuildStates(ThreadStateArgs... args){
        for (size_t i = 0; i<numThreads; i += 1){
            ThreadState* statePtr = &(threadStates[i].state);
            try{
                statePtr->~ThreadState();
                new (statePtr) ThreadState(args...);
            } catch (...){
                //ensure a valid state is in place before raising
                new (statePtr) ThreadState();
                throw;
            }
        }
    }

    void StartThreads(){
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
        DelegateTask(std::move(task), [](const size_t count){ return count < 15; });
    }

    template<typename Predicate>
    void DelegateTask(ThreadTask&& task, Predicate pred){
        //implement something smarter here
        //or maybe I won't need to if/when I implement task stealing
        size_t numQueuesChecked(0);
        while(numQueuesChecked<numThreads){
            if (pred(threadStates[delegationCounter].workQueue.GetCount())) break;
            numQueuesChecked += 1;
            delegationCounter = (delegationCounter+1)%numThreads;
        }
        if (numQueuesChecked == numThreads){
            while(!pred(threadStates[delegationCounter].workQueue.WaitOnCount()));
        }
        threadStates[delegationCounter].workQueue.Put(std::move(task));
        delegationCounter = (delegationCounter+1)%numThreads;

    }

    size_t ThreadCount(){
        return numThreads;
    }

    size_t TaskCount(){
        size_t count(0);
        for (size_t i = 0; i<numThreads; i+=1){
            count += threadStates[i].workQueue.GetCount();
        }
        return count;
    }

    void WaitOnCounts(size_t num = 0){
        auto waiter = [num](const size_t count){return count < num; };
        for (size_t i = 0; i<numThreads; i+=1){
            while(!waiter(threadStates[i].workQueue.WaitOnCount()));
        }
    }

    //I hate that I have to write this, but should be a temporary crutch
    void Latch(){
        auto latcher = [](const size_t count){return count == 0; };
        auto noopTask = [](ThreadState&)->void{return;};
        for (size_t i = 0; i<numThreads; i+=1){
            threadStates[i].workQueue.Put(noopTask);
        }
        for (size_t i = 0; i<numThreads; i+=1){
            while(!latcher(threadStates[i].workQueue.WaitOnCount()));
        }  
    }

    private:
    
    const size_t numThreads;
    size_t delegationCounter;
    ann::dynamic_array<TaskThread<ThreadState>> threadStates; 
    ann::dynamic_array<std::jthread> threadHandles;
    
};

template<>
struct ThreadPool<void>{


    using ThreadTask = std::function<void(void)>;

    ThreadPool(): numThreads(std::jthread::hardware_concurrency()),
                  delegationCounter(0),
                  threadStates(std::make_unique<TaskThread<void>[]>(numThreads)), 
                  threadHandles(std::make_unique<std::jthread[]>(numThreads)) {};

    ThreadPool(const size_t numThreads): numThreads(numThreads),
                                         delegationCounter(0),
                                         threadStates(std::make_unique<TaskThread<void>[]>(numThreads)), 
                                         threadHandles(std::make_unique<std::jthread[]>(numThreads)) {};


    void StartThreads(){
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
        DelegateTask(std::forward<ThreadTask>(task), [](const size_t count){ return count < 10; });
    }

    template<typename Predicate>
    void DelegateTask(ThreadTask&& task, Predicate pred){
        //implement something smarter here
        //or maybe I won't need to if/when I implement task stealing
        size_t numQueuesChecked(0);
        while(numQueuesChecked<numThreads){
            if (pred(threadStates[delegationCounter].workQueue.GetCount())) break;
            numQueuesChecked += 1;
            delegationCounter = (delegationCounter+1)%numThreads;
        }
        if (numQueuesChecked == numThreads){
            while(!pred(threadStates[delegationCounter].workQueue.WaitOnCount())){
                delegationCounter = (delegationCounter+1)%numThreads;
            };
        }
        threadStates[delegationCounter].workQueue.Put(std::move(task));
        delegationCounter = (delegationCounter+1)%numThreads;

    }

    size_t ThreadCount(){
        return numThreads;
    }

    //I hate that I have to write this, but should be a temporary crutch
    void Latch(){
        auto latcher = [](const size_t count){return count == 0; };
        auto noopTask = [](void)->void{return;};
        for (size_t i = 0; i<numThreads; i+=1){
            threadStates[i].workQueue.Put(noopTask);
        }
        for (size_t i = 0; i<numThreads; i+=1){
            while(!latcher(threadStates[i].workQueue.WaitOnCount()));
        }
        
        
    }

    private:
    
    const size_t numThreads;
    size_t delegationCounter;
    std::unique_ptr<TaskThread<void>[]> threadStates; 
    std::unique_ptr<std::jthread[]> threadHandles;
    
};

template<typename ThreadState, std::invocable<> Region>
void threaded_region(ThreadPool<ThreadState>& pool, Region&& region){
    pool.StartThreads();
    std::invoke(region);
    pool.StopThreads();
}

}

#endif