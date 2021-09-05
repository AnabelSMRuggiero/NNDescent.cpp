/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_ASYNCQUEUE_HPP
#define NND_ASYNCQUEUE_HPP

#include <mutex>
#include <vector>
#include <list>
#include <atomic>
#include <condition_variable>
#include <optional>

namespace nnd{

template<typename Element>
struct AsyncQueue{

    using ElementType = Element;

    AsyncQueue& operator=(AsyncQueue&& rhs) = default;

    void Put(Element&& task){
        std::unique_lock queueLock(queueMutex);
        tasks.push_back(std::move(task));
        taskCounter++;
        
        queueLock.unlock();
        queueUpdated.notify_one();
    }

    void Put(std::vector<Element>&& newTasks){
        std::unique_lock queueLock(queueMutex);
        for (auto& task: newTasks){
            tasks.push_back(std::move(task));
            taskCounter++;
        }

        queueLock.unlock();
        queueUpdated.notify_one();
    }

    void Put(std::list<Element>&& newTasks){
        std::unique_lock queueLock(queueMutex);
        taskCounter += newTasks.size();
        tasks.splice(tasks.end(), std::move(newTasks));
            
        queueLock.unlock();
        queueUpdated.notify_one();
    }

    Element Take(){
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0){
            queueUpdated.wait(queueLock, [&]{return this->taskCounter != 0;});
        }
        taskCounter--;
        Element retTask = std::move(tasks.front());
        tasks.pop_front();

        queueLock.unlock();
        queueUpdated.notify_all();
        
        return retTask;
    }

    std::optional<Element> TryTake(){
        if (taskCounter == 0) return std::nullopt;
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0) return std::nullopt;
        taskCounter--;
        Element retTask = std::move(tasks.front());
        tasks.pop_front();

        queueLock.unlock();
        queueUpdated.notify_all();
        
        return retTask;
    }
    
    std::list<Element> TakeAll(){
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0){
            queueUpdated.wait(queueLock, [&]{return this->taskCounter != 0;});
        }
        taskCounter = 0;
        std::list<Element> retTasks = std::move(tasks);
        tasks = std::list<Element>();

        queueLock.unlock();
        queueUpdated.notify_all();

        return retTasks;
    }

    std::list<Element> TryTakeAll(){
        if (taskCounter == 0) return std::list<Element>();
        std::unique_lock queueLock(queueMutex);
        if (taskCounter == 0) return std::list<Element>();
        taskCounter = 0;
        std::list<Element> retTasks = std::move(tasks);
        tasks = std::list<Element>();

        queueLock.unlock();
        queueUpdated.notify_all();

        return retTasks;
    }

    operator bool(){
        return taskCounter != 0;
    }

    size_t GetCount() const{
        return taskCounter.load();
    }
    
    size_t WaitOnCount() const{
        std::unique_lock lock(queueMutex);
        if(GetCount() == 0) return 0;  
        queueUpdated.wait(lock);
        return GetCount();
    }
    
    private:
    std::list<Element> tasks;
    std::mutex queueMutex;
    std::atomic<size_t> taskCounter; //Leaving this in for now for expanding functionality
    std::condition_variable queueUpdated;

};



}


#endif
