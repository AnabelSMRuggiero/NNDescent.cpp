/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_TASKQUEUER_HPP
#define NND_TASKQUEUER_HPP

#include <tuple>
#include <vector>
#include <utility>
#include "AsyncQueue.hpp"

namespace nnd{

template<typename Element>
void EraseNulls(std::vector<std::optional<Element>>& optVector){
    optVector.erase(std::remove_if(optVector.begin(), optVector.end(), std::logical_not()),
                             optVector.end());  
}

template<typename GenType, typename ConsType>
struct TaskQueuer{
    using Generator = GenType;
    using Consumer = ConsType;

    static_assert(std::same_as<typename Generator::TaskResult, typename Consumer::TaskResult>);
    using TaskResult = typename Generator::TaskResult;

    using TasksToDo = std::vector<std::optional<typename Generator::TaskArgs>>;
    
    
    template<typename NextStepComponent>
    static constexpr bool hasValidOnCompletion = requires(Consumer cons, NextStepComponent& nextGen){
        cons.OnCompletion(nextGen);
    };
    /*
    template<>
    static constexpr bool hasValidOnCompletion<void> = false;
    */
    template<typename NextGenerator, typename NextConsumer>
    struct OnCompletionTraits
    {   
        
        private:
        template<bool validGen, bool validCons>
        struct Arg{
            using type = void;
        };

        template<>
        struct Arg<true, false>{
            using type = NextGenerator;
        };

        template<>
        struct Arg<false, true>{
            using type = NextConsumer;
        };
        
        public:
        using type = typename Arg<hasValidOnCompletion<NextGenerator>, hasValidOnCompletion<NextConsumer>>::type;
    };
    

    Generator generator;
    Consumer consumer;

    

    template<typename... GenArgs, typename... ConsArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs): 
        generator(std::make_from_tuple<Generator>(std::forward<std::tuple<GenArgs...>>(generatorArgs))),
        consumer(std::make_from_tuple<Consumer>(std::forward<std::tuple<ConsArgs...>>(consumerArgs))),
        doneGenerating(false),
        doneConsuming(false){};

    TaskQueuer(const TaskQueuer&) = delete;
    TaskQueuer(TaskQueuer&&) = delete;

    bool ConsumeResults(){
        std::list<TaskResult> newResults = incomingResults.TryTakeAll();
        for (auto& entry: newResults){
            doneConsuming = consumer(std::move(entry));
        }
        return doneConsuming;
    }

    template<typename NextTask>
    bool ConsumeResults(NextTask& nextGen){
        std::list<TaskResult> newResults = incomingResults.TryTakeAll();
        for (auto& entry: newResults){
            doneConsuming = consumer(std::move(entry));
        }
        if constexpr (hasValidOnCompletion<typename NextTask::Generator>){
            if (doneConsuming) consumer.OnCompletion(nextGen.generator);
        } else if constexpr (!std::is_void_v<typename NextTask::Consumer>){
            if constexpr (hasValidOnCompletion<typename NextTask::Consumer>){
                if (doneConsuming) consumer.OnCompletion(nextGen.consumer);
            }
        }
        return doneConsuming;
    }

    template<typename Pool>
    bool QueueTasks(Pool& pool, TasksToDo& tasks){
        doneGenerating = generator(pool, incomingResults, tasks);
        return doneGenerating;
    }

    auto& GetTaskArgs(){
        return consumer.GetTaskArgs();
    }

    bool DoneGenerating(){
        return doneGenerating;
    }

    bool DoneConsuming(){
        return doneConsuming;
    }

    bool IsDone(){
        return DoneConsuming() && DoneGenerating();
    }


    private:
    bool doneGenerating;
    bool doneConsuming;
    AsyncQueue<TaskResult> incomingResults;

};

template<typename GenType>
struct TaskQueuer<GenType, void>{
    using Generator = GenType;
    using Consumer = void;

    using TaskResult = typename Generator::TaskResult;
    static_assert(std::is_void_v<TaskResult>);

    using TasksToDo = std::vector<std::optional<typename Generator::TaskArgs>>;

    Generator generator;

    template<typename... GenArgs>
    TaskQueuer(std::tuple<GenArgs...>&& generatorArgs): 
        generator(std::make_from_tuple<Generator>(std::forward<std::tuple<GenArgs...>>(generatorArgs))){};

    TaskQueuer(const TaskQueuer&) = delete;
    TaskQueuer(TaskQueuer&&) = delete;


    template<typename Pool>
    bool QueueTasks(Pool& pool, TasksToDo& tasks){
        doneGenerating = generator(pool, tasks);
        return doneGenerating;
    }

    bool DoneGenerating(){
        return doneGenerating;
    }

    bool DoneConsuming(){
        return true;
    }

    bool IsDone(){
        return DoneGenerating();
    }


    private:
    bool doneGenerating;

};

template<typename Task, typename... GenArgs, typename... ConsArgs>
auto GenerateTaskBuilder(std::tuple<GenArgs...>&& generatorArgs, std::tuple<ConsArgs...>&& consumerArgs){
    struct {
        std::tuple<GenArgs...> genArgs;
        std::tuple<ConsArgs...> consArgs;

        operator Task(){
            return Task(std::move(genArgs), std::move(consArgs));
        }
    } builder{std::move(generatorArgs), std::move(consumerArgs)};
    return builder;
}

template<typename Task, typename... GenArgs>
auto GenerateTaskBuilder(std::tuple<GenArgs...>&& generatorArgs){
    static_assert(std::is_void_v<typename Task::Consumer>);

    struct {
        std::tuple<GenArgs...> genArgs;

        operator Task(){
            return Task(std::move(genArgs));
        }
    } builder{std::move(generatorArgs)};

    return builder;
}

//Pipeline code
template<typename Task, typename NextTask>
constexpr static bool consumeWithNext = requires(Task cons, NextTask& nextGen){
    cons.ConsumeResults(nextGen);
};

template<size_t idx, typename TaskTuple>
consteval bool ConsumeHelper(){
    return consumeWithNext<typename std::tuple_element<idx, TaskTuple>::type,
                           typename std::tuple_element<idx+1, TaskTuple>::type>;
}

template<size_t idx, typename... Tasks, typename... TaskArgs, typename TaskPool>
void TaskLoopBody(std::tuple<Tasks...>& tasks, std::tuple<TaskArgs...>& taskArgs, TaskPool& pool){
    if constexpr(!std::is_void_v<typename std::tuple_element_t<idx, std::tuple<Tasks...>>::Consumer>){
        if (std::get<idx>(tasks).DoneConsuming()) return;
    }
    std::get<idx>(tasks).QueueTasks(pool, std::get<idx>(taskArgs));
    if constexpr (!std::is_void_v<typename std::tuple_element<idx, std::tuple<Tasks...>>::type::Consumer>){
        if constexpr ((idx != sizeof...(Tasks)-1) && ConsumeHelper<idx, std::remove_reference_t<decltype(tasks)>>()){
            std::get<idx>(tasks).ConsumeResults(std::get<idx+1>(tasks));
        } else {
            std::get<idx>(tasks).ConsumeResults();
        }
    }
};



}

#endif