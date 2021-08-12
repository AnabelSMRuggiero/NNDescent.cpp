/*
NNDescent.cpp: Copyright (c) Anabel Ruggiero
At the time of writting, this code is unreleased and not published under a license.
As a result, I currently retain all legal rights I am legally entitled to.

I am currently considering a permissive license for releasing this code, such as the Apache 2.0 w/LLVM exception.
Please refer to the project repo for any updates regarding liscensing.
https://github.com/AnabelSMRuggiero/NNDescent.cpp
*/

#ifndef NND_HASHLOCK_HPP
#define NND_HASHLOCK_HPP

#include <functional>
#include <atomic>
#include <utility>
#include <cassert>
#include <memory>
#include <optional>

namespace nnd{






template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashKey;

template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashPtr;

template<typename Locked, typename Hasher = std::hash<size_t>>
class HashLock{

    friend HashLocked<Locked, Hasher>;
    friend HashLocked<Locked[], Hasher>;
    friend HashKey<Locked, Hasher>;
    friend HashPtr<Locked, Hasher>;

    HashLock(): lockedValue(), lock(0), nextLock(reinterpret_cast<size_t>(this)) {};

    HashLock(Locked&& lockedValue): lockedValue(std::forward<Locked>(lockedValue)), lock(), nextLock(reinterpret_cast<size_t>(this)) {};

    template<typename... ConstructorArgs>
    HashLock(ConstructorArgs args...): lockedValue(args...), lock(), nextLock(reinterpret_cast<size_t>(this)) {}
    

    [[nodiscard]] HashKey GetKey(std::weak_ptr<void> weakPtr){
        size_t expectZero(0);
        if(lock.compare_exchange_strong(expectZero, nextLock)){
            HashKey retKey{nextLock, this, weakPtr};
            nextLock = hash(nextLock) | 1;
            return retKey;
        } else return {0, nullptr, std::weak_ptr<void>()}; 
    }

    [[nodiscard]] std::optional<HashPtr<Locked,Hasher>> GetAccess(size_t keyVal, std::shared_ptr<void> ownership){
        size_t expectedVal = key.key;
        if(lock.compare_exchange_strong(expectedVal, nextLock)){
            std::optional<HashPtr<Locked,Hasher>> retPtr = std::make_optional(HashPtr<Locked,Hasher>{nextLock, std::shared_ptr(ownership, &lockedValue)});
            nextLock = hash(nextLock) | 1;
            return retPtr;
        } else return std::nullopt;
    }
    
    bool Unlock(size_t key){
        return lock.compare_exchange_strong(key, 0);
    }

    Locked lockedValue;
    std::atomic<size_t> lock;
    size_t nextLock;
    [[no_unique_address]] Hasher hash;

};

template<typename PtrLike, typename Compose>
struct HashRelease{

    using pointer = PtrLike;

    void operator()(PtrLike keyLike){
        keyLike->Unlock();
        composedFunction(keyLike);
    }

    private:
    [[no_unique_address]] Compose composedFunction;
};

template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashKeyImpl{

    using Lock = HashLock<Locked, Hasher>;



    operator bool() const {
        return key != 0 && lock != nullptr;
    }

    [[nodiscard]] std::optional<HashPtr<Locked,Hasher>> GetAccess() const{
        //assert(lock != nullptr);
        std::shared_ptr<void> ownerShip = weakPtr.lock();
        if (!ownerShip) std::nullopt;
        return lock->GetAccess(key, ownerShip);
    }

    private:

    bool Unlock() const {
        return lock->Unlock(key);
    }

    const size_t key;
    HashLock* lock;
    std::weak_ptr<void> weakPtr;
};


template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashKey{
    [[nodiscard]] std::optional<HashPtr<Locked,Hasher>> GetAccess() const{
        return ptrToImpl->GetAccess();
    }

    private:
    std::shared_ptr<HashKeyImpl<Locked, Hasher>> ptrToImpl;
};

template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashPtr{

    HashPtr(const HashPtr&) = delete;
    HashPtr& operator=(const HashPtr&) = delete;

    operator bool(){
        return bool(accessor);
    }

    Locked* operator->(){
        return accessor->*lockedValue;
    }

    Locked& operator*(){
        return accessor->lockedValue;
    }

    /*
    ~HashPtr(){
        //size_t expectedVal = unlock;
        
        if (accessor != nullptr){
            assert(accessor->Unlock(unlock));
        }
        
    }
    */

    private:

    bool Unlock() const {
        return lock->Unlock(unlock);
    }

    size_t unlock;
    std::shared_ptr<HashLock<Locked, Hasher>> lock;
};
//template<typename Locked, typename Hasher = std::hash<size_t>>


template<typename Locked, typename Hasher = std::hash<size_t>>
struct HashLocked{

    using Lock = HashLock<Locked, Hasher>;

    HashLocked(Locked&& lockedValue): hashLock(std::make_shared<Lock>(std::forward<Locked>(lockedValue))) {};
    
    template<typename... ConstructorArgs>
    HashLocked(ConstructorArgs&& args...): hashLock(std::make_shared<Lock>(std::forward<ConstructorArgs>(args)...)) {};
    
    

    private:
    //Locked lockedValue;
    //HashLock<Hasher> lock;
    std::shared_ptr<Lock> hashLock;
};

template<typename Locked, typename Hasher>
struct HashLocked<Locked[], Hasher>{

    using Lock = HashLock<Locked, Hasher>;

    HashLocked(size_t size): hashLocks(std::make_shared<Lock>(size)) {};

    private:
    std::shared_ptr<Locked[]> hashLocks;

};

}

#endif