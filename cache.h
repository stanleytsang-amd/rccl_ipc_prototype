#include <hip/hip_runtime_api.h>
#include <list>
#include <unordered_map>
#include <functional>

//#include "llvm/ADT/DenseMap.h"

template <    
    class Key,     
    class Value,
    class Hash,
    class KeyEqual,
    class Allocator
>
class IpcHandleCache
{
    typedef std::unordered_map<Key, std::pair<Value, typename std::list<Key>::iterator>, Hash, KeyEqual, Allocator> LRUCache;    
public:
    using iterator = typename LRUCache::iterator;
    IpcHandleCache(size_t size,
                   size_t bucket_count = 100,
                   const Hash& hash = Hash(),
                   const KeyEqual& eql = KeyEqual(),
                   const Allocator& alloc = Allocator() ) : cache(bucket_count, hash, eql, alloc)
    {
        capacity = size;
    } 

    ~IpcHandleCache()
    {
        lruHistory.clear();
        cache.clear();
    }

    iterator begin()
    {
        return cache.begin();
    }

    iterator end()
    {
        return cache.end();
    }

    iterator find(const Key& key)
    {
        iterator it = cache.find(key);
        if (it != cache.end())
        {
            updateHistory(key);
        }

        return it;
    }

    std::pair<iterator, bool> insert(const Key& key, const Value& value)
    {
        if (cache.size() == capacity)
        {
            // remove entry
            pop();
        }
        
        typename LRUCache::iterator it = cache.find(key);
        bool inserted;
        if (it == cache.end())
        {
            typename std::list<Key>::iterator it = lruHistory.insert(lruHistory.end(), key);
            cache.insert(std::make_pair(key, std::make_pair(value, it)));
            inserted = true;
        }
        else
        {
            inserted = false;
        }

        return std::pair<iterator, bool>(it, inserted);
    } 


private:
    void pop()
    {
        typename LRUCache::iterator it = cache.find(lruHistory.front());
        cache.erase(it);
        lruHistory.pop_front();
    } 

    void updateHistory(const Key& key)
    {
        if (lruHistory.size() > 0)
        {
            lruHistory.splice(lruHistory.end(), lruHistory, cache[key].second);
        }
    }
    size_t capacity;
    std::list<Key> lruHistory; 
    LRUCache cache;
};

// djb2 hash function for hashing char array in hipIpcMemHandle_t
unsigned long HandleHash(const hipIpcMemHandle_t& handle)
{
    const char* str = handle.reserved;
    unsigned long hash = 5381;
    int c;

    while ((c = *(str)++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

// equality function required for unordered_map
auto HandleEqual = [](const hipIpcMemHandle_t& l, const hipIpcMemHandle_t& r)
{
    return memcmp(l.reserved, r.reserved, sizeof(l.reserved)) == 0;
};

//typedef llvm::DenseMap<uint64_t, hipIpcMemHandle_t> SendCache;
//typedef llvm::DenseMap<hipIpcMemHandle_t, void*, decltype(&HandleHash), decltype(HandleEqual)> RecvCache;

typedef IpcHandleCache<uint64_t, hipIpcMemHandle_t, std::hash<uint64_t>, std::equal_to<uint64_t>, std::allocator< std::pair<const uint64_t, hipIpcMemHandle_t>>> SendCache;
typedef IpcHandleCache<hipIpcMemHandle_t, void*, decltype(&HandleHash), decltype(HandleEqual), std::allocator< std::pair<const uint64_t, hipIpcMemHandle_t>>> RecvCache;


//typedef std::unordered_map<uint64_t, hipIpcMemHandle_t> SendCache;
//typedef std::unordered_map<hipIpcMemHandle_t, void*, decltype(&HandleHash), decltype(HandleEqual)> RecvCache;

hipIpcMemHandle_t CheckCacheForPtr(void* devPtr, SendCache& cache, int rank)
{
    hipIpcMemHandle_t handle;
    uint64_t addr = (uint64_t)devPtr;
    //std::cout << rank << " finding" << std::endl;
    SendCache::iterator it = cache.find(addr); 
    //std::cout << rank << " done finding" << std::endl;
    if (it == cache.end())
    {
        hipIpcGetMemHandle(&handle, devPtr);       
        std::pair<uint64_t, hipIpcMemHandle_t> ptrHandleMap(addr, handle) ;
        cache.insert(addr, handle);
        //std::cout << rank <<  " done inserting" << std::endl;
    }
    else
    {
        handle = (it->second).first;
    }

    return handle;
}

void* CheckCacheForHandle(hipIpcMemHandle_t handle, RecvCache& cache)
{
    void* ptr = nullptr;
    RecvCache::iterator it = cache.find(handle); 

    if (it == cache.end())
    {
        hipIpcOpenMemHandle((void**)&ptr, handle, hipIpcMemLazyEnablePeerAccess);
        //std::pair<uint64_t, void*> handlePtrMap(target, ptr);
        cache.insert(handle, ptr);
    }
    else
    {
        ptr = (it->second).first;
    }

    return ptr;
}