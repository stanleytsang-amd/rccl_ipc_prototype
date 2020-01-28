#include <unordered_map>
#include <functional>

#include "hip/hip_runtime.h"

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

typedef std::unordered_map<uint64_t, hipIpcMemHandle_t> SendCache;
typedef std::unordered_map<hipIpcMemHandle_t, void*, decltype(&HandleHash), decltype(HandleEqual)> RecvCache;

hipIpcMemHandle_t CheckCacheForPtr(void* devPtr, SendCache& cache)
{
    hipIpcMemHandle_t handle;
    uint64_t addr = (uint64_t)devPtr;
    SendCache::iterator it = cache.find(addr); 

    if (it == cache.end())
    {
        hipIpcGetMemHandle(&handle, devPtr);       
        std::pair<uint64_t, hipIpcMemHandle_t> ptrHandleMap(addr, handle) ;
        cache.insert(ptrHandleMap);
    }
    else
    {
        handle = it->second;
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
        cache.insert(std::make_pair(handle, ptr));
    }
    else
    {
        ptr = it->second;
    }

    return ptr;
}