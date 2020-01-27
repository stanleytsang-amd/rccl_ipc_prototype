#include <unordered_map>

#include "hip/hip_runtime.h"

typedef std::unordered_map<uint64_t, hipIpcMemHandle_t> SendCache;
typedef std::unordered_map<uint64_t, void*> RecvCache;

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
    uint64_t target = *((uint64_t*)handle.reserved);
    RecvCache::iterator it = cache.find(target); 

    if (it == cache.end())
    {
        hipIpcOpenMemHandle((void**)&ptr, handle, hipIpcMemLazyEnablePeerAccess);
        //std::pair<uint64_t, void*> handlePtrMap(target, ptr);
        cache.insert(std::make_pair(target, ptr));
    }
    else
    {
        ptr = it->second;
    }

    return ptr;
}
/* 
void CheckCacheForHandle(hipIpcMemHandle_t handle, RecvCache& cache, void*& ptr)
{
    //void* ptr = nullptr;
    uint64_t target = *((uint64_t*)handle.reserved);
    RecvCache::iterator it = cache.find(target); 

    if (it == cache.end())
    {
        hipIpcOpenMemHandle((void**)&ptr, handle, hipIpcMemLazyEnablePeerAccess);
        std::pair<uint64_t, void*> handlePtrMap(target, ptr);
        cache.insert(handlePtrMap);
    }
    else
    {
        ptr = it->second;
    }
} */