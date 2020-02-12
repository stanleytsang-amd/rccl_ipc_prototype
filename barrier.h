#include <string>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

class SMBarrier
{
public:
    SMBarrier(int rank, int numProcs, int uniqueId)
    {
        this->numProcs = numProcs;
        std::string uniqueIdString = std::to_string(uniqueId);
        mutexName = std::string("mutex").append(uniqueIdString);
        turnstile1Name = std::string("turnstile1").append(uniqueIdString);
        turnstile2Name = std::string("turnstile2").append(uniqueIdString);
        counterName = std::string("counter").append(uniqueIdString);
        tinyBarrierName = std::string("tinyBarrier").append(uniqueIdString);

        size_t smSize = sizeof(sem_t);

        if (rank == 0)
        {
            initSemaphore(smSize, mutexName, 1, mutex);
            initSemaphore(smSize, turnstile1Name, 0, turnstile1);
            initSemaphore(smSize, turnstile2Name, 0, turnstile2);
            openSharedMemoryVariable(sizeof(int), counterName, true, counter);            
            openSharedMemoryVariable(smSize, tinyBarrierName, true, tinyBarrier);
        }
        else
        {
            openSharedMemoryVariable(smSize, tinyBarrierName, false, tinyBarrier);    
            openSemaphore(smSize, mutexName, mutex);
            openSemaphore(smSize, turnstile1Name, turnstile1);
            openSemaphore(smSize, turnstile2Name, turnstile2);
            openSharedMemoryVariable(sizeof(int), counterName, false, counter);
        }
    }

    void wait()
    {
        part1();
        part2();
    }
    
    ~SMBarrier()
    {
        //if(rank == 0)
        {
            shm_unlink(mutexName.c_str());
            shm_unlink(turnstile1Name.c_str());
            shm_unlink(turnstile2Name.c_str());
            shm_unlink(counterName.c_str());
            shm_unlink(tinyBarrierName.c_str());
        }        
    }
private:
    template <typename T>
    void openSharedMemoryVariable(size_t size, std::string name, bool create, T& val)
    {
        int protection = PROT_READ | PROT_WRITE;
        int visibility = MAP_SHARED;        
        int fd;

        if (create)
        {
            fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);    
            ftruncate(fd, size);
        }    
        else
        {
            do
            {
                // TODO: Error checking so we don't just infinite loop
                fd = shm_open(name.c_str(), O_RDWR, 0666); 
            } while (fd == -1);
        }
        val = (T)mmap(NULL, size, protection, visibility, fd, 0);
        close(fd);
    }

    void initSemaphore(size_t size, std::string name, int semValue, sem_t*& semaphore)
    {
        openSharedMemoryVariable<sem_t*>(size, name, true, semaphore);
        sem_init(semaphore, 1, semValue);
    }

    void openSemaphore(size_t size, std::string name, sem_t*& semaphore)
    {
        openSharedMemoryVariable<sem_t*>(size, name, false, semaphore);
    }

    void part1()
    {
        sem_wait(mutex);       
        if (++(*counter) == numProcs)
        { 
            sem_post_batch(turnstile1, numProcs);
        }
        sem_post(mutex);  
        sem_wait(turnstile1);       
    }

    void part2()
    {      
        sem_wait(mutex);
        if (--(*counter) == 0)
        { 
            sem_post_batch(turnstile2, numProcs);
        }
        sem_post(mutex);       
        sem_wait(turnstile2);      
    }

    int sem_post_batch(sem_t*& sem, int n)
    {
        int ret = 0;
        for (int i = 0; i < n; i++)
        {
            ret = sem_post(sem);
            if (ret != 0) break;
        }

        return ret;
    }
    int numProcs;
    
    int* counter;

    sem_t* mutex;
    sem_t* turnstile1;
    sem_t* turnstile2;
    sem_t* tinyBarrier;

    std::string mutexName;
    std::string turnstile1Name;
    std::string turnstile2Name;
    std::string tinyBarrierName;
    std::string counterName;
};