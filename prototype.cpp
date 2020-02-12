#include <mpi.h>
#include <omp.h>
#include <iomanip>
#include <iostream>
#include <fcntl.h> 
#include <sys/shm.h> 
#include <sys/stat.h> 
#include <stdlib.h> 
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h> 

#include "barrier.h"
#include "common.h"
#include "cache.h"

// Records the time required for all ranks to complete measured operation
void SyncAndPrintElapsedTime(double start, double end, double globalStart, double globalEnd, int rank, const char* taskDesc, double& elapsedTime);
// Records the maximum time elapsed by the slowest rank
void SyncAndPrintElapsedTimeV2(double start, double end, int rank, const char* taskDesc, double& globalElapsedTime);
void CheckData(int* hostData, int rank, int checkRank, unsigned int numElements);

// TODO: Not use MPI since MPI is not an option in RCCL
int main(int argc, char *argv[]) {
   int useCache, iterations, cacheSize;

   if (argc < 4)
   {
      std::cout << "Please specify 0 for no cache, 1 for cache; then specify number of iterations; then specify number of elements in cache." << std::endl;
      return 1;
   }
   useCache = std::atoi(argv[1]);   
   iterations = std::atoi(argv[2]); 
   
   cacheSize = std::atoi(argv[3]);

   // Initial MPI setup
   MPI_Init(NULL, NULL);

   int numRanks, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (useCache && rank == 0)
   {
         std::cout << "Using cache implementation." << std::endl;
   }

   double start, end, globalStart, globalEnd;   

   // Shared memory stuff
   const size_t smHandlesSize = numRanks * 2 * sizeof(hipIpcMemHandle_t);
   const  size_t smBarrierSize = sizeof(SMBarrier);  
   int shm_fd_handles; 
   hipIpcMemHandle_t* shmemHandles;

   // Cache stuff
   SendCache sendCache(cacheSize);
   RecvCache recvCache(cacheSize, 100, HandleHash, HandleEqual);

   size_t idx = rank * 2;
   int targetRank = (rank == numRanks - 1) ? 0 : rank + 1;

   // HIP stuff
   int* devPtrs[2];
   hipIpcMemHandle_t gpuHandles[2];   
   hipIpcMemHandle_t recvHandles[NUM_HANDLES_TOTAL];

   // HIP-related setup
   size_t gpuDataSize = numElements * sizeof(int);      
   
   int* hostData;
   hostData = new int[numElements];

   HIPCHECK(hipSetDevice(rank % 4));
   HIPCHECK(hipMalloc(&devPtrs[0], gpuDataSize));
   HIPCHECK(hipMalloc(&devPtrs[1], gpuDataSize));

   int* otherDevPtr[NUM_HANDLES_TOTAL];
   
   // Rank 0 creates shared memory regions - one for storing handles, one for storing shared barrier
   if (rank == 0)
   {
      shm_fd_handles = shm_open(smHandlesName, O_CREAT | O_RDWR, 0666);
      ftruncate(shm_fd_handles, smHandlesSize);

      int protection = PROT_READ | PROT_WRITE;
      int visibility = MAP_SHARED;
   
      shmemHandles = (hipIpcMemHandle_t*) mmap(NULL, smHandlesSize, protection, visibility, shm_fd_handles, 0);
   }

   // Other ranks open shared memory object created by rank 0
   if (rank != 0)
   {
      do
      {
         shm_fd_handles = shm_open(smHandlesName, O_RDWR, 0666); 
      } while (shm_fd_handles == -1);
      shmemHandles = (hipIpcMemHandle_t*)mmap(0, smHandlesSize, PROT_WRITE, MAP_SHARED, shm_fd_handles, 0);         
   }

   srand (time(NULL));
   int ncclUniqueId = rand() % 32767;

   // Initialize Barrier first
   SMBarrier shmemBarrier(rank, numRanks, ncclUniqueId);
   //MPI_Barrier(MPI_COMM_WORLD);
   
   // Ensure all ranks have opened shared memory before proceeding
   shmemBarrier.wait();

   //std::cout << std::setprecision(7) << std::fixed;
   double averageTime;   
   for(int iteration = 0; iteration < iterations; iteration++)
   {
      double elapsedTime = 0.0;
      if (useCache)
      {
         start = omp_get_wtime();
         gpuHandles[0] = CheckCacheForPtr((void*)devPtrs[0], sendCache, rank);
         gpuHandles[1] = CheckCacheForPtr((void*)devPtrs[1], sendCache, rank);
         end = omp_get_wtime();
      }
      else
      {
         start = omp_get_wtime();
         HIPCHECK(hipIpcGetMemHandle(&gpuHandles[0], devPtrs[0]));
         HIPCHECK(hipIpcGetMemHandle(&gpuHandles[1], devPtrs[1]));   
         end = omp_get_wtime();
      }
      
      SyncAndPrintElapsedTime(start, end, globalStart, globalEnd, rank, "getting handles", elapsedTime);
      //SyncAndPrintElapsedTimeV2(start, end, rank, "getting handles", elapsedTime);

      // Write own handle to shared memory
      start = omp_get_wtime();
      memcpy(shmemHandles + idx, gpuHandles, sizeof(hipIpcMemHandle_t) * 2);
      end = omp_get_wtime();   
      SyncAndPrintElapsedTime(start, end, globalStart, globalEnd, rank, "writing to shared memory twice", elapsedTime);
      
      //MPI_Barrier(MPI_COMM_WORLD);
      // Receive all handles from memory
      start = omp_get_wtime();
      memcpy(recvHandles, shmemHandles, smHandlesSize);
      end = omp_get_wtime();   
      SyncAndPrintElapsedTime(start, end, globalStart, globalEnd, rank, "reading from shared memory twice", elapsedTime);

      if (useCache)
      {
         start = omp_get_wtime();
         for (int i = 0; i < NUM_HANDLES_TOTAL; i++)
         {         
            otherDevPtr[i] = (int*)CheckCacheForHandle(recvHandles[i], recvCache);
         }
         end = omp_get_wtime();        
      }
      else
      {
         start = omp_get_wtime();

         for (int i = 0; i < NUM_HANDLES_TOTAL; i++)
         {     
            HIPCHECK(hipIpcOpenMemHandle((void**)&otherDevPtr[i], recvHandles[i], hipIpcMemLazyEnablePeerAccess));
         }
         end = omp_get_wtime();   
      }
      
      SyncAndPrintElapsedTime(start, end, globalStart, globalEnd, rank, "opening hip IPC handles", elapsedTime);
      //SyncAndPrintElapsedTimeV2(start, end, rank, "opening hip IPC handles", elapsedTime);
      
      if (rank == 0)
      {
         averageTime += elapsedTime;
         std::cout << "--------------------" << std::endl;
      }

      dim3 grid = { 1, 1, 1 };
      dim3 block = { numElements, 1, 1 };
      hipLaunchKernelGGL((setData), grid, block, 0, 0, otherDevPtr[targetRank * 2], rank, numElements);   
      hipLaunchKernelGGL((setData), grid, block, 0, 0, otherDevPtr[(targetRank * 2)+1], rank, numElements); 
      // Ensure all ranks' kernels have completed before checking data
      HIPCHECK(hipDeviceSynchronize());
      shmemBarrier.wait();
   }

   // Check data
   int checkRank = (rank == 0) ? numRanks - 1 : rank - 1;   

   HIPCHECK(hipMemcpy(hostData, devPtrs[0], gpuDataSize, hipMemcpyDeviceToHost));
   CheckData(hostData, rank, checkRank, numElements);

   HIPCHECK(hipMemcpy(hostData, devPtrs[1], gpuDataSize, hipMemcpyDeviceToHost));
   CheckData(hostData, rank, checkRank, numElements); 
   shmemBarrier.wait();

   if (rank == 0)
   {
      averageTime /= (double)iterations;
      std::cout << "Average time required for measured operations (" << iterations << " iterations): " << averageTime << " seconds" << std::endl;
      munmap((void*)shmemHandles, smHandlesSize);
   }

   for (int i = 0; i < NUM_HANDLES_TOTAL; i++)
   {
      if (useCache)
      {
         HIPCHECK(hipIpcCloseMemHandle((void*)otherDevPtr[i]));
      }
      else
      {
         HIPCHECK(hipIpcCloseMemHandle((void*)otherDevPtr[i]));
      }
   }

   shmemBarrier.wait();

   HIPCHECK(hipFree(devPtrs[0]));   
   HIPCHECK(hipFree(devPtrs[1]));  
   shm_unlink(smHandlesName);
   
   delete hostData;

   MPI_Finalize();
   return 0;
}


void SyncAndPrintElapsedTime(double start, double end, double globalStart, double globalEnd, int rank, const char* taskDesc, double& globalElapsedTime)
{
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&start, &globalStart, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&end, &globalEnd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);   

   if (rank == 0)
   {
      std::cout << "Time elapsed for " << taskDesc << ": " << globalEnd - globalStart << " seconds." << std::endl;
      globalElapsedTime += (globalEnd - globalStart);
   }   
}

void SyncAndPrintElapsedTimeV2(double start, double end, int rank, const char* taskDesc, double& globalElapsedTime)
{
   double elapsedTime = end - start;
   double reducedTime;
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(&elapsedTime, &reducedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);   

   if (rank == 0)
   {
      std::cout << "Time elapsed for " << taskDesc << ": " << reducedTime << " seconds." << std::endl;
      globalElapsedTime += reducedTime;
   }   
}

void CheckData(int* hostData, int rank, int checkRank, unsigned int numElements)
{
   bool pass = true;
   for (unsigned int i = 0; i < numElements; i++)
   {
      if (hostData[i] != checkRank)
      {
         pass = false;
         std::cout << "Mismatch at index " << i << ".  Actual value: " << hostData[i];
         break;
      }
   }
}