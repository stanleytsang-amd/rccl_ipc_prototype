# rccl_ipc_prototype

To build, type 'make'.

Prototype is currently hardcoded for 8 MPI processes and will break if not run with exactly 8 MPI processes.

The prototype takes 2 mandatory arguments.  
First argument toggles the cache on/off - '1' for on, '0' for off.  
Second argument is the number of iterations.

ie. 'mpirun -np 8 prototype 1 1000'
