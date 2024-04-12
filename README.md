# Robust Spanner Detection


Dependencies
-----------------------------------------------
- cmake  
- CUDA 11.6 or later  
- SYCL  

Install Intel LLVM SYCL from 
https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda



Build RS
------------------------------------------------
- Change compiler (CXX) and compiler flags (CXXFLAGS) according to your environment.  

build:  
```
make
```

run:  
```
./op_R_spannerNew -g test_graph.mtx -c community_test_graph.txt  -t 4 -t 1 -t 3 -t 5 -t 2
```

arguments:  
g : graph file in .mtx format  
c : community file  
t : targeted community IDs  

remove built executable:
```
make clean
```