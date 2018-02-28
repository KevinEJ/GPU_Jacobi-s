
cpu:
	g++ -o cpu ./CPU/Jacobi_CPU.cpp -O3 -g

#NVCC = nvcc -std=c++11 -lineinfo -O3 -g
NVCC = nvcc -std=c++11 -O3 

Org: ./Organized/*
	$(NVCC) -o Org ./Organized/main.cu 

Uni:
	$(NVCC) -o Uni ./Unified/Jacobi_GPU.cu 
Memc:
	$(NVCC) -o Mem ./Memcopy/Jacobi_GPU_Malloc.cu 
Shared: ./SharedMem/Jacobi_GPU_Malloc.cu ./SharedMem/util.h ./SharedMem/J_kernel.h
	$(NVCC) -o Shared ./SharedMem/Jacobi_GPU_Malloc.cu 
