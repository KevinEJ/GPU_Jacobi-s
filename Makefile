
cpu:
	g++ -o J_CPU Jacobi_CPU.cpp -O3 -g

NVCC = nvcc -std=c++11 -lineinfo -O0 -g
gpu:
	$(NVCC) -o J_GPU Jacobi_GPU.cu 
