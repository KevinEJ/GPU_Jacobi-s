
#NVCC = nvcc -std=c++11 -lineinfo -O3 -g
NVCC = nvcc -std=c++11 -O3 

all: ./Organized/*
	$(NVCC) -o HW1 ./Organized/main.cu 

