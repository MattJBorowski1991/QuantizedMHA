#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include "check_cuda.h"

int main(){
	int devCount=0; CHECK_CUDA(cudaGetDeviceCount(&devCount));
	if(devCount<=0){ std::fprintf(stderr, "No CUDA devices found\n"); return 1; }

	std::ofstream ofs("profiles/txt/device_info.txt");
	if (!ofs.is_open()) { std::fprintf(stderr, "Failed to open profiles/txt/device_info.txt for writing\n"); return 1; }

	int dev=0; cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
	ofs << "Device " << dev << ": " << prop.name << "\n";
	ofs << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
	ofs << "SM Count (multiProcessorCount): " << prop.multiProcessorCount << "\n";
	ofs << "Max Threads Per Block: " << prop.maxThreadsPerBlock << "\n";
	ofs << "Max Threads Dim: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
	ofs << "Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
	ofs << "Warp Size: " << prop.warpSize << "\n";
	ofs << "Shared Mem Per Block: " << (size_t)prop.sharedMemPerBlock << " bytes\n";
	ofs << "Shared Mem Per Multiprocessor: " << (size_t)prop.sharedMemPerMultiprocessor << " bytes\n";
	ofs << "Registers Per Block: " << prop.regsPerBlock << "\n";
	ofs << "Regs Per Multiprocessor: " << prop.regsPerMultiprocessor << "\n";
	ofs << "Max Blocks Per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << "\n";
	ofs << "Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
	ofs << "Memory Clock Rate: " << prop.memoryClockRate << " kHz\n";
	ofs << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
	ofs << "Total Global Memory: " << (size_t)prop.totalGlobalMem << " bytes\n";
	ofs << "L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
	ofs << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
	ofs << "Cooperative Launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << "\n";
	ofs << "Max Shared Mem Optin Per Block: " << (size_t)prop.sharedMemPerBlockOptin << " bytes\n";

	ofs.close();
	return 0;
}