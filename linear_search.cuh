#pragma once
#include "search.cuh"
#define HASH_MAX 1024
#define MODULO (HASH_MAX - 1)
#define WARPSIZE 32
#define shared_BUCKET_SIZE 6
template <typename T = vidType>
__forceinline__ __device__ void gen_bin(T* a, T size_a, T* bin_loc,T*partition_loc,T* bin_count){
	int WARP_TID = threadIdx.x % WARPSIZE;
	for (int i = WARP_TID; i < HASH_MAX; i += 32){
		bin_count[i] = 0;
	}
	// show_bin(bin_loc,partition_loc,bin_count);
	__syncwarp();
	int start = 0;
	int end = size_a;
	int now = threadIdx.x + start;
	while (now < end)
	{
		int temp = a[now];
		int bin = temp & MODULO;
		int index;
		index = atomicAdd(&bin_count[bin], 1);
		if (index < 6)
		{
			bin_loc[index * HASH_MAX + bin] = temp;
		}
		else if (index < HASH_MAX)
		{
			index = index - 6;
			partition_loc[index * HASH_MAX + bin] = temp;
		}else{
			printf("Error: index out of range\n");
		}
		now += WARPSIZE;
	}
}

template <typename T = vidType>
__forceinline__ __device__ T linear_search(T*bin_loc,T*partition_loc,T*bin_count,T tosearch){
	int bin = tosearch & MODULO;
	int len = bin_count[bin];
	int i = bin;
	int step = 0;
	int nowlen;
	if (len < shared_BUCKET_SIZE)//maximum len to search in the shared bin
		nowlen = len;
	else
		nowlen = shared_BUCKET_SIZE;
	while (step < nowlen)
	{
		if (bin_loc[i] == tosearch)
		{
			return 1;
		}
		i += HASH_MAX;
		step += 1;
	}

	len -= shared_BUCKET_SIZE;
	i = bin;
	step = 0;
	while (step < len)
	{
		if (partition_loc[i] == tosearch)
		{
			return 1;
		}
		i += HASH_MAX;
		step += 1;
	}
	return 0;
}

template <typename T = vidType>
__forceinline__ __device__ T single_search_static(T*bin_loc,T*partition_loc,T*bin_count,T*b,T size_b){
	int temp_ans = 0;
	for(int i = threadIdx.x; i < size_b;i+=blockDim.x){
		temp_ans += linear_search(bin_loc,partition_loc,bin_count,b[i]);
	}
	return temp_ans;
}

template <typename T = vidType>
__forceinline__ __device__ T single_search_dynamic(T*bin_loc,T*partition_loc,T*bin_count,T*b,T size_b,int *shared_iret){
	int temp_ans = 0;
	int iret = atomicAdd(shared_iret,1);
	for(; iret < size_b;iret = atomicAdd(shared_iret,1)){
		temp_ans += linear_search(bin_loc,partition_loc,bin_count,b[iret]);
	}
	return temp_ans;
}

template <typename T = vidType>
__forceinline__ __device__ T intersect_hash(T* a, T size_a, T* b, T size_b,T *partition){
	__shared__ int bin_count[HASH_MAX];
	__shared__ int shared_partition[HASH_MAX * shared_BUCKET_SIZE + 1];
	gen_bin(a, size_a, shared_partition,partition,bin_count);
	__syncwarp();
	return single_search_static(shared_partition,partition,bin_count,b,size_b);
}