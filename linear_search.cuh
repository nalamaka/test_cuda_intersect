#pragma once
#include "search.cuh"
#define shared_BUCKET_SIZE 6
#define shared_BUCKET_SIZE 6
#define SUM_SIZE 1
#define USE_WARP 2
#define without_combination 0
#define use_static 1

#define HASE_BIT_SIZE 10
#define HASH_MAX (1 << HASE_BIT_SIZE)
#define THREAD_MODULO (HASH_MAX - 1)
#define WARPSIZE 32
#define WARP_BUCKETNUM (HASH_MAX/WARPSIZE)
#define WARP_MODULO (WARP_BUCKETNUM - 1)

template <typename T = vidType>
__forceinline__ __device__ void gen_block_shared_bin(T* a, T size_a, T* bin_loc,T*partition_loc,T* bin_count){
	for (int i = threadIdx.x; i < HASH_MAX; i += blockDim.x){
		bin_count[i] = 0;
	}
	// show_bin(bin_loc,partition_loc,bin_count);
	__syncthreads();
	int end = size_a;
	int now = threadIdx.x;
	while (now < end)
	{
		int temp = a[now];
		int bin = temp & THREAD_MODULO;
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
		now += blockDim.x;
	}
	__syncthreads();
}

template <typename T = vidType>
__forceinline__ __device__ void gen_warp_shared_bin(T* a, T size_a, T* bin_loc,T*partition_loc,T* bin_count,int bin_offset){
	int WARP_TID = threadIdx.x & WARP_MODULO;
	for (int i = bin_offset + WARP_TID; i < bin_offset + WARP_BUCKETNUM; i += WARPSIZE){
		bin_count[i] = 0;
	}
	__syncwarp();
	int now = threadIdx.x % WARPSIZE;
	int end = size_a;
	// count hash bin
	while (now < end)
	{
		T temp = a[now];
		int bin = temp & WARP_MODULO;
		bin += bin_offset;
		int index;
		index = atomicAdd(&bin_count[bin], 1);
		if (index < shared_BUCKET_SIZE)
		{
			bin_loc[index * HASH_MAX + bin] = temp;
		}
		else if (index < BUCKET_SIZE)
		{
			index = index - shared_BUCKET_SIZE;
			partition_loc[index * HASH_MAX + bin] = temp;
		}
		now += WARPSIZE;
	}
	__syncwarp();
}

template <typename T = vidType>
__forceinline__ __device__ T linear_search(T*bin_loc,T*partition_loc,T*bin_count,T bin,T tosearch){
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
__forceinline__ __device__ T single_search_block_static(T*bin_loc,T*partition_loc,T*bin_count,T*a,T size_a,T*b,T size_b){
	int temp_ans = 0;
	gen_block_shared_bin(a,size_a,bin_loc,partition_loc,bin_count);
	for(int i = threadIdx.x; i < size_b;i+=blockDim.x){
		temp_ans += linear_search(bin_loc,partition_loc,bin_count,b[i] & THREAD_MODULO,b[i]);
	}
	return temp_ans;
}

template <typename T = vidType>
__forceinline__ __device__ T single_search_warp_static(T*bin_loc,T*partition_loc,T*bin_count,T*a,T size_a,T*b,T size_b){
	int temp_ans = 0;
	int bin_offset = (threadIdx.x / WARPSIZE) * WARP_BUCKETNUM;
	gen_warp_shared_bin(a,size_a,bin_loc,partition_loc,bin_count,bin_offset);
	for(int i = threadIdx.x; i < size_b;i+=WARPSIZE){
		temp_ans += linear_search(bin_loc,partition_loc,bin_count,(b[i] & WARP_MODULO) + bin_offset,b[i]);
	}
	return temp_ans;
}