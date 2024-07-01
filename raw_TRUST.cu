#include<iostream>
#define shared_BUCKET_SIZE 6
#define SUM_SIZE 1
#define USE_CTA 100
#define USE_WARP 2
#define without_combination 1
#define MIN_DEGREE 2
#define MAX_DEGREE 30
#define HASE_BIT_SIZE 10
#define HASH_MAX (1 << HASE_BIT_SIZE)
#define warp_bucketnum 32
__device__ int linear_search(int neighbor, int *shared_partition, int *partition, int *bin_count, int bin, int BIN_START){

	int i = bin;
	int len = bin_count[i];
	// unsigned int guess = guess_bin[bin];
	// unsigned int neighbor_guess = 1 << ((neighbor >> HASE_BIT_SIZE) & (31));
	// if((guess & neighbor_guess) == 0){
	// 	return 0;
	// }
	int step = 0;
	int nowlen;
	if (len < shared_BUCKET_SIZE)//maximum len to search in the shared bin
		nowlen = len;
	else
		nowlen = shared_BUCKET_SIZE;
	while (step < nowlen)
	{
		if (shared_partition[i] == neighbor)
		{
			return 1;
		}
		i += HASH_MAX;
		step += 1;
	}

	len -= shared_BUCKET_SIZE;
	i = bin + BIN_START;
	step = 0;
	while (step < len)
	{
		if (partition[i] == neighbor)
		{
			return 1;
		}
		i += HASH_MAX;
		step += 1;
	}
	return 0;
	//if a vertex cannot find in the hash bin it may cost too much time to find the fact
	//also it not idicate a node that is not valid
}
__global__ void dynamic_assign(int *adj_list, int *beg_pos, int edge_count, int vertex_count, int *partition, unsigned long long *GLOBAL_COUNT)
{

	// int tid=threadIdx.x+blockIdx.x*blockDim.x;
	__shared__ int bin_count[HASH_MAX];
	__shared__ int shared_partition[HASH_MAX * shared_BUCKET_SIZE + 1];
	
	// __shared__ int shared_now,shared_workid;
	// __shared__ int useless[1024*9];
	// useless[threadIdx.x]=1;
	unsigned long long __shared__ G_counter;
	int WARPSIZE = 32;
	if (threadIdx.x == 0)
	{
		G_counter = 0;
	}
	// timetest
	unsigned long long TT = 0, HT = 0, IT = 0;
	unsigned long long __shared__ G_TT, G_HT, G_IT;
	G_TT = 0, G_HT = 0, G_IT = 0;

	int BIN_START = blockIdx.x * HASH_MAX * 100;
	// __syncthreads();
	unsigned long long P_counter = 0;

	// unsigned long long start_time;

	// start_time = clock64();
	// CTA for large degree vertex
	int vertex = (blockIdx.x * 1 + 0) * 1;//the first vertex to search
	int vertex_end = vertex + 1;
	__shared__ int ver;
	while (vertex < 0)
	{
		int degree = beg_pos[vertex + 1] - beg_pos[vertex];//the degree of a node to search
		int start = beg_pos[vertex];
		int end = beg_pos[vertex + 1];
		int now = threadIdx.x + start;//divide the task to the whole warp
		int MODULO = HASH_MAX - 1;
		int BIN_OFFSET = 0;
		// clean bin_count
		for (int i = threadIdx.x; i < HASH_MAX; i += blockDim.x){
			bin_count[i] = 0;
			// bin_guess[i] = 0;
		}
		__syncthreads();

		// count hash bin
		while (now < end)
		{
			int temp = adj_list[now];
			int bin = temp & MODULO;//hash the temps
			int index;
			// index = atomicAdd(&bin_count[bin], 1);//bin++
			// index = ++bin_count[bin];
			// atomicOr(&bin_guess[bin],or_bits);
			// atomicAdd(&bin_guess[bin],1);
			if (index < shared_BUCKET_SIZE)//index can fit in the shared bucket
			{
				shared_partition[index * HASH_MAX + bin] = temp;
			}
			else if (index < 100)
			{
				index = index - shared_BUCKET_SIZE;
				partition[index * HASH_MAX + bin + BIN_START] = temp;
			}
			now += blockDim.x;
		}
		__syncthreads();
		
		now = threadIdx.x + start;//divide the task to the whole warp
		// while(now < end){
		// 	int temp = adj_list[now];
		// 	int bin = temp & MODULO;
		// 	unsigned int or_bits = 1 << ((temp >> HASE_BIT_SIZE) & (31));
		// 	int ans = atomicOr(&bin_guess[bin],or_bits);
		// 	now += blockDim.x;
		// }

		// unsigned long long hash_time=clock64()-start_time;
		// start_time = clock64();
		// list intersection
		now = beg_pos[vertex];
		end = beg_pos[vertex + 1];
		if (without_combination)
		{
			while (now < end)
			{
				int neighbor = adj_list[now];
				int neighbor_start = beg_pos[neighbor];
				int neighbor_end = beg_pos[neighbor + 1];
				int neighbor_now = neighbor_start + threadIdx.x;
				while (neighbor_now < neighbor_end)
				{
					int temp = adj_list[neighbor_now];
					int bin = temp & MODULO;
					P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
					neighbor_now += blockDim.x;
				}
				now++;
			}
		}
		else
		{
			int superwarp_ID = threadIdx.x / 64;
			int superwarp_TID = threadIdx.x % 64;
			int workid = superwarp_TID;
			now = now + superwarp_ID;
			int neighbor = adj_list[now];
			int neighbor_start = beg_pos[neighbor];
			int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
			while (now < end)
			{
				while (now < end && workid >= neighbor_degree)
				{
					now += 16;
					workid -= neighbor_degree;
					neighbor = adj_list[now];
					neighbor_start = beg_pos[neighbor];
					neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
				}
				if (now < end)
				{
					int temp = adj_list[neighbor_start + workid];
					int bin = temp & MODULO;
					P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
				}
				workid += 64;
			}
		}
		__syncthreads();
		vertex += gridDim.x * 1;
	}
	// warp method
	int WARPID = threadIdx.x / WARPSIZE;
	int WARP_TID = threadIdx.x % WARPSIZE;
	int WARPDIM = blockDim.x * gridDim.x / WARPSIZE;
	vertex =WARPID + blockIdx.x * blockDim.x / WARPSIZE;
	vertex_end = vertex + 1;
	while (vertex < vertex_count){
		// unsigned long long start_time = clock64();
		int degree = beg_pos[vertex + 1] - beg_pos[vertex];
		if (degree < MIN_DEGREE || degree > MAX_DEGREE){
			vertex += WARPDIM * 1;
			continue;
		}
		int start = beg_pos[vertex];
		int end = beg_pos[vertex + 1];
		int now = WARP_TID + start;
		int MODULO = warp_bucketnum - 1;
		int BIN_OFFSET = WARPID * warp_bucketnum;

		for (int i = BIN_OFFSET + WARP_TID; i < BIN_OFFSET + warp_bucketnum; i += WARPSIZE){
			bin_count[i] = 0;
		}
		__syncwarp();

		// count hash bin
		while (now < end)
		{
			int temp = adj_list[now];
			int bin = temp & MODULO;
			bin += BIN_OFFSET;
			int index;
			index = atomicAdd(&bin_count[bin], 1);
			// index = ++bin_count[bin];
			// atomicAdd(&bin_guess[bin],1);
			// atomicOr(&bin_guess[bin],or_bits);
			if (index < shared_BUCKET_SIZE)
			{
				shared_partition[index * HASH_MAX + bin] = temp;
			}
			else if (index < 100)
			{
				index = index - shared_BUCKET_SIZE;
				partition[index * HASH_MAX + bin + BIN_START] = temp;
			}
			now += WARPSIZE;
		}
		__syncwarp();
		now = threadIdx.x + start;//divide the task to the whole warp
		// while(now < end){
		// 	int temp = adj_list[now];
		// 	int bin = temp & MODULO;
		// 	unsigned int or_bits = 1 << ((temp >> HASE_BIT_SIZE) & (31));
		// 	int ans = atomicOr(&bin_guess[bin],or_bits);
		// 	now += blockDim.x;
		// }

		// list intersection
		now = beg_pos[vertex];
		end = beg_pos[vertex + 1];

		if (without_combination)
		{
			while (now < end)
			{
				int neighbor = adj_list[now];
				int neighbor_start = beg_pos[neighbor];
				int neighbor_end = beg_pos[neighbor + 1];
				int neighbor_now = neighbor_start + WARP_TID;
				while (neighbor_now < neighbor_end)
				{
					int temp = adj_list[neighbor_now];
					int bin = temp & MODULO;
					P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
					neighbor_now += WARPSIZE;
				}
				now++;
			}
		}
		else
		{
			int workid = WARP_TID;
			while (now < end)
			{
				int neighbor = adj_list[now];
				int neighbor_start = beg_pos[neighbor];
				int neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;

				while (now < end && workid >= neighbor_degree)
				{
					now++;
					workid -= neighbor_degree;
					neighbor = adj_list[now];
					neighbor_start = beg_pos[neighbor];
					neighbor_degree = beg_pos[neighbor + 1] - neighbor_start;
				}
				if (now < end)
				{
					int temp = adj_list[neighbor_start + workid];
					int bin = temp & MODULO;
					P_counter += linear_search(temp, shared_partition, partition, bin_count, bin + BIN_OFFSET, BIN_START);
				}
				__syncwarp();
				now = __shfl_sync(0xffffffff, now, 31);
				workid = __shfl_sync(0xffffffff, workid, 31);

				workid += WARP_TID + 1;

				// workid+=WARPSIZE;
			}
		}
		__syncwarp();
		vertex += WARPDIM;
	}
	atomicAdd(&G_counter, P_counter);

	__syncthreads();
	if (threadIdx.x == 0)
	{
		atomicAdd(&GLOBAL_COUNT[0], G_counter);
	}
}