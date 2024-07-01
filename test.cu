#include<iostream>
#include"set_intersect.cuh"
#include"error.cuh"
#include"time.h"
#include"linear_search.cuh"
#include"gen_test.cuh"

#define MAX_COUNT 100000
#define STEP 10
// #define STEP 10
#define TEST_NUM 3
#define SINGLE_SIZE (MAX_COUNT/STEP)
#define CONTAINER_SIZE (SINGLE_SIZE * TEST_NUM)
// #define TEST_ROUNDS 20
#define TEST_MIN_DEGREE 2
#define TEST_MAX_DEGREE 30

typedef struct{
    double time;
    int ans;
}container;

container cpu_container[CONTAINER_SIZE];

__global__ void test_bs(int *beg_pos_device,int * adj_list_device,int num_vertex,unsigned long long *ans_pos){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ unsigned long long G_counter[BLOCK_SIZE/WARPSIZE];
    unsigned long long P_counter = 0;
    int warp_id = (threadIdx.x >> 5) & 31;
    int In_warp_id = threadIdx.x & 31;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
	}
    if(In_warp_id == 0){
        G_counter[warp_id] = 0;
    }
    __syncthreads();
    int now = warp_id;
    while(now < num_vertex){
        int vertex_start = beg_pos_device[now];
        int vertex_degree = beg_pos_device[now + 1] - vertex_start;
        if(vertex_degree >= TEST_MAX_DEGREE || vertex_degree <= TEST_MIN_DEGREE){
            now += BLOCK_SIZE/WARPSIZE;
            continue;
        }
        int warp_iret = vertex_start;
        while(warp_iret < vertex_degree){
            int to_search = adj_list_device[warp_iret];
            int to_search_start = beg_pos_device[to_search];
            int to_search_degree = beg_pos_device[to_search+1] - to_search_start;
            P_counter += intersect_bs(adj_list_device + vertex_start,vertex_degree,adj_list_device + to_search_start, to_search_degree);
            warp_iret += 1;
        }
#ifndef __DYNAMIC
        now += BLOCK_SIZE/WARPSIZE;
#endif
    }
    if(In_warp_id == 31){
        atomicAdd(&G_counter[warp_id],P_counter);
    }
    __syncthreads();
    if (In_warp_id == 0)
	{
		atomicAdd(&ans_pos[0], G_counter[warp_id]);
	}
}

__global__ void test_merge(int *beg_pos_device,int * adj_list_device,int num_vertex,unsigned long long *ans_pos){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ int G_counter[BLOCK_SIZE/WARPSIZE];
    unsigned long long P_counter = 0;
    int warp_id = (threadIdx.x >> 5) & 31;
    int In_warp_id = threadIdx.x & 31;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
	}
    if(In_warp_id == 0){
        G_counter[warp_id] = 0;
    }
    __syncthreads();
    int now = warp_id;
    while(now < num_vertex){
        int vertex_start = beg_pos_device[now];
        int vertex_degree = beg_pos_device[now + 1] - vertex_start;
        if(vertex_degree >= TEST_MAX_DEGREE || vertex_degree <= TEST_MIN_DEGREE){
            now += BLOCK_SIZE/WARPSIZE;
            continue;
        }
        int warp_iret = vertex_start;
        while(warp_iret < vertex_degree){
            int to_search = adj_list_device[warp_iret];
            int to_search_start = beg_pos_device[to_search];
            int to_search_degree = beg_pos_device[to_search+1] - to_search_start;
            P_counter += intersect_num_merge(adj_list_device + vertex_start,vertex_degree,adj_list_device + to_search_start, to_search_degree);
            warp_iret += 1;
        }
#ifndef __DYNAMIC
        now += BLOCK_SIZE/WARPSIZE;
#endif
    }
    atomicAdd(&G_counter[warp_id],P_counter);
    __syncthreads();
    if (In_warp_id == 0)
	{
		atomicAdd(&ans_pos[0], G_counter[warp_id]);
	}
}

__global__ void test_linear(int *beg_pos_device,int * adj_list_device,int num_vertex,unsigned long long *ans_pos,int *partition){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ unsigned long long G_counter[BLOCK_SIZE/WARPSIZE];
    unsigned long long P_counter = 0;
    int warp_id = (threadIdx.x >> 5) & 31;
    int In_warp_id = threadIdx.x & 31;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
	}
    if(In_warp_id == 0){
        G_counter[warp_id] = 0;
    }
    __syncthreads();
    __shared__ int bin_count[HASH_MAX];
	__shared__ int shared_partition[HASH_MAX * shared_BUCKET_SIZE + 1];//shared hash bin
    int now = warp_id;
    while(now < num_vertex){
        int vertex_start = beg_pos_device[now];
        int vertex_degree = beg_pos_device[now + 1] - vertex_start;
        if(vertex_degree >= TEST_MAX_DEGREE || vertex_degree <= TEST_MIN_DEGREE){
            now += BLOCK_SIZE/WARPSIZE;
            continue;
        }
        int warp_iret = vertex_start + warp_id;
        while(warp_iret < vertex_degree){
            int to_search = adj_list_device[warp_iret];
            int to_search_start = beg_pos_device[to_search];
            int to_search_degree = beg_pos_device[to_search+1] - to_search_start;
            P_counter += single_search_warp_static(shared_partition,partition,bin_count,adj_list_device + vertex_start,vertex_degree,adj_list_device + to_search_start, to_search_degree);
            warp_iret += 1;
        }
#ifndef __DYNAMIC
        now += BLOCK_SIZE/WARPSIZE;
#endif
    }
    atomicAdd(&G_counter[warp_id],P_counter);
    __syncthreads();
    if (In_warp_id == 0)
	{
		atomicAdd(&ans_pos[0], G_counter[warp_id]);
	}
}

void test(int *beg_pos,int *adj_list,int num_vertex,int num_edge){
    
    int *beg_pos_device;
    int *adj_list_device;
    unsigned long long *ans_pos;
    unsigned long long  cpu_ans;

    HRR(cudaMalloc((void **)&beg_pos_device,sizeof(int)*(num_vertex + 1)));
    HRR(cudaMalloc((void **)&adj_list_device,sizeof(int)*num_edge));
    HRR(cudaMalloc((void **)&ans_pos,sizeof(unsigned long long )));

    HRR(cudaMemcpy(beg_pos_device, beg_pos, sizeof(int)*(num_vertex + 1), cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(adj_list_device, adj_list, sizeof(int)*num_edge, cudaMemcpyHostToDevice));

    //bs test
    double time_start = clock();
    test_bs<<<1, BLOCK_SIZE>>>(beg_pos_device, adj_list_device, num_vertex,ans_pos);
    HRR(cudaDeviceSynchronize());
    double cmp_time = clock() - time_start;
    double cmptime = cmp_time / CLOCKS_PER_SEC;
    cpu_container[0].time = cmptime;
    HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(unsigned long long ), cudaMemcpyDeviceToHost));
    cpu_container[0].ans = cpu_ans;

    //merge test
    time_start = clock();
    test_merge<<<1, BLOCK_SIZE>>>(beg_pos_device, adj_list_device, num_vertex,ans_pos);
    HRR(cudaDeviceSynchronize());
    cmp_time = clock() - time_start;
    cmptime = cmp_time / CLOCKS_PER_SEC;
    cpu_container[SINGLE_SIZE].time = cmptime;
    HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(unsigned long long ), cudaMemcpyDeviceToHost));
    cpu_container[SINGLE_SIZE].ans = cpu_ans;

    int *partition_gpu;
    HRR(cudaMalloc((void **)&partition_gpu,sizeof(int)*1024*HASH_MAX));

    time_start = clock();
    dynamic_assign<<<1, BLOCK_SIZE>>>(adj_list_device, beg_pos_device,num_edge,num_vertex,partition_gpu, ans_pos);
    HRR(cudaDeviceSynchronize());
    cmp_time = clock() - time_start;
    cmptime = cmp_time / CLOCKS_PER_SEC;
    cpu_container[SINGLE_SIZE * 2].time = cmptime;
    HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(unsigned long long ), cudaMemcpyDeviceToHost));
    cpu_container[SINGLE_SIZE * 2].ans = cpu_ans;
    
    HRR(cudaFree(beg_pos_device));
    HRR(cudaFree(adj_list_device));
    HRR(cudaFree(ans_pos));
    HRR(cudaFree(partition_gpu));
}

void write_results_to_file(char *filename){
    FILE *file = fopen(filename, "w"); // 打开文件以便写入
    if(file == NULL){
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "gold_ans\tsize\tbs\tTime(s)\tCorrectness\tmerge\tTime(s)\tCorrectness\tlinear\tTime(s)\tCorrectness\n"); // 写入表头
    for(int i = 0; i < SINGLE_SIZE ; i++){
        fprintf(file, "%10d\t%10f\t%1d\t\t%10f\t%1d\t\t%10f\t%1d\n", 
        i*STEP,cpu_container[i].time, cpu_container[i].ans,cpu_container[SINGLE_SIZE+i].time,cpu_container[SINGLE_SIZE+i].ans,cpu_container[2*SINGLE_SIZE+i].time,cpu_container[2*SINGLE_SIZE+i].ans); // 将每个元素写入文件
    }
    fclose(file); // 关闭文件
}

int main(int argc,char ** argv ){
    // generate_test(a,MAX_COUNT);
    // generate_test(b,MAX_COUNT);
    int *beg_pos;//list of the start of the vertex
    int *adj_list;//list of the end of the edge
    int num_edge;
    int num_vertex;
    gen_test(argv[1],&beg_pos,&adj_list,&num_vertex,&num_edge);
    
    test(beg_pos,adj_list,num_vertex,num_edge);
    write_results_to_file(argv[2]);
}