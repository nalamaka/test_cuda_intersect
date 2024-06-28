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

typedef struct{
    double time;
    int ans;
}container;

int a[MAX_COUNT];
int b[MAX_COUNT];
int gold_ans[SINGLE_SIZE];
container cpu_container[CONTAINER_SIZE];

void generate_test(int *a,int count){
    for(int i=0;i<count;i++){
        a[i] = i;
    }
}

void gen_gold_once(int *a,int size_a,int *b,int size_b,int *ans,int pos){
    int temp = 0;
    for(int i=0;i<size_a;i++){
        for(int j=0;j<size_b;j++){
            if(a[i] == b[j]){
                temp = temp + 1;
                break;
            }
        }
    }
    ans[pos] = temp;
}

void generate_gold_ans(int *a,int size_a,int *b,int size_b,int gold_step,int *gold_container){
    // int end = size_a / gold_step;
    for(int curr_pos = 0;curr_pos < size_a;curr_pos += gold_step){
        if(curr_pos > size_b){
            gen_gold_once(a,curr_pos,b,size_b,gold_container,curr_pos/gold_step);
        }else{
            gen_gold_once(a,curr_pos,b,curr_pos,gold_container,curr_pos/gold_step);
        }
    }
}
__global__ void test_bs(int *a,int size_a,int *b,int size_b,int *ans_pos){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ int G_counter;
    int P_counter;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
        G_counter = 0;
	}
    __syncthreads();
#ifndef TEST_ROUNDS
    P_counter = intersect_bs_cache(a,size_a,b,size_b);
#else
    for(int i=0;i < TEST_ROUNDS;i++){
        P_counter = intersect_bs_cache(a,size_a,b,size_b);
    }
#endif
    // printf("Pcounter:%d\n",P_counter);
    atomicMax(&G_counter,P_counter);
    __syncwarp();
    if (threadIdx.x  == 0)
	{
		atomicAdd(&ans_pos[0], G_counter);
	}
}

__global__ void test_merge(int *a,int size_a,int *b,int size_b,int *ans_pos){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ int G_counter;
    int P_counter;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
        G_counter = 0;
	}
    __syncthreads();
#ifndef TEST_ROUNDS
    P_counter = intersect_num_merge(a,size_a,b,size_b);
    // printf("Pcounter:%d\n",P_counter);
#else
    for(int i=0;i < TEST_ROUNDS;i++){
        P_counter = intersect_num_merge(a,size_a,b,size_b);
    }
#endif
    atomicAdd(&G_counter,P_counter);
    __syncwarp();
    if (threadIdx.x == 0)
	{
		atomicAdd(&ans_pos[0], G_counter);
	}
}

__global__ void test_linear(int *a,int size_a,int *b,int size_b,int *ans_pos,int *partition){
    // printf("a1:%d,b1:%d\n",a[1],b[1]);
    // printf("a10:%d,b10:%d\n",a[10],b[10]);
    // printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ int G_counter;
    int P_counter;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
        G_counter = 0;
	}
    __syncthreads();
    __shared__ int bin_count[HASH_MAX];
	__shared__ int shared_partition[HASH_MAX * shared_BUCKET_SIZE + 1];
#ifndef TEST_ROUNDS
    P_counter = single_search_warp_static(shared_partition,partition,bin_count,a,size_a,b,size_b);
#else
    // __shared__ int bin_count[HASH_MAX];
	// __shared__ int shared_partition[HASH_MAX * shared_BUCKET_SIZE + 1];
	// gen_bin(a, size_a, shared_partition,partition,bin_count);
	// __syncwarp();
    // for(int i=0;i < TEST_ROUNDS;i++){
    //     P_counter = single_search_static(shared_partition,partition,bin_count,b,size_b);
    // }
#endif
    atomicAdd(&G_counter,P_counter);
    __syncwarp();
    if (threadIdx.x == 0)
	{
		atomicAdd(&ans_pos[0], G_counter);
	}
}

void test(int count_a,int count_b){
    
    int *a_device;
    int *b_device;
    int *ans_pos;
    int cpu_ans;

    HRR(cudaMalloc((void **)&a_device,sizeof(int)*MAX_COUNT));
    HRR(cudaMalloc((void **)&b_device,sizeof(int)*MAX_COUNT));
    HRR(cudaMalloc((void **)&ans_pos,sizeof(int)*MAX_COUNT));

    HRR(cudaMemcpy(a_device, a, sizeof(int) * MAX_COUNT, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(b_device, b, sizeof(int) * MAX_COUNT, cudaMemcpyHostToDevice));

    for(int test_size = 0; test_size < count_a && test_size < count_b;test_size += STEP){
        double time_start = clock();
        test_bs<<<1, BLOCK_SIZE>>>(a_device, test_size, b_device, test_size,ans_pos);
        HRR(cudaDeviceSynchronize());
        double cmp_time = clock() - time_start;
        double cmptime = cmp_time / CLOCKS_PER_SEC;
        cpu_container[test_size/STEP].time = cmptime;
        HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(int), cudaMemcpyDeviceToHost));
        cpu_container[test_size/STEP].ans = cpu_ans;

    }

    for(int test_size = 0; test_size < count_a && test_size < count_b;test_size += STEP){
        double time_start = clock();
        test_merge<<<1, BLOCK_SIZE>>>(a_device, test_size, b_device, test_size,ans_pos);
        HRR(cudaDeviceSynchronize());
        double cmp_time = clock() - time_start;
        double cmptime = cmp_time / CLOCKS_PER_SEC;
        cpu_container[SINGLE_SIZE + test_size/STEP].time = cmptime;
        HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(int), cudaMemcpyDeviceToHost));
        cpu_container[SINGLE_SIZE+test_size/STEP].ans = cpu_ans;
    }

    int *partition_gpu;
    HRR(cudaMalloc((void **)&partition_gpu,sizeof(int)*1024*HASH_MAX));

    for(int test_size = 0; test_size < count_a && test_size < count_b;test_size += STEP){
        double time_start = clock();
        test_linear<<<1, BLOCK_SIZE>>>(a_device, test_size, b_device, test_size,ans_pos,partition_gpu);
        HRR(cudaDeviceSynchronize());
        double cmp_time = clock() - time_start;
        double cmptime = cmp_time / CLOCKS_PER_SEC;
        cpu_container[SINGLE_SIZE * 2 + test_size/STEP].time = cmptime;
        HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(int), cudaMemcpyDeviceToHost));
        cpu_container[SINGLE_SIZE * 2+test_size/STEP].ans = cpu_ans;
    }
    
    HRR(cudaFree(a_device));
    HRR(cudaFree(b_device));
}

void write_results_to_file(char *filename){
    FILE *file = fopen(filename, "w"); // 打开文件以便写入
    if(file == NULL){
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "gold_ans\tsize\tbs\tTime(s)\tCorrectness\tmerge\tTime(s)\tCorrectness\tlinear\tTime(s)\tCorrectness\n"); // 写入表头
    for(int i = 0; i < SINGLE_SIZE ; i++){
        fprintf(file, "%10d\t%10d\t%10f\t%1d\t\t%10f\t%1d\t\t%10f\t%1d\n", 
        gold_ans[i],i*STEP,cpu_container[i].time, cpu_container[i].ans,cpu_container[SINGLE_SIZE+i].time,cpu_container[SINGLE_SIZE+i].ans,cpu_container[2*SINGLE_SIZE+i].time,cpu_container[2*SINGLE_SIZE+i].ans); // 将每个元素写入文件
    }
    fclose(file); // 关闭文件
}

int main(int argc,char ** argv ){
    // generate_test(a,MAX_COUNT);
    // generate_test(b,MAX_COUNT);
    int count_a = 100;
    int count_b;
    gen_test(argv[1],a,&count_a,b,&count_b);
    // generate_test(a,count_a);
    // generate_test(b,count_b);
    generate_gold_ans(a,count_a,b,count_b,STEP,gold_ans);
    
    test(count_a,count_b);

    write_results_to_file(argv[2]);
}