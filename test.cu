#include<iostream>
#include"set_intersect.cuh"
#include"error.cuh"
#include"time.h"

#define MAX_COUNT 100000
#define STEP 1000
#define TEST_NUM 3
#define SINGLE_SIZE (MAX_COUNT/STEP)
#define CONTAINER_SIZE (SINGLE_SIZE * TEST_NUM)

typedef struct{
    double time;
    int TorF;
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
    ans[pos] = pos * STEP;
}

void generate_gold_ans(int *a,int size_a,int *b,int size_b,int gold_step,int *gold_container){
    int end = size_a / gold_step;
    for(int curr_pos = 0;curr_pos < size_a;curr_pos += gold_step){
        gen_gold_once(a,curr_pos,b,curr_pos,gold_container,curr_pos/gold_step);
    }
}

void write_results_to_file(int *results, int size, const char *filename){
    FILE *file = fopen(filename, "w"); // 打开文件以便写入
    if(file == NULL){
        printf("Error opening file!\n");
        return;
    }
    for(int i = 0; i < size; i++){
        fprintf(file, "%d\n", results[i]); // 将每个元素写入文件
    }
    fclose(file); // 关闭文件
}

__global__ void test_bs(int *a,int size_a,int *b,int size_b,int *ans_pos){
    printf("a1:%d,b1:%d\n",a[1],b[1]);
    printf("a10:%d,b10:%d\n",a[10],b[10]);
    printf("sizea:%d,sizeb:%d\n",size_a,size_b);
    __shared__ int G_counter;
    if (threadIdx.x == 0)
	{
		ans_pos[0] = 0;
        G_counter = 0;
	}
    __syncwarp();
    int P_counter = intersect_bs_cache(a,size_a,b,size_b);
    printf("Pcounter:%d\n",P_counter);
    atomicAdd(&G_counter,P_counter);
    __syncwarp();
    if (threadIdx.x == 0)
	{
		atomicAdd(&ans_pos[0], G_counter);
	}
}

void test(){
    
    int *a_device;
    int *b_device;
    int *ans_pos;
    int cpu_ans;

    HRR(cudaMalloc((void **)&a_device,sizeof(int)*MAX_COUNT));
    HRR(cudaMalloc((void **)&b_device,sizeof(int)*MAX_COUNT));
    HRR(cudaMalloc((void **)&ans_pos,sizeof(int)*MAX_COUNT));

    HRR(cudaMemcpy(a_device, a, sizeof(int) * MAX_COUNT, cudaMemcpyHostToDevice));
    HRR(cudaMemcpy(b_device, b, sizeof(int) * MAX_COUNT, cudaMemcpyHostToDevice));

    for(int test_size = 0; test_size < STEP * 2;test_size += STEP){
        double time_start = clock();
        test_bs<<<1, 32>>>(a_device, test_size, b_device, test_size,ans_pos);
        HRR(cudaDeviceSynchronize());
        double cmp_time = clock() - time_start;
        double cmptime = cmp_time / CLOCKS_PER_SEC;
        cpu_container[test_size/STEP].time = cmptime;
        HRR(cudaMemcpy(&cpu_ans, ans_pos , sizeof(int), cudaMemcpyDeviceToHost));
        if(cpu_ans == gold_ans[test_size/STEP]){
            cpu_container[test_size/STEP].TorF = 1;
        }else{
            printf("wrong ans is %d\n",cpu_ans);
            cpu_container[test_size/STEP].TorF = 0;
        }

    }
    
    HRR(cudaFree(a_device));
    HRR(cudaFree(b_device));
}

void write_results_to_file(){
    FILE *file = fopen("./out.txt", "w"); // 打开文件以便写入
    if(file == NULL){
        printf("Error opening file!\n");
        return;
    }
    fprintf(file, "Time(s),Correctness\n"); // 写入表头
    for(int i = 0; i < CONTAINER_SIZE ; i++){
        fprintf(file, "%f,%d\n", cpu_container[i].time, cpu_container[i].TorF); // 将每个元素写入文件
    }
    fclose(file); // 关闭文件
}

int main(){
    generate_test(a,MAX_COUNT);
    generate_test(b,MAX_COUNT);
    generate_gold_ans(a,MAX_COUNT,b,MAX_COUNT,STEP,gold_ans);
    
    test();

    write_results_to_file();
}