#include <iostream>
#include "graph.h"
#include<string.h>
int my_search(int len, int val, int *degree)
{
	int l = 0, r = len;
	while (l < r - 1)
	{
		if(degree[l] >= val)
			return l;
		l++;
	}
	return -1;
}

void gen_test(char *file_name,int *a,int * count_a,int *b,int *count_b){
	std::string path = std::string(file_name);
	graph hg;
	hg.read_graph(path);
	int select_a = my_search(hg.num_vertex, *count_a, hg.vertex_degree);//select a start node
	if(select_a == -1){
		printf("not_enough\n");
		exit(1);
	}
	int select_b = hg.end_list[select_a + *count_a/2];
	memcpy(a,hg.end_list + hg.vertex_start[select_a],hg.vertex_degree[select_a] * sizeof(int));
	memcpy(b,hg.end_list + hg.vertex_start[select_b],hg.vertex_degree[select_b] * sizeof(int));
	count_a[0] = hg.vertex_degree[select_a];
	count_b[0] = hg.vertex_degree[select_b];
	// delete hg;
}