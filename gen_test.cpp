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

void gen_test(char *file_name,int **vertex_start,int **end_list,int *vertex_count,int *edge_count){
	std::string path = std::string(file_name);
	graph hg;
	hg.read_graph(path);
	*vertex_start = hg.vertex_start;
	*end_list = hg.end_list;
	*vertex_count = hg.num_vertex;
	*edge_count = hg.num_edges;
	// delete hg;
}