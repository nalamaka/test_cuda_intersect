#include <iostream>
#include "graph.h"
#include "io.h"
#include<string.h>
int my_binary_search(int len, int val, int *beg)
{
	int l = 0, r = len;
	while (l < r - 1)
	{
		int mid = (l + r) / 2;
		if (beg[mid + 1] - beg[mid] > val)
			l = mid;
		else
			r = mid;
	}
	if (beg[l + 1] - beg[l] <= val)
		return -1;
	return l;
}

void gen_test(char *file_name,int *a,int * count_a,int *b,int *count_b){
	std::string path = std::string(file_name);
	int select_a = my_binary_search(hg.vert_count, *count_a, hg.beg_pos);
	if(select_a == -1){
		printf("not_enough\n");
		exit(1);
	}
	int select_b = hg.adj_list[select_a + *count_a/2];
	memcpy(a,hg.adj_list + hg.beg_pos[select_a],hg.beg_pos[select_a+1] - hg.beg_pos[select_a]);
	memcpy(b,hg.adj_list + hg.beg_pos[select_b],hg.beg_pos[select_b+1] - hg.beg_pos[select_b]);
	count_a[0] = hg.beg_pos[select_a+1] - hg.beg_pos[select_a];
	count_b[0] = hg.beg_pos[select_b+1] - hg.beg_pos[select_b];
	// delete hg;
}