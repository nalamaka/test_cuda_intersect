#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

struct graph_vertex
{
	int id;
	int *start;//pos to start
	int degree;
};

struct graph_edge
{
	int start_pos;
	int end_pos;
};


class graph
{
	private:
	// std::vector<graph_vertex> vertices;
	std::vector<graph_edge> edges;
	std::vector<int> adj_list;
	int current_vertex;
	public:
	int *start_list;
	int *end_list;
	int *vertex_start;
	int *vertex_degree;
	int num_vertex;
	int num_edges;
	graph(){
		// vertices.clear();
		edges.clear();
	}
	void read_graph(const std::string filename);
	~graph(){
		// vertices.clear();
		edges.clear();
	}
};

#endif