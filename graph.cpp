#include<iostream>
#include"graph.h"
void graph::read_graph(const std::string filename){
	std::ifstream file(filename);
	if(!file.is_open()){
		std::cout<<"Error opening file"<<std::endl;
		return;
	}
	std::string line;
	std::string token;
	
	std::getline(file,line);
	std::istringstream iss(line);
	iss >> this->num_vertex >> this->num_vertex >> this->num_edges;
	while(std::getline(file,line)){
		graph_edge edge;
		std::istringstream iss(line);
		int id;
		iss >> id;
		edge.start_pos = id;
		iss >> edge.end_pos;
		this->edges.push_back(edge);
	}
	file.close();
	this->start_list = new int[this->num_edges];
	this->end_list = new int[this->num_edges];
	this->vertex_start = new int[this->num_vertex + 1];
	this->vertex_degree = new int[this->num_vertex];
	this->vertex_start[0] = 0;
	current_vertex = 0;
	for(int i = 0; i < this->num_edges; i++){
		if(this->edges[i].start_pos != current_vertex){
			this->vertex_degree[current_vertex] = i - this->vertex_start[current_vertex];
			this->vertex_start[current_vertex + 1] = i;
			current_vertex = this->edges[i].start_pos;
		}
		this->start_list[i] = this->edges[i].start_pos;
		this->end_list[i] = this->edges[i].end_pos;
	}
	this->vertex_start[this->num_vertex] = this->num_edges;
	this->edges.clear();
}