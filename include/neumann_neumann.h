#include "nnedge.h"
#include "qgedge.h"
#include <vector>
#include <unordered_set>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#pragma once

class NeumannNeumann {
public:
	int vertices;
	int max_iter;
	double theta;
	std::vector<NNEdge*> dirichlet_edges;
	std::vector<NNEdge*> neumann_edges;
	std::vector<std::unordered_set<int>> vertex_to_edges;
	std::vector<int> vertex_weights;
	std::unordered_set<int> boundary;
	Eigen::VectorXd vertex_values;

	NeumannNeumann(int N, int _vertices, std::vector<QGEdge>& edges, int _max_iter, double _theta);
	Eigen::VectorXd solve();
	~NeumannNeumann();
};
