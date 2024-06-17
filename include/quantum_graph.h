#include "qgedge.h"
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#pragma once

class QuantumGraph {
public:
	const int N;
	const int vertices;
	int vertex_offset;
	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd b;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;	
	Eigen::VectorXd u;

	QuantumGraph(int _N, int _vertices, std::vector<QGEdge>& edges, std::unordered_set<int>& dirichlet_vertices);
	void setDirichletValues(std::vector<int>& indices, Eigen::VectorXd& values);
	void solve();
	Eigen::VectorXd getVertexValues();
};
