#include "bc.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#pragma once

class NNEdge {
public:
	int out;
	int in;
	int N;
	double h;
	Eigen::VectorXd cx;
	Eigen::VectorXd vx;
	Eigen::VectorXd fx;
	BC lbc;
	BC rbc;
	Eigen::Vector2d Al;
	Eigen::Vector2d Ar;
	Eigen::VectorXd b;
	Eigen::VectorXd u;
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

	NNEdge(int _out, int _in, int _N, Eigen::VectorXd& _cx, Eigen::VectorXd& _vx, Eigen::VectorXd& _fx, Eigen::SparseMatrix<double>& _A, BC _lbc, BC _rbc);
	double Dirichlet(int endpoint);
	double approxNeumann(int endpoint);
	void setBC(int endpoint, double value);
	void solve();
};
