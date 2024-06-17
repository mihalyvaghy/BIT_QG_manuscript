#include "bc.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

Eigen::SparseMatrix<double> stiffnessMatrix(Eigen::VectorXd cx, Eigen::VectorXd vx, double h, BC lbc, BC rbc) {
	int N = cx.size();
	size_t nnz = (N-2)*3+2*2-(lbc.type == BCType::Dirichlet)-(rbc.type == BCType::Dirichlet);
	std::vector<Eigen::Triplet<double>> coefficients;
	coefficients.reserve(nnz);

	for (int i=1; i<N-1; ++i) {
		coefficients.push_back({i, i-1, -(cx[i-1]+cx[i])/(2*h)});
		coefficients.push_back({i, i, (cx[i-1]+2*cx[i]+cx[i+1])/(2*h)+h*vx[i]});
		coefficients.push_back({i, i+1, -(cx[i]+cx[i+1])/(2*h)});
	}

	if (lbc.type == BCType::Dirichlet) {
		coefficients.push_back({0,0,1});
	} else {
		coefficients.push_back({0,0,(cx[0]+cx[1])/(2*h)+h/2*vx[0]});
		coefficients.push_back({0,1,-(cx[0]+cx[1])/(2*h)});
	}
	if (rbc.type == BCType::Dirichlet) {
		coefficients.push_back({N-1,N-1,1});
	} else {
		coefficients.push_back({N-1,N-1,(cx[N-2]+cx[N-1])/(2*h)+h/2*vx[N-1]});
		coefficients.push_back({N-1,N-2,-(cx[N-2]+cx[N-1])/(2*h)});
	}

	Eigen::SparseMatrix<double> A(N,N);
	A.setFromTriplets(coefficients.begin(), coefficients.end());
	return A;
}

Eigen::VectorXd loadVector(Eigen::VectorXd fx, double h, BC lbc, BC rbc) {
	int N = fx.size();
	Eigen::VectorXd b(N);

	for (int i=1; i<N-1; ++i)
		b(i) = h*fx[i];

	if (lbc.type == BCType::Dirichlet)
		b(0) = lbc.value;
	else
		b(0) = h/2*fx[0]-lbc.value;
	if (rbc.type == BCType::Dirichlet)
		b(N-1) = rbc.value;
	else
		b(N-1) = h/2*fx[N-1]+rbc.value;

	return b;
}

double errorTrapz(Eigen::VectorXd u, Eigen::VectorXd solution, double h) {
	int N = u.size();
	double errorIntegral = ((u(0)-solution(0))*(u(0)-solution(0))+(u(N-1)-solution(N-1))*(u(N-1)-solution(N-1)))/2;
	for (int i = 1; i < N-2; ++i)
		errorIntegral += (u(i)-solution(i))*(u(i)-solution(i));
	return sqrt(h*errorIntegral);
}

int main() {
	// setup
	auto c = [](double x) { return 1.0; };
	auto v = [](double x) { return 0.0; };
	auto f = [](double x) { return 6*x; };
	BC lbc = {Dirichlet, 0};
	BC rbc = {Neumann, 0};

	std::vector<double> errors;

	for (int i = 2; i < 11; ++i) {
		size_t N = (2 << i)+1;

		// assembly
		Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 1);
		double h = x[1]-x[0];

		Eigen::VectorXd cx = x.unaryExpr(c);
		Eigen::VectorXd vx = x.unaryExpr(v);
		Eigen::VectorXd fx = x.unaryExpr(f);

		Eigen::SparseMatrix<double> A = stiffnessMatrix(cx, vx, h, lbc, rbc);
		Eigen::VectorXd b = loadVector(fx, h, lbc, rbc);

		// solution
		Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(A);	
		Eigen::VectorXd u = solver.solve(b);

		errors.push_back(errorTrapz(u, x.unaryExpr([](double x) { return -x*x*x+3*x; }), h));
	}

	for (int i = 1; i < errors.size(); ++i) {
		std::cout << log2(errors[i-1]/errors[i]) << "\n";
	}

	return 0;
}
