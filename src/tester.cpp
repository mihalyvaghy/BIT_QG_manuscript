#include "example.h"
#include "mf_diagonal_quantum_graph.h"
#include "mf_neumann_neumann.h"
#include "mf_polynomial_quantum_graph.h"
#include "mf_quantum_graph.h"
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

struct Result {
	double time;
	int iterations;
	double error;
};

template<typename Problem, typename Preconditioner, template<typename, typename> class Solver> 
Result run_mf_example(Example& ex, int N, double eps) {
	auto start = std::chrono::system_clock::now();
	Problem problem(N, ex.vertices, ex.edges);
	Solver<Problem, Preconditioner> solver;
	solver.setTolerance(eps);
	solver.compute(problem);
	Eigen::VectorXd solution = solver.solve(problem.bG);
	auto end = std::chrono::system_clock::now();
	return {std::chrono::duration<double>(end-start).count(), static_cast<int>(solver.iterations()), solver.error()};
}

template<typename Problem, typename Preconditioner, template<typename, typename> class Solver> 
void measure_mf_example(Example& ex, int N, double eps, int runs) {
	double time = 0;
	Result res;
	for (int i = 0; i < runs; ++i) {
		Result res = run_mf_example<Problem, Preconditioner, Solver>(ex, N, eps);
		time += res.time;
	}
	std::cout << "runtime: " << time/runs << " iterations: " << res.iterations << " error: " << res.error << "\n";
}

int main(int argc, char* argv[]) {

	std::string graph = argv[1];
	int size = std::stoi(argv[2]);
	int logN = std::stoi(argv[3]);
	int runs = 1;
	if (argc == 5)
		runs = std::stoi(argv[4]);

	int N = pow(2, logN)-1+2;
	double eps = sqrt(2.2204e-16);
	std::string filename = graph+"_"+argv[2];
	Example ex(filename);

	std::cout << "Neumann-Neumann\n";
	measure_mf_example<MFNeumannNeumann, Eigen::IdentityPreconditioner, Eigen::BiCGSTAB>(ex, N, eps, runs);

	std::cout << "\nVanilla\n";
	measure_mf_example<MFQuantumGraph, Eigen::IdentityPreconditioner, Eigen::BiCGSTAB>(ex, N, eps, runs);

	std::cout << "\nDiagonal\n";
	measure_mf_example<MFDiagonalQuantumGraph, Eigen::IdentityPreconditioner, Eigen::BiCGSTAB>(ex, N, eps, runs);

	std::cout << "\nPolynomial\n";
	measure_mf_example<MFPolynomialQuantumGraph, Eigen::IdentityPreconditioner, Eigen::BiCGSTAB>(ex, N, eps, runs);

	return 0;
}
