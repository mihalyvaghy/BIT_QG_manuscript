#include "quantum_graph.h"
#include <chrono>
#include <iostream>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

std::unordered_set<int> setDiff(std::unordered_set<int>& s1, std::unordered_set<int>& s2) {
	std::unordered_set<int> result;
	for (auto it = s1.begin(); it != s1.end(); ++it)
		if (!s2.contains(*it))
			result.emplace(*it);
	return result;
}

std::unordered_set<int> computeBoundary(int vertices, std::vector<QGEdge>& edges) {
	std::vector<int> degrees(vertices, 0);
	for (QGEdge edge: edges) {
		++degrees[edge.out];
		++degrees[edge.in];
	}

	std::unordered_set<int> boundary;
	for (int i = 0; i < degrees.size(); ++i)
		if (degrees[i] == 1)
			boundary.emplace(i);

	return boundary;
}

class MatrixFreeQuantumGraph;

namespace Eigen {
namespace internal {
template<>
struct traits<MatrixFreeQuantumGraph> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}
}

class MatrixFreeQuantumGraph : public Eigen::EigenBase<MatrixFreeQuantumGraph> {
private:
	int subproblems;
	int vertices;
	std::vector<std::unordered_map<int,int>> global_local_maps;
	std::vector<std::unordered_map<int,int>> local_global_maps;
	std::vector<std::vector<QGEdge>> edges_container;
	std::vector<std::vector<int>> local_boundaries_indices;
	std::vector<std::vector<int>> global_boundaries_indices;
	std::vector<QuantumGraph*> quantum_graphs;

public:
	typedef double Scalar;
	typedef double RealScalar;
	typedef int StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = false
	};

	Index rows() const { return vertices; }
	Index cols() const { return vertices; }

	template<typename Rhs>
	Eigen::Product<MatrixFreeQuantumGraph, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
		return Eigen::Product<MatrixFreeQuantumGraph, Rhs, Eigen::AliasFreeProduct> (*this, x.derives());
	}

	MatrixFreeQuantumGraph(int N, int _vertices, std::vector<QGEdge>& edges, std::vector<std::unordered_set<int>>& domains, std::vector<std::set<std::pair<int,int>>>& omitted_edges) : vertices(_vertices) {
		subproblems = domains.size();

		std::unordered_set<int> global_boundary = computeBoundary(vertices, edges);
		for (int i = 0; i < domains.size(); ++i) {
			std::unordered_set<int> domain = domains[i];
			int local_vertices = domain.size();
			std::vector<QGEdge> local_edges;
			std::unordered_map<int,int> global_local_map;
			std::unordered_map<int,int> local_global_map;

			for (auto it = domain.begin(); it != domain.end(); ++it) {
				global_local_map[*it] = std::distance(domain.begin(), it);
				local_global_map[std::distance(domain.begin(), it)] = *it;
			}
			for (QGEdge edge: edges) {
				if (domain.contains(edge.out) and domain.contains(edge.in) and !omitted_edges[i].contains({edge.out, edge.in})) {
					edge.out = global_local_map[edge.out];
					edge.in = global_local_map[edge.in];
					local_edges.push_back(edge);
				}
			}
			std::unordered_set<int> tmp_local_boundary = computeBoundary(local_vertices, local_edges);
			std::unordered_set<int> local_boundary_tmp;
			std::vector<int> local_boundary_indices;
			std::vector<int> global_boundary_indices;
			int global_index;
			for (auto it = tmp_local_boundary.begin(); it != tmp_local_boundary.end(); ++it) {
				global_index = local_global_map[*it];
				if (!global_boundary.contains(global_index)) {
					local_boundary_tmp.emplace(*it);
					local_boundary_indices.emplace_back(*it);
					global_boundary_indices.emplace_back(global_index);
				}
			}

			edges_container.push_back(local_edges);
			global_local_maps.push_back(global_local_map);
			local_global_maps.push_back(local_global_map);
			local_boundaries_indices.push_back(local_boundary_indices);
			global_boundaries_indices.push_back(global_boundary_indices);
			quantum_graphs.push_back(new QuantumGraph(N, local_vertices, local_edges, local_boundary_tmp));
		}
	}

	~MatrixFreeQuantumGraph() {
		for (auto qg: quantum_graphs)
		delete qg;
		quantum_graphs.clear();
	}

	int get_subproblems() const { return subproblems; }
	int get_vertices() const { return vertices; }
	std::unordered_map<int,int> get_global_local_map(int ix) const { return global_local_maps[ix]; }
	std::unordered_map<int,int> get_local_global_map(int ix) const { return local_global_maps[ix]; }
	std::vector<QGEdge> get_edges(int ix) const { return edges_container[ix]; }
	std::vector<int> get_local_boundary_indices(int ix) const { return local_boundaries_indices[ix]; }
	std::vector<int> get_global_goundary_indices(int ix) const { return global_boundaries_indices[ix]; }
	QuantumGraph* get_quantum_graph(int ix) const { return quantum_graphs[ix]; }
};

namespace Eigen {
namespace internal {

template<typename Rhs>
struct generic_product_impl<MatrixFreeQuantumGraph, Rhs,SparseShape, DenseShape, GemvProduct> : generic_product_impl_base<MatrixFreeQuantumGraph, Rhs, generic_product_impl<MatrixFreeQuantumGraph, Rhs>> {
	typedef typename Product<MatrixFreeQuantumGraph, Rhs>::Scalar Scalar;

	template<typename Dest>
	static void scaleAndAddto(Dest& dst, const MatrixFreeQuantumGraph& lhs, const Rhs& rhs, const Scalar& alpha) {
		assert(alpha == Scalar(1) && "scaling is not implemented");
		EIGEN_ONLY_USED_FOR_DEBUG(alpha);

		Eigen::VectorXd subres = Eigen::VectorXd::Zero(lhs.get_vertices());
		for (int ix = 0; ix < lhs.get_subproblems(); ++ix) {
			lhs.get_quantum_graph(ix)->setDirichletValues(lhs.get_local_boundary_indices(ix), rhs(lhs.get_global_goundary_indices(ix)));

			lhs.get_quantum_graph(ix)->solve();
			Eigen::VectorXd tmp = lhs.get_quantum_graph(ix)->getVertexValues();
			for (int j = 0; j < tmp.size(); ++j)
				subres(lhs.get_local_global_map(ix)[j]) = tmp(j);
		}

		dst.noalias() += alpha*subres;
	}
};
}
}

Eigen::VectorXd multiplicative_schwarz(int N, int vertices, std::vector<QGEdge>& edges, std::vector<std::unordered_set<int>>& domains, std::vector<std::set<std::pair<int,int>>>& omitted_edges, int max_iter) {
	int subproblems = domains.size();

	// data
	std::vector<std::unordered_map<int,int>> global_local_maps;
	std::vector<std::unordered_map<int,int>> local_global_maps;
	std::vector<std::vector<QGEdge>> edges_container;
	std::vector<std::vector<int>> local_boundaries_indices;
	std::vector<std::vector<int>> global_boundaries_indices;
	std::vector<QuantumGraph*> quantum_graphs;

	// setup
	std::unordered_set<int> global_boundary = computeBoundary(vertices, edges);
	for (int i = 0; i < domains.size(); ++i) {
		std::unordered_set<int> domain = domains[i];
		int local_vertices = domain.size();
		std::vector<QGEdge> local_edges;
		std::unordered_map<int,int> global_local_map;
		std::unordered_map<int,int> local_global_map;

		for (auto it = domain.begin(); it != domain.end(); ++it) {
			global_local_map[*it] = std::distance(domain.begin(), it);
			local_global_map[std::distance(domain.begin(), it)] = *it;
		}
		for (QGEdge edge: edges) {
			if (domain.contains(edge.out) and domain.contains(edge.in) and !omitted_edges[i].contains({edge.out, edge.in})) {
				edge.out = global_local_map[edge.out];
				edge.in = global_local_map[edge.in];
				local_edges.push_back(edge);
			}
		}
		std::unordered_set<int> tmp_local_boundary = computeBoundary(local_vertices, local_edges);
		std::unordered_set<int> local_boundary_tmp;
		std::vector<int> local_boundary_indices;
		std::vector<int> global_boundary_indices;
		int global_index;
		for (auto it = tmp_local_boundary.begin(); it != tmp_local_boundary.end(); ++it) {
			global_index = local_global_map[*it];
			if (!global_boundary.contains(global_index)) {
				local_boundary_tmp.emplace(*it);
				local_boundary_indices.emplace_back(*it);
				global_boundary_indices.emplace_back(global_index);
			}
		}

		edges_container.push_back(local_edges);
		global_local_maps.push_back(global_local_map);
		local_global_maps.push_back(local_global_map);
		local_boundaries_indices.push_back(local_boundary_indices);
		global_boundaries_indices.push_back(global_boundary_indices);
		quantum_graphs.push_back(new QuantumGraph(N, local_vertices, local_edges, local_boundary_tmp));
	}

	// solution
	Eigen::VectorXd vertex_values = Eigen::VectorXd::Zero(vertices);
	for (int i = 0; i < max_iter; ++i) {
		for (int ix = 0; ix < domains.size(); ++ix) {
			quantum_graphs[ix]->setDirichletValues(local_boundaries_indices[ix], vertex_values(global_boundaries_indices[ix]));
			quantum_graphs[ix]->solve();
			Eigen::VectorXd tmp = quantum_graphs[ix]->getVertexValues();
			for (int j = 0; j < tmp.size(); ++j)
				vertex_values(local_global_maps[ix][j]) = tmp(j);
		}
	}

	for (auto qg: quantum_graphs)
	delete qg;
	quantum_graphs.clear();

	return vertex_values;
}

Eigen::VectorXd additive_schwarz(int N, int vertices, std::vector<QGEdge>& edges, std::vector<std::unordered_set<int>>& domains, std::vector<std::set<std::pair<int,int>>>& omitted_edges, int max_iter) {
	int subproblems = domains.size();

	// data
	std::vector<std::unordered_map<int,int>> global_local_maps;
	std::vector<std::unordered_map<int,int>> local_global_maps;
	std::vector<std::vector<QGEdge>> edges_container;
	std::vector<std::vector<int>> local_boundaries_indices;
	std::vector<std::vector<int>> global_boundaries_indices;
	std::vector<QuantumGraph*> quantum_graphs;
	std::vector<Eigen::VectorXd> partition_of_unity;

	// setup
	std::unordered_set<int> global_boundary = computeBoundary(vertices, edges);
	for (int i = 0; i < domains.size(); ++i) {
		std::unordered_set<int> domain = domains[i];
		int local_vertices = domain.size();
		std::vector<QGEdge> local_edges;
		std::unordered_map<int,int> global_local_map;
		std::unordered_map<int,int> local_global_map;
		Eigen::VectorXd part_unity = Eigen::VectorXd::Constant(local_vertices, 1);

		for (auto it = domain.begin(); it != domain.end(); ++it) {
			global_local_map[*it] = std::distance(domain.begin(), it);
			local_global_map[std::distance(domain.begin(), it)] = *it;
		}
		for (QGEdge edge: edges) {
			if (domain.contains(edge.out) and domain.contains(edge.in) and !omitted_edges[i].contains({edge.out, edge.in})) {
				edge.out = global_local_map[edge.out];
				edge.in = global_local_map[edge.in];
				local_edges.push_back(edge);
			}
		}
		std::unordered_set<int> tmp_local_boundary = computeBoundary(local_vertices, local_edges);
		std::unordered_set<int> local_boundary_tmp;
		std::vector<int> local_boundary_indices;
		std::vector<int> global_boundary_indices;
		for (auto it = tmp_local_boundary.begin(); it != tmp_local_boundary.end(); ++it) {
			int global_index = local_global_map[*it];
			if (!global_boundary.contains(global_index)) {
				local_boundary_tmp.emplace(*it);
				local_boundary_indices.emplace_back(*it);
				global_boundary_indices.emplace_back(global_index);
				part_unity(*it) = 0;
			}
		}

		edges_container.push_back(local_edges);
		global_local_maps.push_back(global_local_map);
		local_global_maps.push_back(local_global_map);
		local_boundaries_indices.push_back(local_boundary_indices);
		global_boundaries_indices.push_back(global_boundary_indices);
		quantum_graphs.push_back(new QuantumGraph(N, local_vertices, local_edges, local_boundary_tmp));
		partition_of_unity.push_back(part_unity);
	}

	// solution
	Eigen::VectorXd vertex_values = Eigen::VectorXd::Zero(vertices);
	for (int i = 0; i < max_iter; ++i) {
		for (int ix = 0; ix < domains.size(); ++ix) {
			quantum_graphs[ix]->setDirichletValues(local_boundaries_indices[ix], vertex_values(global_boundaries_indices[ix]));
			quantum_graphs[ix]->solve();
		}
		vertex_values = Eigen::VectorXd::Zero(vertices);
		for (int ix = 0; ix < domains.size(); ++ix) {
			Eigen::VectorXd local_vertex_values = quantum_graphs[ix]->getVertexValues().cwiseProduct(partition_of_unity[ix]);
			for (int j = 0; j < local_vertex_values.size(); ++j)
				vertex_values(local_global_maps[ix][j]) += local_vertex_values(j);
		}
	}

	for (auto qg: quantum_graphs)
	delete qg;
	quantum_graphs.clear();

	return vertex_values;
}

int main() {
	// iteration setup
	int N = 1024-1+2;
	int max_iter = 3;
	//	omp_set_num_threads(2); // max is 8

	// bolt problem definition
	/*
		 int vertices = 4;
		 std::vector<QGEdge> edges;
		 edges.push_back({0, 1, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({1, 2, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({2, 3, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});

		 std::vector<std::unordered_set<int>> domains = {{0, 1, 2}, {1, 2, 3}};
		 std::vector<std::unordered_set<int>> omitted_edges = {{}, {}};
	*/

	// hourglass problem definition
	/*
		 int vertices = 6;
		 std::vector<QGEdge> edges;
		 edges.push_back({0, 1, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({0, 2, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({1, 2, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({2, 3, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({3, 4, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({3, 5, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});
		 edges.push_back({4, 5, N,
		 [](double x){return 1/(2-x*x);},
		 [](double x){return 2-x;},
		 [](double x){return 2+x;}});

		 std::vector<std::unordered_set<int>> domains = {{0, 1, 2, 3}, {2, 3, 4, 5}};
		 std::vector<std::set<std::pair<int,int>>> omitted_edges = {{}, {}};
	*/

	// star-windmill
	int num = 15;
	int size = 15;
	int vertices = num*size+1;
	std::vector<QGEdge> edges;
	for (int i = 0; i < num; ++i) {
		edges.emplace_back(0, size*i+1,
			[](double x){return 1/(2-x*x);},
			[](double x){return 2-x;},
			[](double x){return 2+x;});
		for (int j = 1; j < size+1; ++j) {
			for (int k = j+1; k < size+1; ++k) {
				edges.emplace_back(size*i+j, size*i+k,
					[](double x){return 1/(2-x*x);},
					[](double x){return 2-x;},
					[](double x){return 2+x;});
			}
		}
	}
	std::unordered_set<int> dirichlet_boundary;
	std::vector<std::unordered_set<int>> domains(num+1);
	domains[0].insert(0);
	for (int i = 0; i < num; ++i) {
		domains[0].insert(size*i+1);
		domains[i+1].insert(0);
		for (int j = 1; j < size+1; ++j) {
			domains[i+1].insert(size*i+j);
		}
	}
	std::vector<std::set<std::pair<int,int>>> omitted_edges(num+1);

	// multiplicative Schwarz solution
	auto multiplicative_schwarz_start{std::chrono::steady_clock::now()};
	Eigen::VectorXd multiplicative_schwarz_vertex_values = multiplicative_schwarz(N, vertices, edges, domains, omitted_edges, max_iter);
	auto multiplicative_schwarz_end{std::chrono::steady_clock::now()};
	std::chrono::duration<double> multiplicative_schwarz_time{multiplicative_schwarz_end-multiplicative_schwarz_start};

	// additive Schwarz solution
	auto additive_schwarz_start{std::chrono::steady_clock::now()};
	Eigen::VectorXd additive_schwarz_vertex_values = additive_schwarz(N, vertices, edges, domains, omitted_edges, max_iter);
	auto additive_schwarz_end{std::chrono::steady_clock::now()};
	std::chrono::duration<double> additive_schwarz_time{additive_schwarz_end-additive_schwarz_start};

	// evaluation
	std::cout << "MULTIPLICATIVE SCHWARZ\nTIME: " << multiplicative_schwarz_time << "\n";
	std::cout << "ADDITIVE SCHWARZ\nTIME: " << additive_schwarz_time << "\n";
	std::cout << "FEM\nTIME: " << fem_time << "\n";

	// iterative stuff
	/*
	MatrixFreeQuantumGraph A(N, vertices, edges, domains, omitted_edges);
	Eigen::BiCGSTAB<MatrixFreeQuantumGraph, Eigen::IdentityPreconditioner> bicg;
	bicg.compute(A);
	Eigen::VectorXd b = A.compute_b();
	Eigen::VectorXd bicg_sol = bicg.solve(b);
	*/

	return 0;
}
