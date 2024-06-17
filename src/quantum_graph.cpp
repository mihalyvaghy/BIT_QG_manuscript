#include "quantum_graph.h"

QuantumGraph::QuantumGraph(int _N, int _vertices, std::vector<QGEdge>& edges, std::unordered_set<int>& dirichlet_vertices) : N(_N), vertices(_vertices) {
	const int size = edges.size()*(N-2)+vertices;
	const int nnz = edges.size()*(3*(N-4)+2*2+2*2)+vertices;
	vertex_offset = edges.size()*(N-2);

	std::vector<Eigen::Triplet<double>> coefficients;
	coefficients.reserve(nnz);
	b.resize(size);
	b.tail(vertices) = Eigen::VectorXd::Zero(vertices);
	std::vector<double> vertex_stiffness(vertices, 0.0);
	std::vector<double> vertex_load(vertices, 0.0);

	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 1);
	Eigen::VectorXd cx, vx, fx;
	double h = x(1)-x(0);
	int edge_offset;
	double lcoeff, rcoeff;
	int out, in;
	for (int i = 0; i < edges.size(); ++i) {
		QGEdge edge = edges[i];
		out = edge.out;
		in = edge.in;

		cx = x.unaryExpr(edge.c);
		vx = x.unaryExpr(edge.v);
		fx = x.unaryExpr(edge.f);

		edge_offset = i*(N-2);

		coefficients.emplace_back(edge_offset, edge_offset, (cx(0)+2*cx(1)+cx(2))/(2*h)+h*vx(1));
		coefficients.emplace_back(edge_offset, edge_offset+1, -(cx(1)+cx(2))/(2*h));
		for (int i=1; i<N-3; ++i) {
			coefficients.emplace_back(edge_offset+i, edge_offset+i-1, -(cx(i)+cx(i+1))/(2*h));
			coefficients.emplace_back(edge_offset+i, edge_offset+i, (cx(i)+2*cx(i+1)+cx(i+2))/(2*h)+h*vx(i+1));
			coefficients.emplace_back(edge_offset+i, edge_offset+i+1, -(cx(i+1)+cx(i+2))/(2*h));
		}
		coefficients.emplace_back(edge_offset+N-3, edge_offset+N-4, -(cx(N-3)+cx(N-2))/(2*h));
		coefficients.emplace_back(edge_offset+N-3, edge_offset+N-3, (cx(N-3)+2*cx(N-2)+cx(N-1))/(2*h)+h*vx(N-2));

		lcoeff = -(cx(0)+cx(1))/(2*h);
		rcoeff = -(cx(N-2)+cx(N-1))/(2*h);
		coefficients.emplace_back(edge_offset, vertex_offset+out, lcoeff);
		if (!dirichlet_vertices.contains(out))
			coefficients.emplace_back(vertex_offset+out, edge_offset, lcoeff);
		coefficients.emplace_back(edge_offset+N-3, vertex_offset+in, rcoeff);
		if (!dirichlet_vertices.contains(in))
			coefficients.emplace_back(vertex_offset+in, edge_offset+N-3, rcoeff);
		vertex_stiffness[out] += -lcoeff+vx(0)*h/2;
		vertex_stiffness[in] += -rcoeff+vx(N-1)*h/2;

		b.middleRows(edge_offset, N-2) = h*fx(Eigen::seq(1, N-2));
		b(vertex_offset+out) += fx(0)*h/2;
		b(vertex_offset+in) += fx(N-1)*h/2;
	}

	for (int i = 0; i < vertex_stiffness.size(); ++i) {
		if (dirichlet_vertices.contains(i))
			coefficients.emplace_back(vertex_offset+i, vertex_offset+i, 1.0);
		else
			coefficients.emplace_back(vertex_offset+i, vertex_offset+i, vertex_stiffness[i]);
	}

	A.resize(size, size);
	A.reserve(nnz);
	A.setFromTriplets(coefficients.begin(), coefficients.end());

	solver.compute(A);
}

void QuantumGraph::setDirichletValues(std::vector<int>& indices, Eigen::VectorXd& values) {
	for (int i = 0; i < indices.size(); ++i)
		b(vertex_offset+indices[i]) = values(i);
}

void QuantumGraph::solve() {
	u = solver.solve(b);
}

Eigen::VectorXd QuantumGraph::getVertexValues() {
	return u.tail(vertices);
}
