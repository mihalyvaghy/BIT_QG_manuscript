#include "neumann_neumann.h"

NeumannNeumann::NeumannNeumann(int N, int _vertices, std::vector<QGEdge>& edges, int _max_iter, double _theta) : vertices(_vertices), max_iter(_max_iter), theta(_theta), vertex_to_edges(_vertices), vertex_weights(_vertices) {
	// data
	dirichlet_edges.reserve(edges.size());
	neumann_edges.reserve(edges.size());
	vertex_values = Eigen::VectorXd::Zero(vertices);

	// setup
	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 1);
	double h = x(1)-x(0);
	Eigen::VectorXd	cx;
	Eigen::VectorXd	vx;
	Eigen::VectorXd	dirichlet_fx;
	Eigen::VectorXd neumann_fx = Eigen::VectorXd::Zero(N);
	BC dirichlet_lbc, dirichlet_rbc;
	BC neumann_lbc, neumann_rbc;

	int nnz = (N-2)*3+2*2;
	Eigen::SparseMatrix<double> local_A;
	std::vector<Eigen::Triplet<double>> coefficients;
	local_A.resize(N, N);
	local_A.reserve(nnz);

	for (int i = 0; i < edges.size(); ++i) {
		int out = edges[i].out;
		int in = edges[i].in;
		vertex_to_edges[out].emplace(i);
		vertex_to_edges[in].emplace(i);
		++vertex_weights[out];
		++vertex_weights[in];
	}

	for (int i = 0; i < vertices; ++i)
		if (vertex_weights[i] == 1)
			boundary.emplace(i);

	int out, in;
	for (QGEdge& edge: edges) {
		out = edge.out;
		in = edge.in;
		cx = x.unaryExpr(edge.c);
		vx = x.unaryExpr(edge.v);
		dirichlet_fx = x.unaryExpr(edge.f);

		if (boundary.contains(out))
			dirichlet_lbc = {BCType::Neumann, 0.0};
		else
			dirichlet_lbc = {BCType::Dirichlet, 0.0};
		if (boundary.contains(in))
			dirichlet_rbc = {BCType::Neumann, 0.0};
		else
			dirichlet_rbc = {BCType::Dirichlet, 0.0};
		neumann_lbc = {BCType::Neumann, 0.0};
		neumann_rbc = {BCType::Neumann, 0.0};

		coefficients.reserve(nnz);
		for (int i=1; i<N-1; ++i) {
			coefficients.emplace_back(i, i-1, -(cx(i-1)+cx(i))/(2*h));
			coefficients.emplace_back(i, i, (cx(i-1)+2*cx(i)+cx(i+1))/(2*h)+h*vx(i));
			coefficients.emplace_back(i, i+1, -(cx(i)+cx(i+1))/(2*h));
		}

		coefficients.emplace_back(0, 0, 1);
		coefficients.emplace_back(N-1, N-1, 1);
		local_A.setFromTriplets(coefficients.begin(), coefficients.end());
		dirichlet_edges.push_back(new NNEdge(out, in, N, cx, vx, dirichlet_fx, local_A, dirichlet_lbc, dirichlet_rbc));

		coefficients.pop_back();
		coefficients.pop_back();
		coefficients.emplace_back(0, 0, (cx(0)+cx(1))/(2*h)+h/2*vx(0));
		coefficients.emplace_back(0, 1, -(cx(0)+cx(1))/(2*h));
		coefficients.emplace_back(N-1, N-1, (cx(N-2)+cx(N-1))/(2*h)+h/2*vx(N-1));
		coefficients.emplace_back(N-1, N-2, -(cx(N-2)+cx(N-1))/(2*h));

		local_A.setFromTriplets(coefficients.begin(), coefficients.end());
		neumann_edges.push_back(new NNEdge(out, in, N, cx, vx, neumann_fx, local_A, neumann_lbc, neumann_rbc));

		coefficients.clear();
	}
}

Eigen::VectorXd NeumannNeumann::solve() {
	Eigen::VectorXd vertex_values_update = Eigen::VectorXd::Zero(vertices);
	int out, in;
	double neumann_lbc_value, neumann_rbc_value;
	for (int iter = 0; iter < max_iter; ++iter) {
		vertex_values	-= theta*vertex_values_update;
		vertex_values_update = Eigen::VectorXd::Zero(vertices);

		for (NNEdge* edge: dirichlet_edges) {
			out = edge->out;
			in = edge->in;
			if (!boundary.contains(out))
				edge->setBC(out, vertex_values(out));
			if (!boundary.contains(in))
				edge->setBC(in, vertex_values(in));
			edge->solve();
		}

		for (NNEdge* edge: neumann_edges) {
			out = edge->out;
			in = edge->in;
			neumann_lbc_value = 0;
			neumann_rbc_value = 0;
			if (!boundary.contains(out)) {
				for (int neighbour: vertex_to_edges[out])
					neumann_lbc_value -= dirichlet_edges[neighbour]->approxNeumann(out);
				edge->setBC(out, -neumann_lbc_value/vertex_weights[out]);
			}
			if (!boundary.contains(in)) {
				for (int neighbour: vertex_to_edges[in])
					neumann_rbc_value -= dirichlet_edges[neighbour]->approxNeumann(in);
			edge->setBC(in, neumann_rbc_value/vertex_weights[in]);
			}
			edge->solve();
			vertex_values_update(out) += edge->Dirichlet(out)/vertex_weights[out];
			vertex_values_update(in) += edge->Dirichlet(in)/vertex_weights[in];
		}
	}

	return vertex_values;
}
NeumannNeumann::~NeumannNeumann() {
	for (NNEdge* edge: dirichlet_edges)
		delete edge;
	dirichlet_edges.clear();
	for (NNEdge* edge: neumann_edges)
		delete edge;
	neumann_edges.clear();
}
