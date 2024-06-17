#include "mf_neumann_neumann.h"

MFNeumannNeumann::MFNeumannNeumann(int _N, int _vertices, std::vector<QGEdge>& _edges) : N(_N), vertices(_vertices), edges(_edges), dirichlet_solvers(edges.size()), neumann_solvers(edges.size()), vertex_to_edges(_vertices), vertex_weights(_vertices) {
	const int sizeI = N-2;
	const int sizeG = vertices;

	const int nnzII = 3*(N-4)+2*2;
	const int nnzIG = 2;
	const int nnzGG = 2;

	Eigen::SparseMatrix<double> local_AII(sizeI, sizeI);
	std::vector<Eigen::Triplet<double>> local_AII_coefficients;
	local_AII.reserve(nnzII);
	Eigen::SparseMatrix<double> local_A(sizeI+2, sizeI+2);
	std::vector<Eigen::Triplet<double>> local_A_coefficients;
	local_A.reserve(nnzII+2*nnzIG+nnzGG);

	AIGs.reserve(_vertices);
	AGGs.reserve(_vertices);
	bIs.reserve(_vertices);
	bGs.reserve(_vertices);
	bG = Eigen::VectorXd::Zero(vertices);
	vertex_values = Eigen::VectorXd::Zero(vertices);

	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N, 0, 1);
	double h = x(1)-x(0);
	Eigen::VectorXd	cx;
	Eigen::VectorXd	vx;
	Eigen::VectorXd	dirichlet_fx;

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

	int ix = 0;
	int out, in;
	double lcoeff, rcoeff;
	for (QGEdge& edge: edges) {
		out = edge.out;
		in = edge.in;
		cx = x.unaryExpr(edge.c);
		vx = x.unaryExpr(edge.v);
		dirichlet_fx = x.unaryExpr(edge.f);
		lcoeff = -(cx(0)+cx(1))/(2*h);
		rcoeff = -(cx(N-2)+cx(N-1))/(2*h);

		local_A_coefficients.clear();
		local_A_coefficients.reserve(nnzII+2*nnzIG+nnzGG);
		local_A_coefficients.emplace_back(0, 0, (cx(0)+2*cx(1)+cx(2))/(2*h)+h*vx(1));
		local_A_coefficients.emplace_back(0, 1, -(cx(1)+cx(2))/(2*h));
		for (int i=1; i<N-3; ++i) {
			local_A_coefficients.emplace_back(i, i-1, -(cx(i)+cx(i+1))/(2*h));
			local_A_coefficients.emplace_back(i, i, (cx(i)+2*cx(i+1)+cx(i+2))/(2*h)+h*vx(i+1));
			local_A_coefficients.emplace_back(i, i+1, -(cx(i+1)+cx(i+2))/(2*h));
		}
		local_A_coefficients.emplace_back(N-3, N-4, -(cx(N-3)+cx(N-2))/(2*h));
		local_A_coefficients.emplace_back(N-3, N-3, (cx(N-3)+2*cx(N-2)+cx(N-1))/(2*h)+h*vx(N-2));
		local_A.setFromTriplets(local_A_coefficients.begin(), local_A_coefficients.end());
		local_A.insert(0, sizeI) = lcoeff;
		local_A.insert(sizeI, 0) = lcoeff;
		local_A.insert(N-3, sizeI+1) = rcoeff;
		local_A.insert(sizeI+1, N-3) = rcoeff;
		local_A.insert(sizeI, sizeI) = -lcoeff+vx(0)*h/2;
		local_A.insert(sizeI+1, sizeI+1) = -rcoeff+vx(N-1)*h/2;

		local_AII_coefficients.clear();
		local_AII_coefficients.reserve(nnzII);
		local_AII_coefficients.emplace_back(0, 0, (cx(0)+2*cx(1)+cx(2))/(2*h)+h*vx(1));
		local_AII_coefficients.emplace_back(0, 1, -(cx(1)+cx(2))/(2*h));
		for (int i=1; i<N-3; ++i) {
			local_AII_coefficients.emplace_back(i, i-1, -(cx(i)+cx(i+1))/(2*h));
			local_AII_coefficients.emplace_back(i, i, (cx(i)+2*cx(i+1)+cx(i+2))/(2*h)+h*vx(i+1));
			local_AII_coefficients.emplace_back(i, i+1, -(cx(i+1)+cx(i+2))/(2*h));
		}
		local_AII_coefficients.emplace_back(N-3, N-4, -(cx(N-3)+cx(N-2))/(2*h));
		local_AII_coefficients.emplace_back(N-3, N-3, (cx(N-3)+2*cx(N-2)+cx(N-1))/(2*h)+h*vx(N-2));
		local_AII.setFromTriplets(local_AII_coefficients.begin(), local_AII_coefficients.end());

		AIGs.push_back(new Eigen::SparseMatrix<double>(sizeI, sizeG));
		AIGs[ix]->reserve(nnzIG);
		AIGs[ix]->insert(0, out) = lcoeff;
		AIGs[ix]->insert(N-3, in) = rcoeff;

		AGGs.push_back(new Eigen::SparseMatrix<double>(sizeG, sizeG));
		AGGs[ix]->reserve(nnzGG);
		AGGs[ix]->insert(out, out) = -lcoeff+vx(0)*h/2;
		AGGs[ix]->insert(in, in) = -rcoeff+vx(N-1)*h/2;

		bIs.push_back(new Eigen::VectorXd(sizeI));
		(*bIs[ix]) = h*dirichlet_fx(Eigen::seq(1, N-2));

		bGs.push_back(new Eigen::VectorXd(sizeG));
		(*bGs[ix]) = Eigen::VectorXd::Zero(sizeG);
		(*bGs[ix])(out) = dirichlet_fx(0)*h/2;
		(*bGs[ix])(in) = dirichlet_fx(N-1)*h/2;

		dirichlet_solvers[ix].compute(local_AII);
		neumann_solvers[ix].compute(local_A);
		bG += (*bGs[ix])-AIGs[ix]->transpose()*dirichlet_solvers[ix].solve(*bIs[ix]);
		++ix;
	}

	Eigen::VectorXd neumann_fx = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd newbG = Eigen::VectorXd::Zero(vertices);
	Eigen::VectorXd tmp;
	for (int i = 0; i < neumann_solvers.size(); ++i) {
		out = edges[i].out;
		in = edges[i].in;
		neumann_fx(N-2) = bG(out)/vertex_weights[out];
		neumann_fx(N-1) = bG(in)/vertex_weights[in];
		tmp = neumann_solvers[i].solve(neumann_fx);
		newbG(out) += tmp(N-2)/vertex_weights[out];
		newbG(in) += tmp(N-1)/vertex_weights[in];
	}
	bG = newbG;
}

MFNeumannNeumann::~MFNeumannNeumann() {
	for (auto AIG: AIGs)
		delete AIG;
	AIGs.clear();
	for (auto AGG: AGGs)
		delete AGG;
	AGGs.clear();
	for (auto bI: bIs)
		delete bI;
	bIs.clear();
	for (auto bG: bGs)
		delete bG;
	bGs.clear();
}
