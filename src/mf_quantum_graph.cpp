#include "mf_quantum_graph.h"

MFQuantumGraph::MFQuantumGraph(int N, int vertices, const std::vector<QGEdge>& edges) {
	size = vertices;
	const int sizeI = edges.size()*(N-2);
	const int sizeG = vertices;

	const int nnzII = edges.size()*(3*(N-4)+2*2);
	const int nnzIG = edges.size()*2;
	const int nnzGG = vertices;

	std::vector<Eigen::Triplet<double>> coefficientsII, coefficientsIG, coefficientsGG;
	coefficientsII.reserve(nnzII);
	coefficientsIG.reserve(nnzIG);
	coefficientsGG.reserve(nnzGG);
	bI.resize(sizeI);
	bG = Eigen::VectorXd::Zero(vertices);

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

		coefficientsII.emplace_back(edge_offset, edge_offset, (cx(0)+2*cx(1)+cx(2))/(2*h)+h*vx(1));
		coefficientsII.emplace_back(edge_offset, edge_offset+1, -(cx(1)+cx(2))/(2*h));
		for (int i=1; i<N-3; ++i) {
			coefficientsII.emplace_back(edge_offset+i, edge_offset+i-1, -(cx(i)+cx(i+1))/(2*h));
			coefficientsII.emplace_back(edge_offset+i, edge_offset+i, (cx(i)+2*cx(i+1)+cx(i+2))/(2*h)+h*vx(i+1));
			coefficientsII.emplace_back(edge_offset+i, edge_offset+i+1, -(cx(i+1)+cx(i+2))/(2*h));
		}
		coefficientsII.emplace_back(edge_offset+N-3, edge_offset+N-4, -(cx(N-3)+cx(N-2))/(2*h));
		coefficientsII.emplace_back(edge_offset+N-3, edge_offset+N-3, (cx(N-3)+2*cx(N-2)+cx(N-1))/(2*h)+h*vx(N-2));

		lcoeff = -(cx(0)+cx(1))/(2*h);
		rcoeff = -(cx(N-2)+cx(N-1))/(2*h);
		coefficientsIG.emplace_back(edge_offset, out, lcoeff);
		coefficientsIG.emplace_back(edge_offset+N-3, in, rcoeff);
		vertex_stiffness[out] += -lcoeff+vx(0)*h/2;
		vertex_stiffness[in] += -rcoeff+vx(N-1)*h/2;

		bI.middleRows(edge_offset, N-2) = h*fx(Eigen::seq(1, N-2));
		bG(out) += fx(0)*h/2;
		bG(in) += fx(N-1)*h/2;
	}

	for (int i = 0; i < vertex_stiffness.size(); ++i)
		coefficientsGG.emplace_back(i, i, vertex_stiffness[i]);

	AII.resize(sizeI, sizeI);
	AII.reserve(nnzII);
	AII.setFromTriplets(coefficientsII.begin(), coefficientsII.end());
	AIG.resize(sizeI, sizeG);
	AIG.reserve(nnzIG);
	AIG.setFromTriplets(coefficientsIG.begin(), coefficientsIG.end());
	AGG.resize(sizeG, sizeG);
	AGG.reserve(nnzGG);
	AGG.setFromTriplets(coefficientsGG.begin(), coefficientsGG.end());

	mysolver.compute(AII);
	bG -= AIG.transpose()*mysolver.solve(bI);
}
