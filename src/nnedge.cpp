#include "nnedge.h"

NNEdge::NNEdge(int _out, int _in, int _N, Eigen::VectorXd& _cx, Eigen::VectorXd& _vx, Eigen::VectorXd& _fx, Eigen::SparseMatrix<double>& _A, BC _lbc, BC _rbc) : out(_out), in(_in), N(_N), cx(_cx), vx(_vx), fx(_fx), lbc(_lbc), rbc(_rbc) {
	h = 1.0/(N-1);
	Al = { (cx(0)+cx(1))/(2*h)+h/2*vx(0), -(cx(0)+cx(1))/(2*h) };
	Ar = { -(cx(N-2)+cx(N-1))/(2*h), (cx(N-2)+cx(N-1))/(2*h)+h/2*vx(N-1) };

	solver.compute(_A);

	b = h*fx;
	u = Eigen::VectorXd::Zero(N);
}

double NNEdge::Dirichlet(int endpoint) {
	if (endpoint == out)
		return u(0);
	else
		return u(N-1);
}

double NNEdge::approxNeumann(int endpoint) {
	if (endpoint == out)
		return -(Al.dot(u(Eigen::seq(0, 1)))-h/2*fx(0));
	else
		return -(Ar.dot(u(Eigen::seq(N-2,N-1)))-h/2*fx(N-1));
}

void NNEdge::setBC(int endpoint, double value) {
	if (endpoint == out) {
		lbc.value = value;
		if (lbc.type == BCType::Dirichlet)
			b(0) = lbc.value;
		else
			b(0) = h/2*fx(0)-lbc.value;
	}
	else {
		rbc.value = value;
		if (rbc.type == BCType::Dirichlet)
			b(N-1) = rbc.value;
		else
			b(N-1) = h/2*fx(N-1)+rbc.value;
	}
}

void NNEdge::solve() {
	u = solver.solve(b);
}
