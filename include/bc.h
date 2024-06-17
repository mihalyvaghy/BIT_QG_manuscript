#pragma once

enum BCType { Dirichlet, Neumann };

struct BC {
	BCType type;
	double value;
};
