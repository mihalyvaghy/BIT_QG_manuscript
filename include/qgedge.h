#pragma once

class QGEdge {
public:
	int out;
	int in;
	double (*c)(double);
	double (*v)(double);
	double (*f)(double);
	
	QGEdge(int _out, int _in, double (*_c)(double), double (*_v)(double), double (*_f)(double)) : out(_out), in(_in), c(_c), v(_v), f(_f) {};
};
