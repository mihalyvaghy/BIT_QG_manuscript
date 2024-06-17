#include "qgedge.h"
#include <fstream>
#include <vector>
#include <Eigen/Core>

#pragma once

#define PI 3.14159265358979323846

class Example {
public:
	std::vector<QGEdge> edges;
	int vertices;

	Example() {}
	Example(std::vector<QGEdge> _edges, int _vertices) : edges(_edges), vertices(_vertices) {}
	Example(std::string filename) {
		int nnz = 0;
		int row = 0;
		int col;
		int num;
		std::ifstream f("../graphs/"+filename+".txt");
		std::string line;
		while (getline(f, line)) {
			col = 0;

			std::istringstream iss(line);
			while (iss >> num) {
				if (num) {
					edges.emplace_back(row, col,
						[](double x){return 5+x*x;},
						[](double x){return 0.1;},
						[](double x){return 2+x;});
					++nnz;
				}
				++col;

				if (iss.peek() == ',')
					iss.ignore();
			}
			++row;
		}
		vertices = row;
	}
};

Example square_grid(int size) {
	int vertices = size*size;
	std::vector<QGEdge> edges;
	edges.reserve(2*size*(size-1));
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			if (j != size-1) {
				edges.emplace_back(i*size+j, i*size+(j+1),
											 [](double x){return x*x;},
											 [](double x){return sin(x);},
											 [](double x){return exp(x);});
			}
			if (i != size-1) {
				edges.emplace_back(i*size+j, (i+1)*size+j,
											 [](double x){return x*x;},
											 [](double x){return sin(x);},
											 [](double x){return exp(x);});
			}
		}
	}
	return {edges, vertices};
}
