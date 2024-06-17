#include "qgedge.h"
#include <unordered_set>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#pragma once

class MFNeumannNeumann;

template<>
struct Eigen::internal::traits<MFNeumannNeumann> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

class MFNeumannNeumann : public Eigen::EigenBase<MFNeumannNeumann> {
public:
  // must have API
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Eigen::Index rows() const { return vertices; }
  Eigen::Index cols() const { return vertices; }

  template<typename Rhs>
  Eigen::Product<MFNeumannNeumann, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MFNeumannNeumann, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // my API
  int vertices;
  int N;
  std::vector<QGEdge> edges;
  std::vector<Eigen::SparseMatrix<double>*> AIGs;
  std::vector<Eigen::SparseMatrix<double>*> AGGs;
  std::vector<Eigen::VectorXd*> bIs;
  std::vector<Eigen::VectorXd*> bGs;
  std::vector<Eigen::SparseLU<Eigen::SparseMatrix<double>>> dirichlet_solvers;
  std::vector<Eigen::SparseLU<Eigen::SparseMatrix<double>>> neumann_solvers;
  Eigen::VectorXd bG;
  std::vector<std::unordered_set<int>> vertex_to_edges;
  std::vector<int> vertex_weights;
  std::unordered_set<int> boundary;
  Eigen::VectorXd vertex_values;

  MFNeumannNeumann(int _N, int _vertices, std::vector<QGEdge>& _edges);
  Eigen::VectorXd solve();

  template<typename Rhs>
  const Rhs solve(const Rhs& rhs) const {
    Rhs neumann_res = Rhs::Zero(vertices);
    Rhs dirichlet_res = Rhs::Zero(vertices);
    Rhs tmp = Rhs::Zero(vertices);

    for (int i = 0; i < dirichlet_solvers.size(); ++i)
      dirichlet_res += (*AGGs[i])*rhs-AIGs[i]->transpose()*dirichlet_solvers[i].solve((*AIGs[i])*rhs);

    Eigen::VectorXd neumann_fx = Eigen::VectorXd::Zero(N);
    int out, in;
    for (int i = 0; i < neumann_solvers.size(); ++i) {
      out = edges[i].out;
      in = edges[i].in;
      neumann_fx(N-2) = dirichlet_res(out)/vertex_weights[out];
      neumann_fx(N-1) = dirichlet_res(in)/vertex_weights[in];
      tmp = neumann_solvers[i].solve(neumann_fx);
      neumann_res(out) += tmp(N-2)/vertex_weights[out];
      neumann_res(in) += tmp(N-1)/vertex_weights[in];
    }

    return neumann_res;
  }

  ~MFNeumannNeumann();
};

namespace Eigen {
namespace internal {
template<typename Rhs>
struct generic_product_impl<MFNeumannNeumann, Rhs, SparseShape, DenseShape, GemvProduct>
: generic_product_impl_base<MFNeumannNeumann, Rhs, generic_product_impl<MFNeumannNeumann, Rhs> >
{
  typedef typename Product<MFNeumannNeumann,Rhs>::Scalar Scalar;

  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MFNeumannNeumann& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    dst.noalias() += alpha*lhs.solve(rhs);
  }
};
}
}
