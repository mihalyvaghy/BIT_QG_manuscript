#include "qgedge.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#pragma once

class MFPolynomialQuantumGraph;

template<>
struct Eigen::internal::traits<MFPolynomialQuantumGraph> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

class MFPolynomialQuantumGraph : public Eigen::EigenBase<MFPolynomialQuantumGraph> {
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

  Eigen::Index rows() const { return size; }
  Eigen::Index cols() const { return size; }

  template<typename Rhs>
  Eigen::Product<MFPolynomialQuantumGraph, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MFPolynomialQuantumGraph, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // my API
  int size;
  Eigen::SparseMatrix<double> AII, AIG, AGG;
  Eigen::SparseMatrix<double> D, Dinv;
  Eigen::VectorXd bI, bG;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mysolver;

  MFPolynomialQuantumGraph(int N, int vertices, const std::vector<QGEdge>& edges);

  template<typename Rhs>
  const Rhs solve(const Rhs& rhs) const {
    Rhs Srhs = AGG*rhs-AIG.transpose()*mysolver.solve(AIG*rhs);
    return (2*Dinv*Srhs-Dinv*(AGG*Dinv*Srhs-AIG.transpose()*mysolver.solve(AIG*Dinv*Srhs)));
  }
};

namespace Eigen {
namespace internal {
template<typename Rhs>
struct generic_product_impl<MFPolynomialQuantumGraph, Rhs, SparseShape, DenseShape, GemvProduct>
: generic_product_impl_base<MFPolynomialQuantumGraph, Rhs, generic_product_impl<MFPolynomialQuantumGraph, Rhs> >
{
  typedef typename Product<MFPolynomialQuantumGraph,Rhs>::Scalar Scalar;

  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const MFPolynomialQuantumGraph& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    dst.noalias() += alpha*lhs.solve(rhs);
  }
};
}
}
