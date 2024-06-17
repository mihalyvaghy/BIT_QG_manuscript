#include "qgedge.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#pragma once

class MFQuantumGraph;

template<>
struct Eigen::internal::traits<MFQuantumGraph> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

class MFQuantumGraph : public Eigen::EigenBase<MFQuantumGraph> {
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
  Eigen::Product<MFQuantumGraph, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MFQuantumGraph, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // my API
  int size;
  Eigen::SparseMatrix<double> AII, AIG, AGG;
  Eigen::VectorXd bI, bG;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mysolver;

  MFQuantumGraph(int N, int vertices, const std::vector<QGEdge>& edges);

  template<typename Rhs>
  const Rhs solve(const Rhs& rhs) const {
    return AGG*rhs-AIG.transpose()*mysolver.solve(AIG*rhs);
  }
};

namespace Eigen {
namespace internal {
template<typename Rhs>
struct generic_product_impl<MFQuantumGraph, Rhs, SparseShape, DenseShape, GemvProduct>
: generic_product_impl_base<MFQuantumGraph, Rhs, generic_product_impl<MFQuantumGraph, Rhs> >
{
	typedef typename Product<MFQuantumGraph,Rhs>::Scalar Scalar;

	template<typename Dest>
	static void scaleAndAddTo(Dest& dst, const MFQuantumGraph& lhs, const Rhs& rhs, const Scalar& alpha)
	{
		dst.noalias() += alpha*lhs.solve(rhs);
	}
};
}
}
