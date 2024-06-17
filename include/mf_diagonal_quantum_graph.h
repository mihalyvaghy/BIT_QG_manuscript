#include "qgedge.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#pragma once

class MFDiagonalQuantumGraph;

template<>
struct Eigen::internal::traits<MFDiagonalQuantumGraph> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

class MFDiagonalQuantumGraph : public Eigen::EigenBase<MFDiagonalQuantumGraph> {
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
  Eigen::Product<MFDiagonalQuantumGraph, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MFDiagonalQuantumGraph, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // my API
  int size;
  Eigen::SparseMatrix<double> AII, AIG, AGG;
	Eigen::SparseMatrix<double> Dinv;
  Eigen::VectorXd bI, bG;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mysolver;

  MFDiagonalQuantumGraph(int N, int vertices, const std::vector<QGEdge>& edges);

  template<typename Rhs>
  const Rhs solve(const Rhs& rhs) const {
    return Dinv*(AGG*rhs-AIG.transpose()*mysolver.solve(AIG*rhs));
  }
};

namespace Eigen {
namespace internal {
template<typename Rhs>
struct generic_product_impl<MFDiagonalQuantumGraph, Rhs, SparseShape, DenseShape, GemvProduct>
: generic_product_impl_base<MFDiagonalQuantumGraph, Rhs, generic_product_impl<MFDiagonalQuantumGraph, Rhs> >
{
	typedef typename Product<MFDiagonalQuantumGraph,Rhs>::Scalar Scalar;

	template<typename Dest>
	static void scaleAndAddTo(Dest& dst, const MFDiagonalQuantumGraph& lhs, const Rhs& rhs, const Scalar& alpha)
	{
		dst.noalias() += alpha*lhs.solve(rhs);
	}
};
}
}
