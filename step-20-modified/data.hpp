#ifndef DATA_HPP_
#define DATA_HPP_

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace PrescribedSolution
{
  using namespace dealii;

  constexpr double alpha = 0.3;
  constexpr double beta  = 1;

  /*
   * Right hand side
   */

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(1)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };


  template <int dim>
  double
  RightHandSide<dim>::value(const Point<dim> & /*p*/,
                            const unsigned int /*component*/) const
  {
    return 0;
  }


  /*
   * Pressure solution
   */

  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues()
      : Function<dim>(1)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  template <>
  double
  PressureBoundaryValues<2>::value(const Point<2> &p,
                                   const unsigned int /*component*/) const
  {
    return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
             alpha * p[0] * p[0] * p[0] / 6);
  }

  template <>
  double
  PressureBoundaryValues<3>::value(const Point<3> &p,
                                   const unsigned int /*component*/) const
  {
    return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
             alpha * p[0] * p[0] * p[0] / 6 + alpha * p[2]);
  }


  /*
   * Full exact solution
   */

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim + 1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;
  };


  template <>
  void
  ExactSolution<2>::vector_value(const Point<2> &p,
                                 Vector<double> &values) const
  {
    Assert(values.size() == 2 + 1, ExcDimensionMismatch(values.size(), 2 + 1));

    values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    values(1) = alpha * p[0] * p[1];
    values(2) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                  alpha * p[0] * p[0] * p[0] / 6);
  }


  template <>
  void
  ExactSolution<3>::vector_value(const Point<3> &p,
                                 Vector<double> &values) const
  {
    Assert(values.size() == 3 + 1, ExcDimensionMismatch(values.size(), 3 + 1));

    values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    values(1) = alpha * p[0] * p[1];
    values(2) = -alpha;
    values(3) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                  alpha * p[0] * p[0] * p[0] / 6 + alpha * p[2]);
  }


  /*
   * Solution for Velocity only
   */

  template <int dim>
  class ExactSolutionVelocity : public Function<dim>
  {
  public:
    ExactSolutionVelocity()
      : Function<dim>(dim)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

    void
    tensor_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Tensor<1, dim>> &  values) const;
  };


  template <>
  void
  ExactSolutionVelocity<2>::vector_value(const Point<2> &p,
                                         Vector<double> &values) const
  {
    Assert(values.size() == 2, ExcDimensionMismatch(values.size(), 2));

    values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    values(1) = alpha * p[0] * p[1];
  }


  template <>
  void
  ExactSolutionVelocity<2>::tensor_value_list(
    const std::vector<Point<2>> &points,
    std::vector<Tensor<1, 2>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        values[i][0] = alpha * points[i][1] * points[i][1] / 2 + beta -
                       alpha * points[i][0] * points[i][0] / 2;
        values[i][1] = alpha * points[i][0] * points[i][1];
      }
  }


  template <>
  void
  ExactSolutionVelocity<3>::vector_value(const Point<3> &p,
                                         Vector<double> &values) const
  {
    Assert(values.size() == 3, ExcDimensionMismatch(values.size(), 3));

    values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
    values(1) = alpha * p[0] * p[1];
    values(2) = -alpha;
  }


  template <>
  void
  ExactSolutionVelocity<3>::tensor_value_list(
    const std::vector<Point<3>> &points,
    std::vector<Tensor<1, 3>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        values[i].clear();

        values[i][0] = alpha * points[i][1] * points[i][1] / 2 + beta -
                       alpha * points[i][0] * points[i][0] / 2;
        values[i][1] = alpha * points[i][0] * points[i][1];
        values[i][2] = -alpha;
      }
  }


  /*
   * Coefficient
   */
  template <int dim>
  class KInverse : public TensorFunction<2, dim>
  {
  public:
    KInverse()
      : TensorFunction<2, dim>()
    {}

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<2, dim>> &  values) const override;
  };


  template <int dim>
  void
  KInverse<dim>::value_list(const std::vector<Point<dim>> &points,
                            std::vector<Tensor<2, dim>> &  values) const
  {
    (void)points;
    AssertDimension(points.size(), values.size());

    for (auto &value : values)
      value = unit_symmetric_tensor<dim>();
  }
} // namespace PrescribedSolution


#endif /* DATA_HPP_ */
