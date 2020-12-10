#ifndef SHAPE_FUN_VECTOR_HPP_
#define SHAPE_FUN_VECTOR_HPP_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  class ShapeFunctionVector : public Function<dim>
  {
  public:
    ShapeFunctionVector(const FiniteElement<dim> &                  fe,
                        typename Triangulation<dim>::cell_iterator &cell,
                        bool verbose = false);

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  values) const override;

    void
    tensor_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Tensor<1, dim>> &  values) const;

    void
    set_current_cell(const typename Triangulation<dim>::cell_iterator &cell);

    void
    set_shape_fun_index(unsigned int index);

  private:
    SmartPointer<const FiniteElement<dim>> fe_ptr;
    const unsigned int                     dofs_per_cell;
    unsigned int                           shape_fun_index;

    const MappingQ<dim> mapping;

    typename Triangulation<dim>::cell_iterator *current_cell_ptr;

    const bool verbose;

    double orientation_corrector;

    unsigned int n_face_dofs;
  };

  template <int dim>
  ShapeFunctionVector<dim>::ShapeFunctionVector(
    const FiniteElement<dim> &                  fe,
    typename Triangulation<dim>::cell_iterator &cell,
    bool                                        verbose)
    : Function<dim>(dim)
    , fe_ptr(&fe)
    , dofs_per_cell(fe_ptr->dofs_per_cell)
    , shape_fun_index(0)
    , mapping(1)
    , current_cell_ptr(&cell)
    , verbose(verbose)
    , orientation_corrector((cell->face_orientation(0) ? 1.0 : -1.0))
  {
    // If element is primitive it is invalid.
    // Also there must not be more than one block.
    // This excludes FE_Systems.
    Assert((!fe_ptr->is_primitive()), FETools::ExcInvalidFE());
    Assert(fe_ptr->n_blocks() == 1,
           ExcDimensionMismatch(1, fe_ptr->n_blocks()));

    n_face_dofs = 0;

    for (unsigned int face_index = 0;
         face_index < GeometryInfo<dim>::faces_per_cell;
         ++face_index)
      {
        n_face_dofs += fe_ptr->n_dofs_per_face(face_index);
      }

    //  if (verbose) {
    //    std::cout << "\n		Constructed vector shape function for   "
    //              << fe_ptr->get_name() << "   on cell   [";
    //    for (unsigned int i = 0; i < (std::pow(2, dim) - 1); ++i) {
    //      std::cout << cell->vertex(i) << ", \n";
    //    }
    //    std::cout << cell->vertex(std::pow(2, dim) - 1) << "]\n" <<
    //    std::endl;
    //  }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_current_cell(
    const typename Triangulation<dim>::cell_iterator &cell)
  {
    current_cell_ptr = &cell;
    if (shape_fun_index < n_face_dofs)
      {
        /*
         * This is integer division
         */
        unsigned int face_index_from_shape_index =
          shape_fun_index / (fe_ptr->n_dofs_per_face(0));

        orientation_corrector =
          (*current_cell_ptr)->face_orientation(face_index_from_shape_index) ?
            1.0 :
            -1.0;
      }
    else
      {
        orientation_corrector = 1.0;
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_shape_fun_index(unsigned int index)
  {
    shape_fun_index = index;
    if (shape_fun_index < n_face_dofs)
      {
        /*
         * This is integer division
         */
        unsigned int face_index_from_shape_index =
          shape_fun_index / (fe_ptr->n_dofs_per_face(0));

        orientation_corrector =
          (*current_cell_ptr)->face_orientation(face_index_from_shape_index) ?
            1.0 :
            -1.0;
      }
    else
      {
        orientation_corrector = 1.0;
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &  value) const
  {
    // Map physical points to reference cell
    Point<dim> point_on_ref_cell(
      mapping.transform_real_to_unit_cell(*current_cell_ptr, p));

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(point_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(mapping,
                            *fe_ptr,
                            fake_quadrature,
                            update_values | update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < dim; ++i)
      {
        value[i] = orientation_corrector *
                   fe_values.shape_value_component(shape_fun_index,
                                                   /* q_index */ 0,
                                                   i);
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell.at(i) =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(mapping,
                            *fe_ptr,
                            fake_quadrature,
                            update_values | update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        for (unsigned int component = 0; component < dim; ++component)
          {
            values.at(i)[component] =
              orientation_corrector *
              fe_values.shape_value_component(shape_fun_index,
                                              /* q_index */ i,
                                              component);
          }
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::tensor_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell.at(i) =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        for (unsigned int component = 0; component < dim; ++component)
          {
            values.at(i)[component] =
              fe_values.shape_value_component(shape_fun_index,
                                              /* q_index */ i,
                                              component);
          }
      }
  }

} // namespace ShapeFun

#endif /* SHAPE_FUN_VECTOR_HPP_ */
