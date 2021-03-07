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
                        bool verbose               = false,
                        bool adjust_index_and_sign = true);

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
    /*
     * If we have a shape function on a face that is flipped then also the
     * enumeration of dofs on that face is flipped. This function checks if a
     * dof_index is an index of a shape function on a flipped face. In this case
     * it maps this index to a corrected index such that a conformity condition
     * across faces is met. maps a "standard order"
     */
    std::pair<unsigned int, bool>
    adjust_dof_index_and_sign_on_face_rt(
      typename Triangulation<dim>::cell_iterator &cell,
      const unsigned int                          shape_fun_index,
      const unsigned int                          order);

    std::pair<unsigned int, bool>
    adjust_dof_index_and_sign_on_face_bdm(
      typename Triangulation<dim>::cell_iterator &cell,
      const unsigned int                          shape_fun_index,
      const unsigned int                          order);

    std::function<std::pair<unsigned int, bool>(
      typename Triangulation<dim>::cell_iterator &,
      const unsigned int,
      const unsigned int)>
      adjust_dof_index_and_sign_on_face;

    SmartPointer<const FiniteElement<dim>> fe_ptr;
    const unsigned int                     dofs_per_cell;
    unsigned int                           shape_fun_index;
    double                                 shape_fun_sign;

    const MappingQ<dim> mapping;

    typename Triangulation<dim>::cell_iterator *current_cell_ptr;

    const bool verbose;

    unsigned int n_face_dofs;

    const bool adjust_index_and_sign;
  };

  template <int dim>
  ShapeFunctionVector<dim>::ShapeFunctionVector(
    const FiniteElement<dim> &                  fe,
    typename Triangulation<dim>::cell_iterator &cell,
    bool                                        verbose,
    bool                                        adjust_index_and_sign)
    : Function<dim>(dim)
    , fe_ptr(&fe)
    , dofs_per_cell(fe_ptr->dofs_per_cell)
    , shape_fun_index(0)
    , shape_fun_sign(1.0)
    , mapping(1)
    , current_cell_ptr(&cell)
    , verbose(verbose)
    , adjust_index_and_sign(adjust_index_and_sign)
  {
    // If element is primitive it is invalid.
    // Also there must not be more than one block.
    // This excludes FE_Systems.
    Assert((!fe_ptr->is_primitive()), FETools::ExcInvalidFE());
    Assert(fe_ptr->n_blocks() == 1,
           ExcDimensionMismatch(1, fe_ptr->n_blocks()));

    n_face_dofs = GeometryInfo<dim>::faces_per_cell * fe_ptr->n_dofs_per_face();

    std::string fe_rt_str(
      "FE_RaviartThomas<" + Utilities::int_to_string(dim, 1) + ">(" +
      Utilities::int_to_string((*fe_ptr).degree - 1, 1) + ")");

    std::string fe_bdm_str("FE_BDM<" + Utilities::int_to_string(dim, 1) + ">(" +
                           Utilities::int_to_string((*fe_ptr).degree - 1, 1) +
                           ")");

    if (fe_rt_str.compare((*fe_ptr).get_name()) == 0)
      {
        adjust_dof_index_and_sign_on_face = std::bind(
          &ShapeFunctionVector<dim>::adjust_dof_index_and_sign_on_face_rt,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3);
      }
    else if (fe_bdm_str.compare((*fe_ptr).get_name()) == 0)
      {
        adjust_dof_index_and_sign_on_face = std::bind(
          &ShapeFunctionVector<dim>::adjust_dof_index_and_sign_on_face_bdm,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3);
      }
    else
      {
        std::cerr << "Not implemented..." << std::endl;
        exit(1);
      }

    if (adjust_index_and_sign)
      {
        const auto shape_fun_index_and_sign =
          adjust_dof_index_and_sign_on_face(cell, 0, (*fe_ptr).degree);

        shape_fun_index = shape_fun_index_and_sign.first;
        shape_fun_sign  = (shape_fun_index_and_sign.second ? -1.0 : 1.0);
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_current_cell(
    const typename Triangulation<dim>::cell_iterator &cell)
  {
    current_cell_ptr = &cell;
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_shape_fun_index(unsigned int index)
  {
    if (adjust_index_and_sign)
      {
        const auto shape_fun_index_and_sign =
          adjust_dof_index_and_sign_on_face(*current_cell_ptr,
                                            index,
                                            (*fe_ptr).degree);

        shape_fun_index = shape_fun_index_and_sign.first;
        shape_fun_sign  = (shape_fun_index_and_sign.second ? -1.0 : 1.0);
      }
    else
      {
        shape_fun_index = index;
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
        value[i] =
          shape_fun_sign * fe_values.shape_value_component(shape_fun_index,
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
              shape_fun_sign * fe_values.shape_value_component(shape_fun_index,
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
              shape_fun_sign * fe_values.shape_value_component(shape_fun_index,
                                                               /* q_index */ i,
                                                               component);
          }
      }
  }


  template <int dim>
  std::pair<unsigned int, bool>
  ShapeFunctionVector<dim>::adjust_dof_index_and_sign_on_face_rt(
    typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                          dof_index,
    const unsigned int                          degree)
  {
    unsigned int new_dof_index = dof_index;
    bool         sign_flip     = false;

    const unsigned int n_dofs_per_face = (*fe_ptr).n_dofs_per_face();

    const unsigned int n_face_dofs =
      GeometryInfo<dim>::faces_per_cell * n_dofs_per_face;

    /*
     * Assume that all face dofs come before volume dofs.
     */
    if (dof_index < n_face_dofs)
      {
        /*
         * Find the face belonging to this dof. This is integer division.
         */
        unsigned int face_index_from_shape_index =
          dof_index / (n_dofs_per_face);

        const unsigned int n = degree;

        /*
         * If face does not have standard orientation permute dofs
         */
        if ((!cell->face_orientation(face_index_from_shape_index)) &&
            (!cell->face_rotation(face_index_from_shape_index)))
          {
            unsigned int local_face_dof = dof_index % n_dofs_per_face;
            // Row and column
            unsigned int i = local_face_dof % n, j = local_face_dof / n;

            // We flip across the diagonal
            unsigned int offset = j + i * n - local_face_dof;

            new_dof_index = dof_index + offset;
          } // if face needs dof permutation
        else if ((cell->face_orientation(face_index_from_shape_index)) &&
                 (cell->face_rotation(face_index_from_shape_index)))
          {
            unsigned int local_face_dof = dof_index % n_dofs_per_face;
            // Row and column
            unsigned int i = local_face_dof % n, j = local_face_dof / n;

            // We flip across the diagonal
            unsigned int offset = j + i * n - local_face_dof;
            // unsigned int offset = j + (n - 1 - i) * n - local_face_dof;
            // unsigned int offset = (n - 1 - j) + i * n - local_face_dof;

            new_dof_index = dof_index + offset;
          } // if face needs dof permutation

        if (false)
          {
            /*
             * To determine if a (corrected) sign flip is necessary we need the
             new
             * coordinates of the flipped index
             */
            unsigned int local_face_dof = new_dof_index % n_dofs_per_face;
            // Row and column
            const unsigned int i = local_face_dof % n;
            const unsigned int j = local_face_dof / n;

            /*
             * Maybe switch the sign
             */
            // flip = false, rotation=true
            if (!cell->face_flip(face_index_from_shape_index) &&
                cell->face_rotation(face_index_from_shape_index))
              {
                // Row and column may be switched
                if (cell->face_orientation(face_index_from_shape_index))
                  sign_flip = ((i % 2) == 1);
                else
                  sign_flip = ((j % 2) == 1);
              }
            // flip = true, rotation=false
            else if (cell->face_flip(face_index_from_shape_index) &&
                     !cell->face_rotation(face_index_from_shape_index))
              {
                // This case is symmetric (although row and column may be
                // switched)
                sign_flip = ((j % 2) == 1) != ((i % 2) == 1);
              }
            // flip = true, rotation=true
            else if (cell->face_flip(face_index_from_shape_index) &&
                     cell->face_rotation(face_index_from_shape_index))
              {
                // Row and column may be switched
                if (cell->face_orientation(face_index_from_shape_index))
                  sign_flip = ((j % 2) == 1);
                else
                  sign_flip = ((i % 2) == 1);
              }
            // flip = false, rotation=false => nothing to do

            /*
             * If we are on a face that does not have standard orientation we
             must
             * flip all signs again
             */
            if (!cell->face_orientation(face_index_from_shape_index))
              sign_flip = !sign_flip;
          }

      } // if dof_index < n_face_dofs

    std::pair<unsigned int, bool> new_dof_index_and_sign_flip(new_dof_index,
                                                              sign_flip);

    return new_dof_index_and_sign_flip;
  }


  template <int dim>
  std::pair<unsigned int, bool>
  ShapeFunctionVector<dim>::adjust_dof_index_and_sign_on_face_bdm(
    typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                          dof_index,
    const unsigned int                          degree)
  {
    unsigned int new_dof_index = dof_index;
    bool         sign_flip     = false;

    const unsigned int n_dofs_per_face = (*fe_ptr).n_dofs_per_face();

    const unsigned int n_face_dofs =
      GeometryInfo<dim>::faces_per_cell * n_dofs_per_face;


    // /*
    //  * Assume that all face dofs come before volume dofs.
    //  */
    // if (dof_index < n_face_dofs)
    //   {
    /*
     * Find the face belonging to this dof. This is integer division.
     */
    // unsigned int face_index_from_shape_index = dof_index / (n_dofs_per_face);

    // const unsigned int n = degree;

    //   // If false then permutations are not corrected
    //   if (false)
    //     {
    //       /*
    //        * If face does not have standard orientation permute dofs
    //        */
    //       if (((!cell->face_orientation(face_index_from_shape_index)) &&
    //            (!cell->face_rotation(face_index_from_shape_index))) ||
    //           ((cell->face_orientation(face_index_from_shape_index)) &&
    //            (cell->face_rotation(face_index_from_shape_index))))
    //         {
    //           unsigned int local_face_dof = dof_index % n_dofs_per_face;
    //           // Row and column
    //           unsigned int i = local_face_dof % n, j = local_face_dof /
    //           n;

    //           // We flip across the diagonal
    //           unsigned int offset = j + i * n - local_face_dof;

    //           new_dof_index = dof_index + offset;
    //         } // if face needs dof permutation
    //     }

    //   // If false then signs are not corrected
    //   if (false)
    //     {
    //       /*
    //        * To determine if a (corrected) sign flip is necessary we need
    //        the new
    //        * coordinates of the flipped index
    //        */
    //       unsigned int local_face_dof = new_dof_index % n_dofs_per_face;
    //       // Row and column
    //       const unsigned int i = local_face_dof % n;
    //       const unsigned int j = local_face_dof / n;

    //       /*
    //        * Maybe switch the sign
    //        */
    //       // flip = false, rotation=true
    //       if (!cell->face_flip(face_index_from_shape_index) &&
    //           cell->face_rotation(face_index_from_shape_index))
    //         {
    //           // Row and column may be switched
    //           if (cell->face_orientation(face_index_from_shape_index))
    //             sign_flip = ((i % 2) == 1);
    //           else
    //             sign_flip = ((j % 2) == 1);
    //         }
    //       // flip = true, rotation=false
    //       else if (cell->face_flip(face_index_from_shape_index) &&
    //                !cell->face_rotation(face_index_from_shape_index))
    //         {
    //           // This case is symmetric (although row and column may be
    //           // switched)
    //           sign_flip = ((j % 2) == 1) != ((i % 2) == 1);
    //         }
    //       // flip = true, rotation=true
    //       else if (cell->face_flip(face_index_from_shape_index) &&
    //                cell->face_rotation(face_index_from_shape_index))
    //         {
    //           // Row and column may be switched
    //           if (cell->face_orientation(face_index_from_shape_index))
    //             sign_flip = ((j % 2) == 1);
    //           else
    //             sign_flip = ((i % 2) == 1);
    //         }
    //       // flip = false, rotation=false => nothing to do

    //       /*
    //        * If we are on a face that does not have standard orientation
    //        we must
    //        * flip all signs again
    //        */
    //       if (!cell->face_orientation(face_index_from_shape_index))
    //         sign_flip = !sign_flip;
    //     }

    // } // if dof_index < n_face_dofs

    /////////////////////////////////////////////////////////

    /*
     * Assume that all face dofs come before volume dofs.
     */
    if (dof_index < n_face_dofs)
      {
        /*
         * Find the face belonging to this dof. This is integer division.
         */
        unsigned int face_index_from_shape_index =
          dof_index / (n_dofs_per_face);

        /*
         * If face does not have standard orientation permute dofs
         */
        if (((!cell->face_orientation(face_index_from_shape_index)) &&
             (!cell->face_rotation(face_index_from_shape_index))) ||
            ((cell->face_orientation(face_index_from_shape_index)) &&
             (cell->face_rotation(face_index_from_shape_index))))
          {
            if (degree == 2)
              {
                if (dof_index % n_dofs_per_face == 0)
                  {
                    new_dof_index = dof_index;
                  }
                else if (dof_index % n_dofs_per_face == 1)
                  {
                    new_dof_index = dof_index + 1;
                  }
                else if (dof_index % n_dofs_per_face == 2)
                  {
                    new_dof_index = dof_index - 1;
                  }
              } // degree == 2
            else if (degree == 3)
              {
                if (dof_index % n_dofs_per_face == 0)
                  {
                    new_dof_index = dof_index;
                  }
                else if (dof_index % n_dofs_per_face == 1)
                  {
                    new_dof_index = dof_index + 1;
                  }
                else if (dof_index % n_dofs_per_face == 2)
                  {
                    new_dof_index = dof_index - 1;
                  }
                else if (dof_index % n_dofs_per_face == 3)
                  {
                    new_dof_index = dof_index;
                  }
                else if (dof_index % n_dofs_per_face == 4)
                  {
                    new_dof_index = dof_index + 1;
                  }
                else if (dof_index % n_dofs_per_face == 5)
                  {
                    new_dof_index = dof_index - 1;
                  }
              } // degree == 3
          }     // if face flipped
      }         // if dof_index < n_face_dofs

    std::pair<unsigned int, bool> new_dof_index_and_sign_flip(new_dof_index,
                                                              sign_flip);

    return new_dof_index_and_sign_flip;
  }

} // namespace ShapeFun

#endif /* SHAPE_FUN_VECTOR_HPP_ */
