#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <shape_fun_scalar.hpp>
#include <shape_fun_vector.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace Step20
{
  using namespace dealii;

  void
  generate_test_mesh(Triangulation<3> &triangulation,
                     const bool        face_orientation,
                     const bool        face_flip,
                     const bool        face_rotation,
                     const bool        manipulate_first_cube)
  {
    std::vector<Point<3>>    vertices;
    const unsigned int       n_cells = 2;
    std::vector<CellData<3>> cells(n_cells);

    const Point<3> p(1, 0, 0);

    static const std::array<Point<3>, 12> double_cube = {{{0, 0, 0},   // 0
                                                          {1, 0, 0},   // 1
                                                          {0, 1, 0},   // 2
                                                          {1, 1, 0},   // 3
                                                          {0, 0, 1},   // 4
                                                          {1, 0, 1},   // 5
                                                          {0, 1, 1},   // 6
                                                          {1, 1, 1},   // 7
                                                          {2, 0, 0},   // 8
                                                          {2, 1, 0},   // 9
                                                          {2, 0, 1},   // 10
                                                          {2, 1, 1}}}; // 11

    for (unsigned int i = 0; i < 12; ++i)
      vertices.push_back(double_cube[i]);

    int cell_vertices[n_cells][8] = {
      {0, 1, 2, 3, 4, 5, 6, 7},    // unit cube
      {1, 8, 3, 9, 5, 10, 7, 11}}; // shifted cube

    // binary to case number
    int this_case = 4 * face_orientation + 2 * face_flip + face_rotation;

    if (manipulate_first_cube)
      {
        switch (this_case)
          {
              case 0: {
                cell_vertices[0][0] = 1;
                cell_vertices[0][1] = 0;
                cell_vertices[0][2] = 5;
                cell_vertices[0][3] = 4;
                cell_vertices[0][4] = 3;
                cell_vertices[0][5] = 2;
                cell_vertices[0][6] = 7;
                cell_vertices[0][7] = 6;
                break;
              }

              case 1: {
                cell_vertices[0][0] = 5;
                cell_vertices[0][1] = 4;
                cell_vertices[0][2] = 7;
                cell_vertices[0][3] = 6;
                cell_vertices[0][4] = 1;
                cell_vertices[0][5] = 0;
                cell_vertices[0][6] = 3;
                cell_vertices[0][7] = 2;
                break;
              }

              case 2: {
                cell_vertices[0][0] = 7;
                cell_vertices[0][1] = 6;
                cell_vertices[0][2] = 3;
                cell_vertices[0][3] = 2;
                cell_vertices[0][4] = 5;
                cell_vertices[0][5] = 4;
                cell_vertices[0][6] = 1;
                cell_vertices[0][7] = 0;
                break;
              }
              case 3: {
                cell_vertices[0][0] = 3;
                cell_vertices[0][1] = 2;
                cell_vertices[0][2] = 1;
                cell_vertices[0][3] = 0;
                cell_vertices[0][4] = 7;
                cell_vertices[0][5] = 6;
                cell_vertices[0][6] = 5;
                cell_vertices[0][7] = 4;
                break;
              }

              case 4: {
                cell_vertices[0][0] = 0;
                cell_vertices[0][1] = 1;
                cell_vertices[0][2] = 2;
                cell_vertices[0][3] = 3;
                cell_vertices[0][4] = 4;
                cell_vertices[0][5] = 5;
                cell_vertices[0][6] = 6;
                cell_vertices[0][7] = 7;
                break;
              }

              case 5: {
                cell_vertices[0][0] = 2;
                cell_vertices[0][1] = 3;
                cell_vertices[0][2] = 6;
                cell_vertices[0][3] = 7;
                cell_vertices[0][4] = 0;
                cell_vertices[0][5] = 1;
                cell_vertices[0][6] = 4;
                cell_vertices[0][7] = 5;
                break;
              }

              case 6: {
                cell_vertices[0][0] = 6;
                cell_vertices[0][1] = 7;
                cell_vertices[0][2] = 4;
                cell_vertices[0][3] = 5;
                cell_vertices[0][4] = 2;
                cell_vertices[0][5] = 3;
                cell_vertices[0][6] = 0;
                cell_vertices[0][7] = 1;
                break;
              }

              case 7: {
                cell_vertices[0][0] = 4;
                cell_vertices[0][1] = 5;
                cell_vertices[0][2] = 0;
                cell_vertices[0][3] = 1;
                cell_vertices[0][4] = 6;
                cell_vertices[0][5] = 7;
                cell_vertices[0][6] = 2;
                cell_vertices[0][7] = 3;
                break;
              }
          } // switch
      }
    else
      {
        switch (this_case)
          {
              case 0: {
                cell_vertices[1][0] = 8;
                cell_vertices[1][1] = 1;
                cell_vertices[1][2] = 10;
                cell_vertices[1][3] = 5;
                cell_vertices[1][4] = 9;
                cell_vertices[1][5] = 3;
                cell_vertices[1][6] = 11;
                cell_vertices[1][7] = 7;
                break;
              }

              case 1: {
                cell_vertices[1][0] = 10;
                cell_vertices[1][1] = 5;
                cell_vertices[1][2] = 11;
                cell_vertices[1][3] = 7;
                cell_vertices[1][4] = 8;
                cell_vertices[1][5] = 1;
                cell_vertices[1][6] = 9;
                cell_vertices[1][7] = 3;
                break;
              }

              case 2: {
                cell_vertices[1][0] = 11;
                cell_vertices[1][1] = 7;
                cell_vertices[1][2] = 9;
                cell_vertices[1][3] = 3;
                cell_vertices[1][4] = 10;
                cell_vertices[1][5] = 5;
                cell_vertices[1][6] = 8;
                cell_vertices[1][7] = 1;
                break;
              }

              case 3: {
                cell_vertices[1][0] = 9;
                cell_vertices[1][1] = 3;
                cell_vertices[1][2] = 8;
                cell_vertices[1][3] = 1;
                cell_vertices[1][4] = 11;
                cell_vertices[1][5] = 7;
                cell_vertices[1][6] = 10;
                cell_vertices[1][7] = 5;
                break;
              }

              case 4: {
                cell_vertices[1][0] = 1;
                cell_vertices[1][1] = 8;
                cell_vertices[1][2] = 3;
                cell_vertices[1][3] = 9;
                cell_vertices[1][4] = 5;
                cell_vertices[1][5] = 10;
                cell_vertices[1][6] = 7;
                cell_vertices[1][7] = 11;
                break;
              }

              case 5: {
                cell_vertices[1][0] = 5;
                cell_vertices[1][1] = 10;
                cell_vertices[1][2] = 1;
                cell_vertices[1][3] = 8;
                cell_vertices[1][4] = 7;
                cell_vertices[1][5] = 11;
                cell_vertices[1][6] = 3;
                cell_vertices[1][7] = 9;
                break;
              }

              case 6: {
                cell_vertices[1][0] = 7;
                cell_vertices[1][1] = 11;
                cell_vertices[1][2] = 5;
                cell_vertices[1][3] = 10;
                cell_vertices[1][4] = 3;
                cell_vertices[1][5] = 9;
                cell_vertices[1][6] = 1;
                cell_vertices[1][7] = 8;
                break;
              }

              case 7: {
                cell_vertices[1][0] = 3;
                cell_vertices[1][1] = 9;
                cell_vertices[1][2] = 7;
                cell_vertices[1][3] = 11;
                cell_vertices[1][4] = 1;
                cell_vertices[1][5] = 8;
                cell_vertices[1][6] = 5;
                cell_vertices[1][7] = 10;
                break;
              }
          } // switch
      }

    cells.resize(n_cells, CellData<3>());

    for (unsigned int cell_index = 0; cell_index < n_cells; ++cell_index)
      {
        for (const unsigned int vertex_index :
             GeometryInfo<3>::vertex_indices())
          {
            cells[cell_index].vertices[vertex_index] =
              cell_vertices[cell_index][vertex_index];
            cells[cell_index].material_id = 0;
          }
      }

    triangulation.create_triangulation(vertices, cells, SubCellData());
  }


  /*
   *
   */


  template <int dim>
  class ShapeFunctionWriter
  {
  public:
    ShapeFunctionWriter(const FiniteElement<dim> &fe,
                        const unsigned int        n_refine_each_cell,
                        const unsigned int        config_switch);

    void
    run();

  private:
    void
    make_grid_and_dofs_and_project(
      typename Triangulation<dim>::cell_iterator &cell);

    void
    output_results(typename Triangulation<dim>::cell_iterator &cell);

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

    Triangulation<dim> triangulation_coarse;

    const unsigned int degree;
    const unsigned int n_refine_each_cell;

    SmartPointer<const FiniteElement<dim>> fe_ptr;
    Triangulation<dim>                     triangulation;
    DoFHandler<dim>                        dof_handler;
    AffineConstraints<double>              constraints;
    std::vector<Vector<double>>            basis;

    unsigned int n_dofs_per_cell;
    unsigned int n_dofs_per_face;
    unsigned int n_dofs_per_quad;
    unsigned int n_dofs_per_line;
    unsigned int n_dofs_per_vertex;

    unsigned int first_line_index;
    unsigned int first_quad_index;
    unsigned int first_face_line_index;
    unsigned int first_face_quad_index;
  };

  template <int dim>
  ShapeFunctionWriter<dim>::ShapeFunctionWriter(
    const FiniteElement<dim> &_fe,
    const unsigned int        n_refine_each_cell,
    const unsigned int        config_switch)
    : degree(_fe.degree)
    , n_refine_each_cell(n_refine_each_cell)
    , fe_ptr(&_fe)
    , dof_handler(triangulation)
    , basis((*fe_ptr).n_dofs_per_cell())
  {
    /*
     * Assume all faces have the same number of dofs
     */
    n_dofs_per_cell   = (*fe_ptr).n_dofs_per_cell();
    n_dofs_per_face   = (*fe_ptr).n_dofs_per_face();
    n_dofs_per_quad   = (*fe_ptr).n_dofs_per_quad();
    n_dofs_per_line   = (*fe_ptr).n_dofs_per_line();
    n_dofs_per_vertex = (*fe_ptr).n_dofs_per_vertex();

    first_line_index      = (*fe_ptr).first_line_index;
    first_quad_index      = (*fe_ptr).first_quad_index;
    first_face_line_index = (*fe_ptr).first_face_line_index;
    first_face_quad_index = (*fe_ptr).first_face_quad_index;

    std::cout << "Element Info:  " << std::endl
              << "   n_dofs_per_cell      : " << n_dofs_per_cell << std::endl
              << "   n_dofs_per_face      : " << n_dofs_per_face << std::endl
              << "   n_dofs_per_quad      : " << n_dofs_per_quad << std::endl
              << "   n_dofs_per_line      : " << n_dofs_per_line << std::endl
              << "   n_dofs_per_vertex    : " << n_dofs_per_vertex << std::endl
              << "   first_line_index     : " << first_line_index << std::endl
              << "   first_quad_index     : " << first_quad_index << std::endl
              << "   first_face_line_index: " << first_face_line_index
              << std::endl
              << "   first_face_quad_index: " << first_face_quad_index
              << std::endl
              << std::endl
              << std::endl;


    ///////////////////////////////////
    ///////////////////////////////////
    ///////////////////////////////////

    //    GridGenerator::hyper_cube(triangulation_coarse, 0, 1, /* colorize */
    //    true);

    //    GridGenerator::hyper_shell(triangulation_coarse,
    //                               Point<dim>(),
    //                               1,
    //                               2,
    //                               /* n_cells */ 6,
    //                               /* colorize */ false);

    //    GridGenerator::moebius(triangulation_coarse,
    //                           /* n_cells */ 8,
    //                           /* n_rotations by pi/2*/ 1,
    //                           /* R */ 2,
    //                           /* r */ 0.5);

    {
      bool face_orientation = (((config_switch / 4) % 2) == 1);
      bool face_flip        = (((config_switch / 2) % 2) == 1);
      bool face_rotation    = ((config_switch % 2) == 1);

      bool manipulate_first_cube = false;

      generate_test_mesh(triangulation_coarse,
                         face_orientation,
                         face_flip,
                         face_rotation,
                         manipulate_first_cube);
    }

    triangulation_coarse.refine_global(0);

    //    GridTools::distort_random(0.2, triangulation_coarse, false);

    ///////////////////////////////////
    ///////////////////////////////////
    ///////////////////////////////////

    const bool do_plot = true;

    if (do_plot)
      {
        std::cout << "*********************************************************"
                  << std::endl
                  << "Writing finite element shape functions for   "
                  << (*fe_ptr).get_name()
                  << "   elements of (polynomial) degree   " << (*fe_ptr).degree
                  << std::endl
                  << std::endl;

        for (const auto &cell : triangulation_coarse.active_cell_iterators())
          {
            CellId current_cell_id(cell->id());


            std::cout
              << "CellId = " << current_cell_id << std::endl
              << "   {index -> face_orientation | face_flip | face_rotation}: "
              << std::endl;
            for (unsigned int face_index = 0;
                 face_index < GeometryInfo<dim>::faces_per_cell;
                 ++face_index)
              {
                std::cout << "      {" << face_index << " -> "
                          << cell->face_orientation(face_index) << " | "
                          << cell->face_flip(face_index) << " | "
                          << cell->face_rotation(face_index) << " | "
                          << "}" << std::endl;
              } // face_index

            std::cout << "   line orientation: {  ";
            for (unsigned int line_index = 0;
                 line_index < GeometryInfo<dim>::lines_per_cell;
                 ++line_index)
              {
                //        	  auto line = cell->line(line_index);
                std::cout << cell->line_orientation(line_index) << "  ";
              } // line_index
            std::cout << "}" << std::endl << std::endl;
          } // cell
      }     // if do_plot
  }


  template <int dim>
  void
  ShapeFunctionWriter<dim>::make_grid_and_dofs_and_project(
    typename Triangulation<dim>::cell_iterator &cell)
  {
    { // Prepare
      triangulation.clear();
      dof_handler.clear();
      constraints.clear();

      std::vector<Point<dim>> corners(GeometryInfo<dim>::vertices_per_cell);
      for (unsigned int i = 0; i < corners.size(); ++i)
        {
          corners[i] = cell->vertex(i);
        }

      GridGenerator::general_cell(triangulation,
                                  corners,
                                  /* colorize faces */ false);

      triangulation.refine_global(n_refine_each_cell);
    }

    dof_handler.distribute_dofs(*fe_ptr);

    { // Constraints
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
    }

    /*
     * Reinit and project the shape function
     */
    // ShapeFun::ShapeFunctionScalar<dim> shape_function(*fe_ptr,
    //                                                   cell,
    //                                                   /* verbose */ false);
    ShapeFun::ShapeFunctionVector<dim> shape_function(*fe_ptr,
                                                      cell,
                                                      /* verbose */ false);


    QGauss<dim> quad_rule(degree + 1);

    /*
     * Project only face dofs
     */
    std::cout << "Projecting   " << n_dofs_per_cell << "   dofs ...";
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      {
        basis[i].reinit(dof_handler.n_dofs());

        shape_function.set_shape_fun_index(i);
        VectorTools::project(
          dof_handler, constraints, quad_rule, shape_function, basis[i]);
      }
    std::cout << "done." << std::endl;
  }


  template <int dim>
  std::pair<unsigned int, bool>
  ShapeFunctionWriter<dim>::adjust_dof_index_and_sign_on_face_rt(
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
        if (((!cell->face_orientation(face_index_from_shape_index)) &&
             (!cell->face_rotation(face_index_from_shape_index))) ||
            ((cell->face_orientation(face_index_from_shape_index)) &&
             (cell->face_rotation(face_index_from_shape_index))))
          {
            unsigned int local_face_dof = dof_index % n_dofs_per_face;
            // Row and column
            unsigned int i = local_face_dof % n, j = local_face_dof / n;

            // We flip across the diagonal
            unsigned int offset = j + i * n - local_face_dof;

            new_dof_index = dof_index + offset;
          } // if face needs dof permutation

        /*
         * To determine if a (corrected) sign flip is necessary we need the new
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
            // This case is symmetric (although row and column may be switched)
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
         * If we are on a face that does not have standard orientation we must
         * flip all signs again
         */
        if (!cell->face_orientation(face_index_from_shape_index))
          sign_flip = !sign_flip;

      } // if dof_index < n_face_dofs

    std::pair<unsigned int, bool> new_dof_index_and_sign_flip(new_dof_index,
                                                              sign_flip);

    return new_dof_index_and_sign_flip;
  }


  template <int dim>
  std::pair<unsigned int, bool>
  ShapeFunctionWriter<dim>::adjust_dof_index_and_sign_on_face_bdm(
    typename Triangulation<dim>::cell_iterator &cell,
    const unsigned int                          dof_index,
    const unsigned int                          degree)
  {
    unsigned int new_dof_index = dof_index;

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
                                                              false);

    return new_dof_index_and_sign_flip;
  }



  template <int dim>
  void
  ShapeFunctionWriter<dim>::output_results(
    typename Triangulation<dim>::cell_iterator &cell)
  {
    const bool adjust_index_and_sign = false;

    std::function<std::pair<unsigned int, bool>(
      typename Triangulation<dim>::cell_iterator &,
      const unsigned int,
      const unsigned int)>
      adjust_dof_index_and_sign_on_face;

    std::string fe_rt_str(
      "FE_RaviartThomas<" + Utilities::int_to_string(dim, 1) + ">(" +
      Utilities::int_to_string((*fe_ptr).degree - 1, 1) + ")");
    std::string fe_bdm_str("FE_BDM<" + Utilities::int_to_string(dim, 1) + ">(" +
                           Utilities::int_to_string((*fe_ptr).degree - 1, 1) +
                           ")");

    if (fe_rt_str.compare((*fe_ptr).get_name()) == 0)
      {
        adjust_dof_index_and_sign_on_face = std::bind(
          &ShapeFunctionWriter<dim>::adjust_dof_index_and_sign_on_face_rt,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3);
      }
    else if (fe_bdm_str.compare((*fe_ptr).get_name()) == 0)
      {
        adjust_dof_index_and_sign_on_face = std::bind(
          &ShapeFunctionWriter<dim>::adjust_dof_index_and_sign_on_face_bdm,
          this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3);
      }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    if (adjust_index_and_sign)
      {
        std::cout << "Cell with id   " << cell->id().to_string()
                  << "   has permuted dofs on faces:" << std::endl;
      }

    for (unsigned int dof_index_in = 0; dof_index_in < n_dofs_per_cell;
         ++dof_index_in)
      {
        const std::pair<unsigned int, bool> dof_index_and_sign =
          (adjust_index_and_sign ?
             adjust_dof_index_and_sign_on_face(cell,
                                               dof_index_in,
                                               (*fe_ptr).degree) :
             std::pair<unsigned int, bool>(dof_index_in, false));

        const unsigned int dof_index = dof_index_and_sign.first;

        if (dof_index_and_sign.second)
          {
            basis[dof_index] *= -1.0;
          }

        if (adjust_index_and_sign)
          {
            if (((dof_index_in - dof_index) != 0) || dof_index_and_sign.second)
              {
                std::cout << "   " << dof_index_in << " ---> " << dof_index
                          << "   sign change = " << dof_index_and_sign.second
                          << std::endl;
              }
          }

        if (fe_ptr->n_components() == 1)
          {
            const std::vector<std::string> solution_name(
              1, std::string("u") + Utilities::int_to_string(dof_index_in, 3));
            const std::vector<
              DataComponentInterpretation::DataComponentInterpretation>
              interpretation(1,
                             DataComponentInterpretation::component_is_scalar);

            data_out.add_data_vector(basis[dof_index],
                                     solution_name,
                                     DataOut<dim>::type_dof_data,
                                     interpretation);
          }
        else
          {
            const std::vector<std::string> solution_name(
              dim,
              std::string("u") + Utilities::int_to_string(dof_index_in, 3));
            const std::vector<
              DataComponentInterpretation::DataComponentInterpretation>
              interpretation(
                dim, DataComponentInterpretation::component_is_part_of_vector);

            data_out.add_data_vector(basis[dof_index],
                                     solution_name,
                                     DataOut<dim>::type_dof_data,
                                     interpretation);
          }
      }

    data_out.build_patches(degree + 1);

    std::string filename = "basis_cell-";
    filename += cell->id().to_string();

    std::ofstream output(filename + ".vtu");
    data_out.write_vtu(output);
  }


  template <int dim>
  void
  ShapeFunctionWriter<dim>::run()
  {
    for (auto cell : triangulation_coarse.active_cell_iterators())
      {
        make_grid_and_dofs_and_project(cell);

        output_results(cell);
      }
  }
} // namespace Step20


////////////////////////////////////
////////////////////////////////////
////////////////////////////////////


int
main(int argc, char *argv[])
{
  // Very simple way of input handling.
  if (argc < 2)
    {
      std::cout
        << "You must provide an initial number of global refinements for each cell "
           "\"-n n_refine_each_cell\""
        << std::endl;
      exit(1);
    }

  unsigned int n_refine_each_cell = 0;
  unsigned int config_switch      = 4; // (true | false | false)

  std::list<std::string> args;
  for (int i = 1; i < argc; ++i)
    {
      args.push_back(argv[i]);
    }

  while (args.size())
    {
      if (args.front() == std::string("-n"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-n' must be followed by the "
                        << "number of initial refinements." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();

              try
                {
                  std::size_t pos;
                  n_refine_each_cell = std::stoi(args.front(), &pos);
                  if (pos < args.front().size())
                    {
                      std::cerr
                        << "Trailing characters after number: " << args.front()
                        << '\n';
                    }
                }
              catch (std::invalid_argument const &ex)
                {
                  std::cerr << "Invalid number: " << args.front() << '\n';
                }
              catch (std::out_of_range const &ex)
                {
                  std::cerr << "Number out of range: " << args.front() << '\n';
                }

              args.pop_front();
            }
        }
      else if (args.front() == std::string("-c"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-c' must be followed by the "
                        << "cell configuraation." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();

              try
                {
                  std::size_t pos;
                  config_switch = std::stoi(args.front(), &pos);
                  if (pos < args.front().size())
                    {
                      std::cerr
                        << "Trailing characters after number: " << args.front()
                        << '\n';
                    }
                }
              catch (std::invalid_argument const &ex)
                {
                  std::cerr << "Invalid number: " << args.front() << '\n';
                }
              catch (std::out_of_range const &ex)
                {
                  std::cerr << "Number out of range: " << args.front() << '\n';
                }

              args.pop_front();
            }
        }
      else
        {
          std::cerr << "Unknown command line option: " << args.front()
                    << std::endl;
          exit(1);
        }
    } // end while

  try
    {
      using namespace Step20;

      const int          dim       = 3;
      const unsigned int fe_degree = 1;

      //      FE_BDM<dim> fe(fe_degree);
      FE_RaviartThomas<dim> fe(fe_degree);
      //      FE_Nedelec<dim> fe(fe_degree);
      //      FE_NedelecSZ<dim> fe(fe_degree);
      //      FE_BernardiRaugel<dim> fe(fe_degree);

      // FE_Q<dim> fe(fe_degree);

      {
        ShapeFunctionWriter<dim> shape_function_writer(fe,
                                                       n_refine_each_cell,
                                                       config_switch);
        shape_function_writer.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
