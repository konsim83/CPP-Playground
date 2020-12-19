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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
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

  template <int dim>
  class ShapeFunctionWriter
  {
  public:
    ShapeFunctionWriter(const FiniteElement<dim> &fe,
                        const unsigned int        n_refine,
                        const unsigned int        n_refine_each_cell);

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
    unsigned int
    flip_dof_order_on_face_rt(typename Triangulation<dim>::cell_iterator &cell,
                              const unsigned int shape_fun_index,
                              const unsigned int order);

    unsigned int
    flip_dof_order_on_face_bdm(typename Triangulation<dim>::cell_iterator &cell,
                               const unsigned int shape_fun_index,
                               const unsigned int order);

    Triangulation<dim> triangulation_coarse;

    const unsigned int degree;
    const unsigned int n_refine;
    const unsigned int n_refine_each_cell;

    SmartPointer<const FiniteElement<dim>> fe_ptr;
    Triangulation<dim>                     triangulation;
    DoFHandler<dim>                        dof_handler;
    AffineConstraints<double>              constraints;
    std::vector<Vector<double>>            basis;

    unsigned int n_face_dofs;
  };

  template <int dim>
  ShapeFunctionWriter<dim>::ShapeFunctionWriter(
    const FiniteElement<dim> &_fe,
    const unsigned int        n_refine,
    const unsigned int        n_refine_each_cell)
    : degree(_fe.degree)
    , n_refine(n_refine)
    , n_refine_each_cell(n_refine_each_cell)
    , fe_ptr(&_fe)
    , dof_handler(triangulation)
    , basis((*fe_ptr).n_dofs_per_cell())
  {
    /*
     * Assume all faces have the same number of dofs
     */
    n_face_dofs =
      (*fe_ptr).n_dofs_per_face() * GeometryInfo<dim>::faces_per_cell;

    ///////////////////////////////////
    ///////////////////////////////////
    ///////////////////////////////////

    //    GridGenerator::hyper_shell(triangulation_coarse,
    //                               Point<dim>(),
    //                               1,
    //                               2,
    //                               /* n_cells */ 6,
    //                               /* colorize */ false);

    GridGenerator::moebius(triangulation_coarse,
                           /* n_cells */ 8,
                           /* n_rotations by pi/2*/ 3,
                           /* R */ 2,
                           /* r */ 0.5);

    //    GridGenerator::hyper_ball(triangulation_coarse);

    //    GridTools::distort_random(/* factor */ 0.15,
    //                              triangulation_coarse,
    //                              /* keep_boundary */ false);

    triangulation_coarse.refine_global(n_refine);

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
                //                typename Triangulation<dim>::face_iterator
                //                face =
                //                  cell->face(face_index);

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
    ShapeFun::ShapeFunctionVector<dim> shape_function(*fe_ptr,
                                                      cell,
                                                      /* degree */ degree);
    QGauss<dim>                        quad_rule(degree + 3);

    /*
     * Project only face dofs
     */
    std::cout << "Projecting   " << n_face_dofs << "   face dofs ...";
    for (unsigned int i = 0; i < n_face_dofs; ++i)
      {
        basis[i].reinit(dof_handler.n_dofs());

        shape_function.set_shape_fun_index(i);
        VectorTools::project(
          dof_handler, constraints, quad_rule, shape_function, basis[i]);
      }
    std::cout << "done." << std::endl;
  }


  template <int dim>
  unsigned int
  ShapeFunctionWriter<dim>::flip_dof_order_on_face_rt(
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
             (!cell->face_rotation(!face_index_from_shape_index))) ||
            ((!cell->face_orientation(face_index_from_shape_index)) &&
             (!cell->face_rotation(face_index_from_shape_index))) ||
            ((cell->face_orientation(face_index_from_shape_index)) &&
             (cell->face_rotation(face_index_from_shape_index))) ||
            ((cell->face_orientation(face_index_from_shape_index)) &&
             (cell->face_rotation(face_index_from_shape_index))))
          {
            //            if (degree == 2)
            //              {
            //                if (dof_index % n_dofs_per_face == 0)
            //                  {
            //                    new_dof_index = dof_index;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 1)
            //                  {
            //                    new_dof_index = dof_index + 1;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 2)
            //                  {
            //                    new_dof_index = dof_index - 1;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 3)
            //                  {
            //                    new_dof_index = dof_index;
            //                  }
            //              } // degree == 2
            //            else if (degree == 3)
            //              {
            //                if (dof_index % n_dofs_per_face == 0)
            //                  {
            //                    new_dof_index = dof_index;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 1)
            //                  {
            //                    new_dof_index = dof_index + 2;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 2)
            //                  {
            //                    new_dof_index = dof_index + 4;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 3)
            //                  {
            //                    new_dof_index = dof_index - 2;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 4)
            //                  {
            //                    new_dof_index = dof_index;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 5)
            //                  {
            //                    new_dof_index = dof_index + 2;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 6)
            //                  {
            //                    new_dof_index = dof_index - 4;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 7)
            //                  {
            //                    new_dof_index = dof_index - 2;
            //                  }
            //                else if (dof_index % n_dofs_per_face == 8)
            //                  {
            //                    new_dof_index = dof_index;
            //                  }
            //              } // degree == 3
            //            else if (degree == 4)
            //              {
            const unsigned int n              = degree;
            unsigned int       local_face_dof = dof_index % n_dofs_per_face;

            // Row and column
            unsigned int i = local_face_dof % n, j = local_face_dof / n;

            unsigned int offset = j + i * n - local_face_dof;

            new_dof_index = dof_index + offset;
            //              } // degree ==4
          } // if face needs dofpermutation
      }     // if dof_index < n_face_dofs

    return new_dof_index;
  }


  template <int dim>
  unsigned int
  ShapeFunctionWriter<dim>::flip_dof_order_on_face_bdm(
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
             (!cell->face_rotation(!face_index_from_shape_index))) ||
            ((!cell->face_orientation(face_index_from_shape_index)) &&
             (!cell->face_rotation(face_index_from_shape_index))) ||
            ((cell->face_orientation(face_index_from_shape_index)) &&
             (cell->face_rotation(face_index_from_shape_index))) ||
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

    return new_dof_index;
  }



  template <int dim>
  void
  ShapeFunctionWriter<dim>::output_results(
    typename Triangulation<dim>::cell_iterator &cell)
  {
    const bool flip = true;

    std::function<unsigned int(typename Triangulation<dim>::cell_iterator &,
                               const unsigned int,
                               const unsigned int)>
      flip_dof_order_on_face;

    std::string fe_rt_str(
      "FE_RaviartThomas<" + Utilities::int_to_string(dim, 1) + ">(" +
      Utilities::int_to_string((*fe_ptr).degree - 1, 1) + ")");
    std::string fe_bdm_str("FE_BDM<" + Utilities::int_to_string(dim, 1) + ">(" +
                           Utilities::int_to_string((*fe_ptr).degree - 1, 1) +
                           ")");

    if (fe_rt_str.compare((*fe_ptr).get_name()) == 0)
      {
        flip_dof_order_on_face =
          std::bind(&ShapeFunctionWriter<dim>::flip_dof_order_on_face_rt,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    std::placeholders::_3);
      }
    else if (fe_bdm_str.compare((*fe_ptr).get_name()) == 0)
      {
        flip_dof_order_on_face =
          std::bind(&ShapeFunctionWriter<dim>::flip_dof_order_on_face_bdm,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    std::placeholders::_3);
      }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    if (flip)
      {
        std::cout << "Cell with id   " << cell->id().to_string()
                  << "   has permuted dofs on faces:" << std::endl;
      }

    for (unsigned int dof_index_in = 0; dof_index_in < n_face_dofs;
         ++dof_index_in)
      {
        const unsigned int dof_index =
          (flip ? flip_dof_order_on_face(cell, dof_index_in, (*fe_ptr).degree) :
                  dof_index_in);

        if (flip)
          {
            if ((dof_index_in - flip_dof_order_on_face(cell,
                                                       dof_index_in,
                                                       (*fe_ptr).degree)) != 0)
              {
                std::cout << "   " << dof_index_in << " ---> "
                          << flip_dof_order_on_face(cell,
                                                    dof_index_in,
                                                    (*fe_ptr).degree)
                          << std::endl;
              }
          }

        const std::vector<std::string> solution_name(
          dim, std::string("u") + Utilities::int_to_string(dof_index_in, 3));
        const std::vector<
          DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(basis[dof_index],
                                 solution_name,
                                 DataOut<dim>::type_dof_data,
                                 interpretation);
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

        cell++;
        cell++;
        cell++;
        cell++;
        cell++;
        cell++;
        cell++;
        make_grid_and_dofs_and_project(cell);

        output_results(cell);
        break;
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

      const int dim = 3;


      const unsigned int n_refine  = 0; // refines the test domain itself
      const unsigned int fe_degree = 3;

      //      FE_BDM<dim> fe(fe_degree);
      FE_RaviartThomas<dim> fe(fe_degree);
      //      FE_Nedelec<dim> fe(fe_degree);

      {
        ShapeFunctionWriter<dim> shape_function_writer(fe,
                                                       n_refine,
                                                       n_refine_each_cell);
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
