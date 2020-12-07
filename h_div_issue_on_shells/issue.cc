#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
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
#include <iostream>
#include <map>
#include <vector>

namespace Step20
{
  using namespace dealii;

  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem(const unsigned int degree, const unsigned int n_refine);

    void
    run();

  private:
    void
    make_grid_and_dofs_and_project(
      typename Triangulation<dim>::cell_iterator &cell);

    void
    output_results(typename Triangulation<dim>::cell_iterator &cell);

    Triangulation<dim> triangulation_coarse;

    const unsigned int degree;
    const unsigned int n_refine;

    FE_RaviartThomas<dim>       fe;
    Triangulation<dim>          triangulation;
    DoFHandler<dim>             dof_handler;
    AffineConstraints<double>   constraints;
    std::vector<Vector<double>> basis;

    std::vector<bool> orientation;
  };



  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem(const unsigned int degree,
                                                const unsigned int n_refine)
    : degree(degree)
    , n_refine(n_refine)
    , fe(FE_RaviartThomas<dim>(degree))
    //    , fe(FE_BDM<dim>(degree))
    , dof_handler(triangulation)
    , basis(GeometryInfo<dim>::faces_per_cell)
    , orientation(GeometryInfo<dim>::faces_per_cell, true)
  {
    GridGenerator::hyper_shell(triangulation_coarse,
                               Point<dim>(),
                               1,
                               2,
                               /* n_cells */ 6,
                               /* colorize */ false);

    triangulation_coarse.refine_global(n_refine);

    /*
     * Annottate correction
     */

    const bool do_plot = true;

    std::map<CellId, std::vector<bool>> cell_wise_face_orientation;

    for (const auto &cell : triangulation_coarse.active_cell_iterators())
      {
        CellId current_cell_id(cell->id());
        for (unsigned int face_index = 0;
             face_index < GeometryInfo<dim>::faces_per_cell;
             ++face_index)
          {
            orientation[face_index] = cell->face_orientation(face_index);
          }

        cell_wise_face_orientation.emplace(current_cell_id, orientation);

        for (unsigned int face_index = 0;
             face_index < GeometryInfo<dim>::faces_per_cell;
             ++face_index)
          {
            const auto &face = cell->face(face_index);
          } // face_index

        if (do_plot)
          { // Plot orientation
            std::cout << "   CellId = " << current_cell_id
                      << "   orientation: {  ";
            for (const auto &entry : orientation)
              {
                std::cout << entry << "  ";
              }
            std::cout << "}" << std::endl;
          }
      } // cell
  }


  template <int dim>
  void
  MixedLaplaceProblem<dim>::make_grid_and_dofs_and_project(
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

      triangulation.refine_global(3);
    }

    dof_handler.distribute_dofs(fe);

    { // Constraints
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
    }


    /*
     * Reinit and project the shape function
     */
    ShapeFun::ShapeFunctionVector<dim> shape_function_RT(fe,
                                                         cell,
                                                         /* degree */ degree);
    QGauss<dim>                        quad_rule(degree + 3);

    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      {
        basis[i].reinit(dof_handler.n_dofs());

        shape_function_RT.set_shape_fun_index(i);

        VectorTools::project(
          dof_handler, constraints, quad_rule, shape_function_RT, basis[i]);
      }

    output_results(cell);
  }


  template <int dim>
  void
  MixedLaplaceProblem<dim>::output_results(
    typename Triangulation<dim>::cell_iterator &cell)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      {
        const std::vector<std::string> solution_name(
          dim, std::string("u") + Utilities::int_to_string(i, 1));
        const std::vector<
          DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);

        data_out.add_data_vector(basis[i],
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
  MixedLaplaceProblem<dim>::run()
  {
    for (auto cell : triangulation_coarse.active_cell_iterators())
      {
        make_grid_and_dofs_and_project(cell);

        output_results(cell);
      }
  }
} // namespace Step20

int
main(int argc, char *argv[])
{
  // Very simple way of input handling.
  if (argc < 2)
    {
      std::cout
        << "You must provide an initial number of global refinements \"-n n_refine\""
        << std::endl;
      exit(1);
    }

  unsigned int n_refine = 0;

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
                  n_refine = std::stoi(args.front(), &pos);
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

      const unsigned int fe_degree = 0;
      const int          dim       = 3;

      {
        MixedLaplaceProblem<dim> mixed_laplace_problem(fe_degree, n_refine);
        mixed_laplace_problem.run();
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
