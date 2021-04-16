#define Limit_Threads_For_DEBUG

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_abf.h>
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

#include <grid.h>
#include <shape_fun_scalar.hpp>
#include <shape_fun_vector.hpp>
#include <test_mesh.h>

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
                        const unsigned int        n_refine_each_cell,
                        const unsigned int        config_switch);

    void
    run();

  private:
    void
    make_grid_and_dofs_and_project(
      typename Triangulation<dim>::cell_iterator &cell);

    void
    plot_mesh_info();

    void
    output_results(typename Triangulation<dim>::cell_iterator &cell);

    Triangulation<dim> triangulation_coarse;

    const unsigned int degree;
    const unsigned int n_refine_each_cell;
    const unsigned int config_switch;

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

    const bool do_plot = true;
  };

  template <>
  ShapeFunctionWriter<3>::ShapeFunctionWriter(
    const FiniteElement<3> &_fe,
    const unsigned int      n_refine_each_cell,
    const unsigned int      config_switch)
    : degree(_fe.degree)
    , n_refine_each_cell(n_refine_each_cell)
    , config_switch(config_switch)
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
    bool face_orientation = (((config_switch / 4) % 2) == 1);
    bool face_flip        = (((config_switch / 2) % 2) == 1);
    bool face_rotation    = ((config_switch % 2) == 1);

    bool manipulate_first_cube = true;

    GridGenerator::non_standard_orientation_mesh(triangulation_coarse,
                                                 face_orientation,
                                                 face_flip,
                                                 face_rotation,
                                                 manipulate_first_cube);

    triangulation_coarse.refine_global(0);
    // GridTools::distort_random(0.2, triangulation_coarse, false);
    ///////////////////////////////////
    ///////////////////////////////////

    plot_mesh_info();
  }


  template <>
  ShapeFunctionWriter<2>::ShapeFunctionWriter(
    const FiniteElement<2> &_fe,
    const unsigned int      n_refine_each_cell,
    const unsigned int      config_switch)
    : degree(_fe.degree)
    , n_refine_each_cell(n_refine_each_cell)
    , config_switch(config_switch)
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
    AssertThrow(config_switch < 4,
                ExcMessage("If dim=2 the config witch must be less that 3."));

    const unsigned int n_rotate_right_square = config_switch;

    GridGenerator::non_standard_orientation_mesh(triangulation_coarse,
                                                 n_rotate_right_square);

    triangulation_coarse.refine_global(0);
    // GridTools::distort_random(0.2, triangulation_coarse, false);
    ///////////////////////////////////
    ///////////////////////////////////

    plot_mesh_info();
  }

  template <int dim>
  void
  ShapeFunctionWriter<dim>::plot_mesh_info()
  {
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
    ShapeFun::ShapeFunctionVector<dim> shape_function(
      *fe_ptr,
      cell,
      /* verbose */ false,
      /* adjust_index_and_sign */ false);


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
  void
  ShapeFunctionWriter<dim>::output_results(
    typename Triangulation<dim>::cell_iterator &cell)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    for (unsigned int dof_index = 0; dof_index < n_dofs_per_cell; ++dof_index)
      {
        if (fe_ptr->n_components() == 1)
          {
            const std::vector<std::string> solution_name(
              1, std::string("u") + Utilities::int_to_string(dof_index, 3));
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
              dim, std::string("u") + Utilities::int_to_string(dof_index, 3));
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
#ifdef DEBUG
#  ifdef Limit_Threads_For_DEBUG
      dealii::MultithreadInfo::set_thread_limit(1);
#  endif
#endif
      using namespace Step20;

      constexpr int      dim       = 2;
      const unsigned int fe_degree = 0;

      // FE_BDM<dim>           fe(fe_degree);
      // FE_ABF<dim>           fe(fe_degree);
      FE_RaviartThomas<dim> fe(fe_degree);
      // FE_Nedelec<dim> fe(fe_degree);
      //  FE_NedelecSZ<dim> fe(fe_degree);
      //  FE_BernardiRaugel<dim> fe(fe_degree);

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