/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2005 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */



#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
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

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace Step20
{
  using namespace dealii;


  template <int dim>
  class MixedLaplaceProblem
  {
  public:
    MixedLaplaceProblem(const unsigned int degree,
                        const unsigned int _n_refine,
                        const bool         _natural_bc,
                        const bool         _problematic_domain);

    void
    run();

  private:
    void
    make_grid_and_dofs();

    void
    annotate_face_orientation();

    void
    assemble_system();

    void
    solve();

    void
    compute_errors() const;

    void
    output_results() const;

    const unsigned int degree;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    AffineConstraints<double> constraints;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;

    BlockVector<double> projected_exact_solution;

    const unsigned int n_refine;

    const bool natural_bc;
    const bool problematic_domain;

    const std::string domain_info;
  };



  namespace PrescribedSolution
  {
    constexpr double alpha = 0.3;
    constexpr double beta  = 1;


    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>(1)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;
    };



    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
      PressureBoundaryValues()
        : Function<dim>(1)
      {}

      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;
    };


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


    template <int dim>
    double
    RightHandSide<dim>::value(const Point<dim> & /*p*/,
                              const unsigned int /*component*/) const
    {
      return 0;
    }



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



    template <>
    void
    ExactSolution<2>::vector_value(const Point<2> &p,
                                   Vector<double> &values) const
    {
      Assert(values.size() == 2 + 1,
             ExcDimensionMismatch(values.size(), 2 + 1));

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
      Assert(values.size() == 3 + 1,
             ExcDimensionMismatch(values.size(), 3 + 1));

      values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
      values(1) = alpha * p[0] * p[1];
      values(2) = -alpha;
      values(3) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                    alpha * p[0] * p[0] * p[0] / 6 + alpha * p[2]);
    }


    template <int dim>
    class ExactSolutionVelocity : public Function<dim>
    {
    public:
      ExactSolutionVelocity()
        : Function<dim>(dim)
      {}

      virtual void
      vector_value(const Point<dim> &p, Vector<double> &value) const override;
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
    ExactSolutionVelocity<3>::vector_value(const Point<3> &p,
                                           Vector<double> &values) const
    {
      Assert(values.size() == 3, ExcDimensionMismatch(values.size(), 3));

      values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
      values(1) = alpha * p[0] * p[1];
      values(2) = -alpha;
    }


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



  template <int dim>
  MixedLaplaceProblem<dim>::MixedLaplaceProblem(const unsigned int degree,
                                                const unsigned int _n_refine,
                                                const bool         _natural_bc,
                                                const bool _problematic_domain)
    : degree(degree)
    , fe(FE_RaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1)
    //    , fe(FE_BDM<dim>(degree + 1), 1, FE_DGP<dim>(degree), 1)
    , dof_handler(triangulation)
    , n_refine(_n_refine)
    , natural_bc(_natural_bc)
    , problematic_domain(_problematic_domain)
    , domain_info(
        (problematic_domain ? "_problematic_domain" : "_cuboid_domain"))
  {}



  template <int dim>
  void
  MixedLaplaceProblem<dim>::make_grid_and_dofs()
  {
    if (problematic_domain)
      {
        if (false)
          {
            /*
             * Hyper_shell
             */
            std::cout << "Using spherical domain..." << std::endl;

            GridGenerator::hyper_shell(triangulation,
                                       Point<dim>(),
                                       1,
                                       2,
                                       /* n_cells */ (dim == 3) ? 6 : 12,
                                       /* colorize */ false);

            triangulation.refine_global(n_refine);
          }
        else
          {
            /*
             * Plate with hole
             */
            GridGenerator::plate_with_a_hole(triangulation,
                                             /* inner_radius */ 0.5,
                                             /* outer_radius */ 1.,
                                             /* pad_bottom */ 1.,
                                             /* pad_top */ 1.,
                                             /* pad_left */ 1.,
                                             /* pad_right */ 1.,
                                             /* center */ Point<dim>());

            triangulation.refine_global(n_refine);
          }
      }
    else
      {
        /*
         * Cube
         */
        std::cout << "Using cuboid domain..." << std::endl;

        GridGenerator::hyper_cube(triangulation, -1, 1, /* colorize */ false);
        triangulation.refine_global(n_refine);
      }

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::component_wise(dof_handler);

    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim];

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')' << std::endl;


    {
      constraints.clear();

      if (!natural_bc)
        {
          std::cout
            << "Emplying essential boundary conditions in H(div), i.e., on u..."
            << std::endl;

          PrescribedSolution::ExactSolutionVelocity<dim>
            exact_solution_velocity;

          VectorTools::project_boundary_values_div_conforming(
            dof_handler,
            /*first vector component */ 0,
            exact_solution_velocity,
            /*boundary id*/ 0,
            constraints);
        }
      else
        {
          std::cout
            << "Emplying natural boundary conditions in H(div), i.e., on p ..."
            << std::endl;
        }

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      constraints.close();
    }

    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.collect_sizes();
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(2);
    solution.block(0).reinit(n_u);
    solution.block(1).reinit(n_p);
    solution.collect_sizes();

    projected_exact_solution.reinit(2);
    projected_exact_solution.block(0).reinit(n_u);
    projected_exact_solution.block(1).reinit(n_p);
    projected_exact_solution.collect_sizes();

    system_rhs.reinit(2);
    system_rhs.block(0).reinit(n_u);
    system_rhs.block(1).reinit(n_p);
    system_rhs.collect_sizes();
  }


  template <int dim>
  void
  MixedLaplaceProblem<dim>::annotate_face_orientation()
  {
    const bool do_plot = true;


    std::vector<bool> orientation(GeometryInfo<dim>::faces_per_cell, true);
    std::map<CellId, std::vector<bool>> cell_wise_face_orientation;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        CellId current_cell_id(cell->id());
        for (unsigned int face_index = 0;
             face_index < GeometryInfo<dim>::faces_per_cell;
             ++face_index)
          {
            orientation[face_index] = cell->face_orientation(face_index);
          }

        cell_wise_face_orientation.emplace(current_cell_id, orientation);

        //        for (unsigned int face_index = 0;
        //             face_index < GeometryInfo<dim>::faces_per_cell;
        //             ++face_index)
        //          {
        //            const auto &face = cell->face(face_index);
        //          } // face_index

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
  MixedLaplaceProblem<dim>::assemble_system()
  {
    QGauss<dim>     quadrature_formula(degree + 2);
    QGauss<dim - 1> face_quadrature_formula(degree + 2);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    const PrescribedSolution::RightHandSide<dim> right_hand_side;
    const PrescribedSolution::PressureBoundaryValues<dim>
                                            pressure_boundary_values;
    const PrescribedSolution::KInverse<dim> k_inverse;

    std::vector<double>         rhs_values(n_q_points);
    std::vector<double>         boundary_values(n_face_q_points);
    std::vector<Tensor<2, dim>> k_inverse_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);
        k_inverse.value_list(fe_values.get_quadrature_points(),
                             k_inverse_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
              const double div_phi_i_u = fe_values[velocities].divergence(i, q);
              const double phi_i_p     = fe_values[pressure].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_u =
                    fe_values[velocities].value(j, q);
                  const double div_phi_j_u =
                    fe_values[velocities].divergence(j, q);
                  const double phi_j_p = fe_values[pressure].value(j, q);

                  local_matrix(i, j) +=
                    (phi_i_u * k_inverse_values[q] * phi_j_u //
                     - phi_i_p * div_phi_j_u                 //
                     - div_phi_i_u * phi_j_p)                //
                    * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p * rhs_values[q] * fe_values.JxW(q);
            }

        if (natural_bc)
          {
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary())
                {
                  fe_face_values.reinit(cell, face);

                  pressure_boundary_values.value_list(
                    fe_face_values.get_quadrature_points(), boundary_values);

                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      local_rhs(i) +=
                        -(fe_face_values[velocities].value(i, q) * //
                          fe_face_values.normal_vector(q) *        //
                          boundary_values[q] *                     //
                          fe_face_values.JxW(q));
                }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }
  }



  template <int dim>
  void
  MixedLaplaceProblem<dim>::solve()
  {
    const auto &M = system_matrix.block(0, 0);
    const auto &B = system_matrix.block(0, 1);

    const auto &F = system_rhs.block(0);
    const auto &G = system_rhs.block(1);

    auto &U = solution.block(0);
    auto &P = solution.block(1);

    const auto op_M = linear_operator(M);
    const auto op_B = linear_operator(B);

    ReductionControl         reduction_control_M(2000, 1.0e-18, 1.0e-10);
    SolverCG<Vector<double>> solver_M(reduction_control_M);
    PreconditionJacobi<SparseMatrix<double>> preconditioner_M;

    preconditioner_M.initialize(M);

    const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

    const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
    const auto op_aS =
      transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;

    IterationNumberControl   iteration_number_control_aS(30, 1.e-18);
    SolverCG<Vector<double>> solver_aS(iteration_number_control_aS);

    const auto preconditioner_S =
      inverse_operator(op_aS, solver_aS, PreconditionIdentity());

    const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

    SolverControl            solver_control_S(2000, 1.e-12);
    SolverCG<Vector<double>> solver_S(solver_control_S);

    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);

    P = op_S_inv * schur_rhs;

    std::cout << solver_control_S.last_step()
              << " CG Schur complement iterations to obtain convergence."
              << std::endl;

    U = op_M_inv * (F - op_B * P);

    constraints.distribute(solution);

    /*
     * Now project the exact solution. This is meant as a sanity check.
     */
    AffineConstraints<double> no_constraints;
    no_constraints.clear();
    no_constraints.close();

    QGauss<dim> quad_rule(degree + 3);

    VectorTools::project(dof_handler,
                         no_constraints,
                         quad_rule,
                         PrescribedSolution::ExactSolution<dim>(),
                         projected_exact_solution);
  }



  template <int dim>
  void
  MixedLaplaceProblem<dim>::compute_errors() const
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);

    PrescribedSolution::ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors(triangulation.n_active_cells());

    QTrapezoid<1>  q_trapez;
    QIterated<dim> quadrature(q_trapez, degree + 2);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
  }



  template <int dim>
  void
  MixedLaplaceProblem<dim>::output_results() const
  {
    DataOut<dim> data_out;

    std::vector<std::string> solution_names(dim, "u");
    solution_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    std::vector<std::string> exact_solution_names(dim, "u_exact");
    exact_solution_names.emplace_back("p_exact");
    data_out.add_data_vector(dof_handler,
                             projected_exact_solution,
                             exact_solution_names,
                             interpretation);

    data_out.build_patches(degree + 1);

    std::string filename = "solution";
    filename +=
      (natural_bc ? std::string("_natural_bc") : std::string("_essential_bc"));
    filename += domain_info + "-";
    filename += Utilities::int_to_string(n_refine, 2);

    std::ofstream output(filename + ".vtu");
    data_out.write_vtu(output);
  }



  template <int dim>
  void
  MixedLaplaceProblem<dim>::run()
  {
    make_grid_and_dofs();
    annotate_face_orientation();
    assemble_system();
    solve();
    compute_errors();
    output_results();
  }
} // namespace Step20


/*
 * Main function.
 */
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
  unsigned int dim      = 0;

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
      else if (args.front() == std::string("-d"))
        {
          if (args.size() == 1) /* This is not robust. */
            {
              std::cerr << "Error: flag '-d' must be followed by the "
                        << "space dimension." << std::endl;
              exit(1);
            }
          else
            {
              args.pop_front();

              try
                {
                  std::size_t pos;
                  dim = std::stoi(args.front(), &pos);
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

  /*
   * Now run the code
   */
  try
    {
      using namespace Step20;

      const unsigned int fe_degree          = 0;
      const bool         problematic_domain = true;

      if (dim == 2)
        {
          {
            MixedLaplaceProblem<2> mixed_laplace_problem(fe_degree,
                                                         n_refine,
                                                         /* natural_bc
                                                          */
                                                         true,
                                                         problematic_domain);
            mixed_laplace_problem.run();
          }

          {
            MixedLaplaceProblem<2> mixed_laplace_problem(fe_degree,
                                                         n_refine,
                                                         /* natural_bc */
                                                         false,
                                                         problematic_domain);
            mixed_laplace_problem.run();
          }
        }
      else if (dim == 3)
        {
          {
            MixedLaplaceProblem<3> mixed_laplace_problem(fe_degree,
                                                         n_refine,
                                                         /* natural_bc
                                                          */
                                                         true,
                                                         problematic_domain);
            mixed_laplace_problem.run();
          }

          {
            MixedLaplaceProblem<3> mixed_laplace_problem(fe_degree,
                                                         n_refine,
                                                         /* natural_bc */
                                                         false,
                                                         problematic_domain);
            mixed_laplace_problem.run();
          }
        }
      else
        {
          std::cerr << "Dimension not supported." << std::endl;
          exit(1);
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
