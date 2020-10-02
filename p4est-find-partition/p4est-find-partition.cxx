// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/p4est_wrappers.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// p4est
#include <p4est.h>
#include <p4est_search.h>

// Boost
#include <boost/optional.hpp>

// STL
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>

/*
 * Generate a random double number between [a,b)
 */
class RandomNumberDouble
{
public:
  RandomNumberDouble() = delete;
  RandomNumberDouble(const double _a, const double _b)
    : a(_a)
    , b(_b)
    , uniform_distribution(a, b)
    , timeSeed(
        std::chrono::high_resolution_clock::now().time_since_epoch().count())
    , seed_sequence(std::seed_seq{uint32_t(timeSeed & 0xffffffff),
                                  uint32_t(timeSeed >> 32)})
  {
    rng.seed(seed_sequence);
  }

  void
  reinit()
  {
    // re-initialize the random number generator with time-dependent seed
    timeSeed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq seed_sequence{uint32_t(timeSeed & 0xffffffff),
                                uint32_t(timeSeed >> 32)};
    rng.seed(seed_sequence);
  }

  double
  generate()
  {
    return uniform_distribution(rng);
  }

private:
  double                                 a, b;
  std::uniform_real_distribution<double> uniform_distribution;
  uint64_t                               timeSeed;
  std::seed_seq                          seed_sequence;
  std::mt19937_64                        rng;
};


/*
 * Fill a list of points with random numbers between a  and b.
 */
template <int dim>
void
fill_points_randomly(std::vector<Point<dim>> &points,
                     double                   a = 0,
                     double                   b = 1)
{
  RandomNumberDouble random_double_generator(a, b);

  for (auto &p : points)
    {
      for (unsigned int d = 0; d < dim; ++d)
        {
          p(d) = random_double_generator.generate();
        }
    }
}


/*
 * Class finds the MPI rank of the partition of the distributed triangulation
 * object that owns points.
 */
template <int dim>
class PartitionFinder
{
public:
  /*
   * Convenience typedef
   */
  using ForrestType = typename dealii::internal::p4est::types<dim>::forest;

  PartitionFinder();

  PartitionFinder(const PartitionFinder<dim> &other) = delete;

  PartitionFinder<dim> &
  operator=(const PartitionFinder<dim> &other) = delete;

  void
  generate_triangualtion(const unsigned int n_refine);

  unsigned int
  find_owner_rank(const Point<dim> &p) const;

  void
  find_owner_rank_list(const std::vector<Point<dim>> &points,
                  std::vector<unsigned int>      mpi_rank_owner) const;

private:
  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;

  TimerOutput computing_timer;

  parallel::distributed::Triangulation<dim> triangulation;

  typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell_hint;

  const bool is_periodic;
};


template <int dim>
PartitionFinder<dim>::PartitionFinder()
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , cell_hint(triangulation.begin_active())
  , is_periodic(true)
{}


template <int dim>
PartitionFinder<dim>::generate_triangualtion(const unsigned int n_refine)
{
  TimerOutput::Scope t(computing_timer, "mesh generation");

  GridGenerator::hyper_cube(triangulation,
                            0,
                            1,
                            /* colorize */ true);

  if (is_periodic)
    {
      std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id1*/ 2 * (d + 1) - 2,
                                            /*b_id2*/ 2 * (d + 1) - 1,
                                            /*direction*/ d,
                                            periodicity_vector);
        }

      triangulation.add_periodicity(periodicity_vector);
    } // if

  triangulation.refine_global(n_refine);
}


template <int dim>
void
PartitionFinder<dim>::write_mesh(const std::string &filename)
{
  pcout << "*** Writing mesh ***" << std::endl
        << "*** Dimension:           " << dim << std::endl
        << "*** No. of cells:        " << triangulation.n_active_cells()
        << std::endl;

  /*
   * Print some general mesh info
   */
  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (auto &cell : triangulation.active_cell_iterators())
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_id()]++;
          }
      }

    pcout << "*** Boundary indicators: ";
    for (const std::pair<const types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        pcout << pair.first << "(" << pair.second << " times) ";
      }
    pcout << std::endl;
  }

  /*
   * Write a vtu file for each locally owned part of the mesh
   */
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  std::string filename_slave(filename);
  filename_slave +=
    "." + Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

  std::ofstream output(filename_slave.c_str());
  data_out.write_vtu(output);


  /*
   * Write a pvtu-record to collect all files for each time step.
   */
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> file_list;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        {
          file_list.push_back(filename + "." + Utilities::int_to_string(i, 4) +
                              ".vtu");
        }

      std::string filename_master(filename);

      filename_master += ".pvtu";

      std::ofstream master_output(filename_master);
      data_out.write_pvtu_record(master_output, file_list);
    }

  pcout << "*** Written to           " << filename << ".vtu" << std::endl;
}


template <int dim>
unsigned int
PartitionFinder<dim>::find_owner_rank(const Point<dim> &p) const
{
  TimerOutput::Scope t(computing_timer,
                       "finding owner's MPI rank (single point)");

  /*
   * Get access to some p4est internals
   */
  const ForrestType *forrest = triangulation.get_p4est();



  return mpi_rank = 0;
}


template <int dim>
void
PartitionFinder<dim>::find_owner_rank_list(
  const std::vector<Point<dim>> &points,
  std::vector<unsigned int>      mpi_rank_owner) const
{
  TimerOutput::Scope t(computing_timer,
                       "finding owner's MPI rank (" +
                         Utilities::int_to_string(points.size()) + " point)");
}


/*
 * Main function
 */

int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;

      // dimension
      const int          dim = 2;
      const unsigned int n_refine = 3;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      PartitionFinder<dim> partition_finder;
      partition_finder.generate_triangualtion(n_refine);
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
