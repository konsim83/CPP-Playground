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
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// p4est
#include <p4est.h>
#include <p4est_search.h>
#include <p8est.h>
#include <p8est_search.h>

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
  RandomNumberDouble(const double _a,
                     const double _b,
                     const bool   _same_on_all_ranks = true)
    : a(_a)
    , b(_b)
    , same_on_all_ranks(_same_on_all_ranks)
    , uniform_distribution(a, b)
    , timeSeed((
        same_on_all_ranks ?
          0.0 :
          std::chrono::high_resolution_clock::now().time_since_epoch().count()))
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
      (same_on_all_ranks ?
         0.0 :
         std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
  double a, b;

  bool same_on_all_ranks;

  std::uniform_real_distribution<double> uniform_distribution;

  uint64_t timeSeed;

  std::seed_seq seed_sequence;

  std::mt19937_64 rng;
};

//-------------------------------------
//-------------------------------------
//-------------------------------------

/*
 * Fill a list of points with random numbers between a  and b.
 */
template <int dim>
void
fill_points_randomly(std::vector<dealii::Point<dim>> &points,
                     double                           a = 0,
                     double                           b = 1)
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

//-------------------------------------
//-------------------------------------
//-------------------------------------

namespace internal
{
  template <int dim>
  struct types;

  template <>
  struct types<2>
  {
    using connectivity     = p4est_connectivity_t;
    using forest           = p4est_t;
    using tree             = p4est_tree_t;
    using quadrant         = p4est_quadrant_t;
    using topidx           = p4est_topidx_t;
    using locidx           = p4est_locidx_t;
    using gloidx           = p4est_gloidx_t;
    using balance_type     = p4est_connect_type_t;
    using ghost            = p4est_ghost_t;
    using transfer_context = p4est_transfer_context_t;
  };

  template <>
  struct types<3>
  {
    using connectivity     = p8est_connectivity_t;
    using forest           = p8est_t;
    using tree             = p8est_tree_t;
    using quadrant         = p8est_quadrant_t;
    using topidx           = p4est_topidx_t;
    using locidx           = p4est_locidx_t;
    using gloidx           = p4est_gloidx_t;
    using balance_type     = p8est_connect_type_t;
    using ghost            = p8est_ghost_t;
    using transfer_context = p8est_transfer_context_t;
  };

  template <int dim>
  struct functions;

  template <>
  struct functions<2>
  {
    static void (&search_partition)(types<2>::forest *       p4est,
                                    int                      call_post,
                                    p4est_search_partition_t quadrant_fn,
                                    p4est_search_partition_t point_fn,
                                    sc_array_t *             points);
  };

  template <>
  struct functions<3>
  {
    static void (&search_partition)(types<3>::forest *       p4est,
                                    int                      call_post,
                                    p8est_search_partition_t quadrant_fn,
                                    p8est_search_partition_t point_fn,
                                    sc_array_t *             points);
  };

  ////////////////////////////////////////////////////////////////////////

  void (&functions<2>::search_partition)(types<2>::forest *       p4est,
                                         int                      call_post,
                                         p4est_search_partition_t quadrant_fn,
                                         p4est_search_partition_t point_fn,
                                         sc_array_t *             points) =
    p4est_search_partition;


  void (&functions<3>::search_partition)(types<3>::forest *       p4est,
                                         int                      call_post,
                                         p8est_search_partition_t quadrant_fn,
                                         p8est_search_partition_t point_fn,
                                         sc_array_t *             points) =
    p8est_search_partition;

  ////////////////////////////////////////////////////////////////////////

  //  static int
  //  spheres_local_quadrant(p4est_t *         p4est,
  //                         p4est_topidx_t    which_tree,
  //                         p4est_quadrant_t *quadrant,
  //                         p4est_locidx_t    local_num,
  //                         void *            point)
  //  {
  //    return 1;
  //  }
  //
  //  static int
  //  spheres_local_point(p4est_t *         p4est,
  //                      p4est_topidx_t    which_tree,
  //                      p4est_quadrant_t *quadrant,
  //                      p4est_locidx_t    local_num,
  //                      void *            point)
  //  {
  //    return 0;
  //  }



  //    points = sc_array_new_count (sizeof (p4est_locidx_t), g->lsph);
  //    for (li = 0; li < g->lsph; ++li) {
  //      *(p4est_locidx_t *) sc_array_index_int (points, li) = li;
  //    }
  //    P4EST_INFOF ("Searching partition for %ld local spheres\n", (long)
  //    g->lsph); p4est_search_partition (g->p4est, 0,
  //    spheres_partition_quadrant,
  //                            spheres_partition_point, points);
  //    sc_array_destroy_null (&points);

} // namespace internal

//-------------------------------------
//-------------------------------------
//-------------------------------------

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

  void
  write_mesh(const std::string &filename);

  int
  find_owner_rank_p4est(const dealii::Point<dim> &p);

  int
  find_owner_rank(const dealii::Point<dim> &p);

  void
  find_owner_rank_list(const std::vector<dealii::Point<dim>> &points,
                       std::vector<int>                       mpi_rank_owner);

private:
  boost::optional<dealii::Point<dim>>
  get_reference_coordinates(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    const dealii::Point<dim> &                                    point) const;

  MPI_Comm mpi_communicator;

  dealii::parallel::distributed::Triangulation<dim> triangulation;

  dealii::FE_Q<dim> fe;

  dealii::DoFHandler<dim> dof_handler;

  dealii::ConditionalOStream pcout;

  dealii::TimerOutput computing_timer;

  const dealii::Mapping<dim> &mapping;

  typename dealii::DoFHandler<dim>::active_cell_iterator cell_hint;

  const bool is_periodic;

  bool is_initialized;
};


template <int dim>
PartitionFinder<dim>::PartitionFinder()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))
  , fe(1)
  , dof_handler(triangulation)
  , pcout(std::cout,
          (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    dealii::TimerOutput::summary,
                    dealii::TimerOutput::wall_times)
  , mapping(dealii::StaticMappingQ1<dim>::mapping)
  , cell_hint()
  , is_periodic(false)
  , is_initialized(false)
{}


template <int dim>
void
PartitionFinder<dim>::generate_triangualtion(const unsigned int n_refine)
{
  dealii::TimerOutput::Scope t(computing_timer, "mesh generation");

  dealii::GridGenerator::hyper_cube(triangulation,
                                    0,
                                    1,
                                    /* colorize */ true);

  if (is_periodic)
    {
      std::vector<dealii::GridTools::PeriodicFacePair<
        typename dealii::parallel::distributed::Triangulation<
          dim>::cell_iterator>>
        periodicity_vector;

      for (unsigned int d = 0; d < dim; ++d)
        {
          dealii::GridTools::collect_periodic_faces(triangulation,
                                                    /*b_id1*/ 2 * (d + 1) - 2,
                                                    /*b_id2*/ 2 * (d + 1) - 1,
                                                    /*direction*/ d,
                                                    periodicity_vector);
        }

      triangulation.add_periodicity(periodicity_vector);
    } // if

  triangulation.refine_global(n_refine);

  //  cell_hint = triangulation.begin_active();

  dof_handler.distribute_dofs(fe);

  cell_hint = dof_handler.begin_active();

  is_initialized = true;
}


template <int dim>
void
PartitionFinder<dim>::write_mesh(const std::string &filename)
{
  Assert(is_initialized, dealii::ExcNotInitialized());

  dealii::TimerOutput::Scope t(computing_timer, "write mesh");

  pcout << std::endl
        << "*** Writing mesh ***" << std::endl
        << "*** MPI ranks used       "
        << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)
        << std::endl
        << "*** Dimension:           " << dim << std::endl
        << "*** No. of cells:        " << triangulation.n_active_cells()
        << std::endl;

  /*
   * Print some general mesh info
   */
  {
    std::map<dealii::types::boundary_id, unsigned int> boundary_count;
    for (auto &cell : triangulation.active_cell_iterators())
      {
        for (unsigned int face = 0;
             face < dealii::GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_id()]++;
          }
      }

    pcout << "*** Boundary indicators: ";
    for (const std::pair<const dealii::types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        pcout << pair.first << "(" << pair.second << " times) ";
      }
    pcout << std::endl;
  }

  dealii::GridOut grid_out;

  grid_out.write_mesh_per_processor_as_vtu(triangulation,
                                           filename,
                                           /* view_levels */ false,
                                           /* include_artificials */ false);

  pcout << "*** Written to:          " << filename << ".pvtu" << std::endl
        << std::endl;
}


template <int dim>
int
PartitionFinder<dim>::find_owner_rank_p4est(const dealii::Point<dim> &p)
{
  Assert(is_initialized, dealii::ExcNotInitialized());

  dealii::Timer timer;
  timer.restart();

  /*
   * Get access to some p4est internals
   */
  const ForrestType *forrest = triangulation.get_p4est();

  int mpi_rank = -1;



  timer.stop();
  std::cout << "---> MPI rank   "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            << "   search for point   " << p << " ....."
            << " done in   " << timer.cpu_time() << "   seconds.   "
            << " P4EST found owner rank   " << mpi_rank << std::endl;

  return mpi_rank;
}


template <int dim>
int
PartitionFinder<dim>::find_owner_rank(const dealii::Point<dim> &p)
{
  Assert(is_initialized, dealii::ExcNotInitialized());

  dealii::Timer timer;
  timer.restart();

  /*
   * Get access to some p4est internals
   */
  const ForrestType *forrest = triangulation.get_p4est();

  int mpi_rank = -1;

  typename dealii::DoFHandler<dim>::active_cell_iterator cell = cell_hint;
  if (cell == dof_handler.end())
    {
      cell = dof_handler.begin_active();
    }

  boost::optional<dealii::Point<dim>> qp =
    get_reference_coordinates(cell_hint, p);
  if (qp)
    {
      /*
       * If point is found to in the (locally owned) cell = cell_hint
       * return this_mpi_rank
       */
      const dealii::CellId &cell_id = cell->id();

      mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    }
  else
    {
      /*
       * If point is not found to in the (locally owned) cell = cell_hint
       * search for it on this processor.
       */
      const std::pair<
        typename dealii::internal::
          ActiveCellIterator<dim, dim, dealii::DoFHandler<dim>>::type,
        dealii::Point<dim>>
        my_pair = dealii::GridTools::find_active_cell_around_point(mapping,
                                                                   dof_handler,
                                                                   p);

      cell = my_pair.first;
      qp   = my_pair.second;
    }

  if (cell->is_locally_owned())
    {
      /*
       * If the cell found on this processor is neither ghost not artificial
       * then return this_mpi_rank.
       */
      mpi_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    }
  else
    {
      /*
       * Point is not owned by a local cell.
       */
      mpi_rank = -1;
    }

  timer.stop();
  std::cout << "---> MPI rank   "
            << dealii::Utilities::MPI::this_mpi_process(mpi_communicator)
            << "   search for point   " << p << " ....."
            << " done in   " << timer.cpu_time() << "   seconds.   "
            << " Found cell_id   " << cell->id() << "   and owner rank   "
            << mpi_rank << std::endl;

  return mpi_rank;
}


template <int dim>
void
PartitionFinder<dim>::find_owner_rank_list(
  const std::vector<dealii::Point<dim>> &points,
  std::vector<int>                       mpi_rank_owner)
{
  Assert(is_initialized, dealii::ExcNotInitialized());

  dealii::TimerOutput::Scope t(
    computing_timer,
    "finding owner's MPI rank (" +
      dealii::Utilities::int_to_string(points.size()) + " point)");
}


template <int dim>
boost::optional<dealii::Point<dim>>
PartitionFinder<dim>::get_reference_coordinates(
  const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
  const dealii::Point<dim> &                                    point) const
{
  try
    {
      dealii::Point<dim> qp = mapping.transform_real_to_unit_cell(cell, point);
      if (dealii::GeometryInfo<dim>::is_inside_unit_cell(qp))
        return qp;
      else
        return boost::optional<dealii::Point<dim>>();
    }
  catch (const typename dealii::Mapping<dim>::ExcTransformationFailed &)
    {
      // transformation failed, so
      // assume the point is
      // outside
      return boost::optional<dealii::Point<dim>>();
    }
}


/*
 * Main function
 */

int
main(int argc, char *argv[])
{
  try
    {
      // dimension
      const int          dim      = 2;
      const unsigned int n_refine = 4;

      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      PartitionFinder<dim> partition_finder;
      partition_finder.generate_triangualtion(n_refine);
      partition_finder.write_mesh("my_triangulation");

      /*
       * Do dome work on process zero only.
       */
      //      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        // generate a number of points
        unsigned int                    N = 3;
        std::vector<dealii::Point<dim>> test_points(N, dealii::Point<dim>());

        fill_points_randomly(test_points);

        for (auto &p : test_points)
          {
            std::cout << "Rank:   "
                      << dealii::Utilities::MPI::this_mpi_process(
                           MPI_COMM_WORLD)
                      << "    Point:   " << p << std::endl;
          }
        std::cout << std::endl;

        partition_finder.find_owner_rank(test_points[0]);
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
