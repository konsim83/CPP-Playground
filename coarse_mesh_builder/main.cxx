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

// Boost
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

// C
#include <stdio.h>

// STL
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <typeinfo>
#include <vector>


//-------------------------------------
//-------------------------------------
//-------------------------------------

template <int dim>
class CoarseMeshFromData
{
public:
  CoarseMeshFromData() = delete;

  CoarseMeshFromData(const std::string &data_dir_name);

  void
  generate();

  void
  refine_global(unsigned int n_refine_global);

  void
  replicate_mesh_vertically(unsigned int n_layers);

  void
  rescale_mesh_vertically(double scale_factor);

  void
  write_mesh(const std::string &filename);

private:
  struct path_leaf_string
  {
    std::string
    operator()(const boost::filesystem::directory_entry &entry) const
    {
      return entry.path().leaf().string();
    }
  };

  void
  read_directory(const std::string &name, std::vector<std::string> &file_list);

  void
  link_cell_to_data_file();

  typename dealii::Triangulation<dim>::cell_iterator
  get_root_of_tree(
    const typename dealii::Triangulation<dim>::cell_iterator &cell);

  const std::string data_dir_name;

  std::vector<std::string> data_file_list;

  std::map<dealii::CellId, std::string> cell_to_data_file_map;

  dealii::Triangulation<dim> triangulation;

  /*!
   * Number of cell layers up to a height of 2000m
   */
  unsigned int n_layers;

  bool is_initialized;

  double cell_height; // in [m]

  double cell_width; // in [m]

  const double resolution; // in [m]
};


template <int dim>
CoarseMeshFromData<dim>::CoarseMeshFromData(const std::string &_data_dir_name)
  : data_dir_name(_data_dir_name)
  , n_layers(1)
  , is_initialized(false)
  , cell_height((dim == 2 ? 0 : 2000))
  , cell_width(2000)
  , resolution(25)
{
  read_directory(data_dir_name, data_file_list);
}


template <int dim>
void
CoarseMeshFromData<dim>::read_directory(const std::string &       name,
                                        std::vector<std::string> &file_list)
{
  boost::filesystem::path               path(name);
  boost::filesystem::directory_iterator it_start_path(path);
  boost::filesystem::directory_iterator it_end_path;

  std::transform(it_start_path,
                 it_end_path,
                 std::back_inserter(file_list),
                 path_leaf_string());

#ifdef DEBUG
  std::copy(file_list.begin(),
            file_list.end(),
            std::ostream_iterator<std::string>(std::cout, "\n"));
#endif
}


template <int dim>
void
CoarseMeshFromData<dim>::generate()
{
  std::vector<dealii::Triangulation<dim>> triangulation_list(
    data_file_list.size());
  typename std::vector<dealii::Triangulation<dim>>::iterator
    it_triangulation_list     = triangulation_list.begin(),
    it_triangulation_list_end = triangulation_list.end();

  double      x, y;
  std::string delimiter = "_";

  /*
   * For each file read the first two values (lower left corner of the cell)
   */
  for (const auto &current_file_name : data_file_list)
    {
      std::string this_file_name(current_file_name);
      size_t      pos = 0;
      std::string token;
      int         i = 0;
      while ((pos = this_file_name.find(delimiter)) != std::string::npos)
        {
          if (i == 1)
            {
              token = this_file_name.substr(2, pos - 2);
              x     = std::stod(token) * 1000;
#ifdef DEBUG
              std::cout << i << "  " << token << std::endl;
#endif
              this_file_name.erase(0, pos + delimiter.length());
            }
          else if (i == 2)
            {
              token = this_file_name.substr(0, pos);
              y     = std::stod(token) * 1000;
#ifdef DEBUG
              std::cout << i << "  " << token << std::endl;
#endif
              this_file_name.erase(0, pos + delimiter.length());
            }
          else
            {
              token = this_file_name.substr(0, pos);
#ifdef DEBUG
              std::cout << i << "  " << token << std::endl;
#endif
              this_file_name.erase(0, pos + delimiter.length());
            }
          ++i;
        }

#ifdef DEBUG
      std::cout << "x = " << x << "    y = " << y << std::endl;
#endif

      dealii::Point<dim> lower_left_corner;
      lower_left_corner(0) = x;
      lower_left_corner(1) = y;

      dealii::GridGenerator::hyper_cube(*it_triangulation_list, 0, cell_width);
      dealii::GridTools::shift(lower_left_corner, *it_triangulation_list);

      // Increase triangulation iterator for each file.
      ++it_triangulation_list;
    }

  /*
   * We must record a pointer to each triangulation in the list to use the
   * merge_triangualtions function.
   */
  std::vector<const dealii::Triangulation<dim> *> triangulation_ptr_list(
    data_file_list.size());
  typename std::vector<const dealii::Triangulation<dim> *>::iterator
    it_triangulation_ptr_list     = triangulation_ptr_list.begin(),
    it_triangulation_ptr_list_end = triangulation_ptr_list.end();

  for (const auto &tria : triangulation_list)
    {
      *it_triangulation_ptr_list = &tria;
      ++it_triangulation_ptr_list;
    }

  dealii::GridGenerator::merge_triangulations(
    triangulation_ptr_list,
    triangulation,
    /* duplicated_vertex_tolerance = */ 1.0e-12,
    /* copy_manifold_ids =  */ false);

  std::cout << "Triangulations merged into one coarse mesh." << std::endl;

  link_cell_to_data_file();

  is_initialized = true;
}


template <int dim>
void
CoarseMeshFromData<dim>::refine_global(unsigned int n_refine_global)
{
  triangulation.refine_global(n_refine_global);

  link_cell_to_data_file();

  cell_height /= 2;
  cell_width /= 2;
}


template <>
void
CoarseMeshFromData<2>::replicate_mesh_vertically(
  unsigned int n_vertical_replications)
{
  std::cout << "[Info:]   No layers added to 2D triangulation";
}


template <>
void
CoarseMeshFromData<3>::replicate_mesh_vertically(
  unsigned int n_vertical_replications)
{
  Assert(is_initialized, dealii::ExcNotInitialized());
  Assert(n_layers == 1,
         dealii::ExcMessage("Triangulation can only be replicated once."));

  if (n_vertical_replications > 0)
    {
      dealii::Triangulation<3> triangulation_old;
      triangulation_old.copy_triangulation(triangulation);
      triangulation.clear();

      dealii::GridGenerator::replicate_triangulation(
        triangulation_old, {1, 1, 1 + n_vertical_replications}, triangulation);

      n_layers = n_layers + n_vertical_replications;

      std::cout << "[Info:]   Mesh replicated vertically "
                << n_vertical_replications << " times." << std::endl;

      link_cell_to_data_file();
    }
}


template <>
void
CoarseMeshFromData<2>::rescale_mesh_vertically(double scale_factor)
{
  std::cout << "[Info:]   No vertical rescaling done with 2D triangulation";
}


template <>
void
CoarseMeshFromData<3>::rescale_mesh_vertically(double scale_factor)
{
  Assert(scale_factor > 0,
         dealii::ExcMessage("Vertical scale factor must be positive"));

  std::function<dealii::Point<3>(const dealii::Point<3> &)> rescale_func =
    [scale_factor](const dealii::Point<3> &in) -> dealii::Point<3> {
    return {in[0], in[1], scale_factor * in[2]};
  };

  dealii::GridTools::transform(rescale_func, triangulation);

  std::cout << "[Info:]   Mesh rescaled vertically by a factor of "
            << scale_factor << std::endl;

  cell_height *= scale_factor;
}


template <int dim>
void
CoarseMeshFromData<dim>::link_cell_to_data_file()
{
  std::cout << "Iterating through coarse mesh to link to data files..."
            << std::endl;

  cell_to_data_file_map.clear();

  /*
   * For each cell we build the associated file from the information of the
   * first vertex.
   */
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      const typename dealii::Triangulation<dim>::cell_iterator root_of_tree =
        get_root_of_tree(cell);
      dealii::CellId root_cell_id(root_of_tree->id());
      dealii::CellId current_cell_id(cell->id());

      const dealii::Point<dim> root_first_vertex = root_of_tree->vertex(0),
                               this_first_vertex = cell->vertex(0);

#ifdef DEBUG
      std::cout << "Cell:   " << current_cell_id.to_string() << "   ";
      std::printf("First Vertex:   %.2f   %.2f   ",
                  this_first_vertex(0),
                  this_first_vertex(1));

      std::cout << "   Root cell:   " << root_cell_id.to_string() << "   ";
      std::printf("First Vertex (root):   %.2f   %.2f   ",
                  root_first_vertex(0),
                  root_first_vertex(1));
#endif

      std::string data_file =
        "DGM" + dealii::Utilities::int_to_string(resolution, 2) + "_32" +
        dealii::Utilities::to_string(root_first_vertex(0) / 1000, 1) + "_" +
        dealii::Utilities::to_string(root_first_vertex(1) / 1000, 1) +
        "_2_FHH.txt";

#ifdef DEBUG
      std::cout << "data file:   " << data_file << std::endl;
#endif

      cell_to_data_file_map.emplace(current_cell_id, data_file);
    }
}


template <int dim>
typename dealii::Triangulation<dim>::cell_iterator
CoarseMeshFromData<dim>::get_root_of_tree(
  const typename dealii::Triangulation<dim>::cell_iterator &cell)
{
  typename dealii::Triangulation<dim>::cell_iterator root_of_tree;

  root_of_tree = cell;
  while (root_of_tree->level() > 0)
    {
      root_of_tree = root_of_tree->parent();
    }

  return root_of_tree;
}


template <int dim>
void
CoarseMeshFromData<dim>::write_mesh(const std::string &filename)
{
  Assert(is_initialized, dealii::ExcNotInitialized());

  std::cout << std::endl
            << "*** Writing mesh ***" << std::endl
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

    std::cout << "*** Boundary indicators: ";
    for (const std::pair<const dealii::types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        std::cout << pair.first << "(" << pair.second << " times) ";
      }
    std::cout << std::endl;
  }

  dealii::GridOut grid_out;

  std::ofstream outstream_ucd(filename + ".inp");
  grid_out.write_ucd(triangulation, outstream_ucd);

  std::ofstream outstream_vtu(filename + ".vtu");
  grid_out.write_vtu(triangulation, outstream_vtu);

  std::ofstream outstream_vtk(filename + ".vtk");
  grid_out.write_vtk(triangulation, outstream_vtk);

  std::cout << "*** Written to:          " << filename << ".inp" << std::endl;
  std::cout << "*** Written to:          " << filename << ".vtu" << std::endl;
  std::cout << "*** Written to:          " << filename << ".vtk" << std::endl
            << std::endl;
}


//-------------------------------------
//-------------------------------------
//-------------------------------------


/*
 * Main function
 */

int
main(int argc, char *argv[])
{
  try
    {
      // dimension
      const int dim = 3;

      //      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      //        argc, argv, dealii::numbers::invalid_unsigned_int);

      std::string data_dir_name =
        "/home/ksimon/SynologyCanopiesCloudShare/data_LiDAR/dgm25_2x2km_xyz_hh";


      CoarseMeshFromData<dim> coarse_mesh_from_data(data_dir_name);
      coarse_mesh_from_data.generate();
      coarse_mesh_from_data.replicate_mesh_vertically(/* n_replications */ 4);
      coarse_mesh_from_data.rescale_mesh_vertically(0.2);
      coarse_mesh_from_data.refine_global(1);
      coarse_mesh_from_data.write_mesh(
        "Hamburg_" + dealii::Utilities::int_to_string(dim, /* digits */ 1) +
        "D");

      /*
       * Generate Hamburg mesh
       */
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
