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

  const std::string data_dir_name;

  std::vector<std::string> data_file_list;

  dealii::Triangulation<dim> triangulation;

  bool is_initialized;

  const double cell_width = 2000; // in [m]

  const bool print_data_file_list = false;
};


template <int dim>
CoarseMeshFromData<dim>::CoarseMeshFromData(const std::string &_data_dir_name)
  : data_dir_name(_data_dir_name)
  , is_initialized(false)
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

  if (print_data_file_list)
    {
      std::copy(file_list.begin(),
                file_list.end(),
                std::ostream_iterator<std::string>(std::cout, "\n"));
    }
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
              std::cout << i << "  " << token << std::endl;
              this_file_name.erase(0, pos + delimiter.length());
            }
          else if (i == 2)
            {
              token = this_file_name.substr(0, pos);
              y     = std::stod(token) * 1000;
              std::cout << i << "  " << token << std::endl;
              this_file_name.erase(0, pos + delimiter.length());
            }
          else
            {
              token = this_file_name.substr(0, pos);
              std::cout << i << "  " << token << std::endl;
              this_file_name.erase(0, pos + delimiter.length());
            }
          ++i;
        }
      std::cout << "x = " << x << "    y = " << y << std::endl;

      dealii::Point<dim> lower_left_corner(x, y);
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

  is_initialized = true;

  //  /*
  //   * For each file read the first two values (lower left corner of the cell)
  //   */
  //  for (const auto &current_file_name : data_file_list)
  //    {
  //      const std::string path_plus_current_file_name =
  //        data_dir_name + "/" + current_file_name;
  //      std::ifstream current_file(path_plus_current_file_name);
  //      if (current_file.is_open())
  //        {
  //          current_file >> x >> y;
  //          current_file.close();
  //
  //          dealii::Point<dim> lower_left_corner(x, y);
  //          dealii::GridGenerator::hyper_cube(*it_triangulation_list,
  //                                            0,
  //                                            cell_width);
  //          dealii::GridTools::shift(lower_left_corner,
  //          *it_triangulation_list);
  //        }
  //      else
  //        {
  //          std::cout << "Unable to open file: " <<
  //          path_plus_current_file_name
  //                    << std::endl;
  //        }
  //
  //      // Increase triangulation iterator for each file.
  //      ++it_triangulation_list;
  //    }
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

  std::cout << "*** Written to:          " << filename << ".inp" << std::endl;
  std::cout << "*** Written to:          " << filename << ".vtu" << std::endl
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
      const int dim = 2;

      //      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
      //        argc, argv, dealii::numbers::invalid_unsigned_int);

      std::string data_dir_name =
        "/home/ksimon/SynologyCanopiesCloudShare/data_LiDAR/dgm25_2x2km_xyz_hh";


      CoarseMeshFromData<dim> coarse_mesh_from_data(data_dir_name);
      coarse_mesh_from_data.generate();
      coarse_mesh_from_data.write_mesh("HH_triangulation");

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
