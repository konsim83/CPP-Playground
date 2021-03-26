#pragma once

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// STL
#include <fstream>
#include <iostream>
#include <vector>

namespace Step20
{
  using namespace dealii;

  void
  generate_test_mesh_3D(Triangulation<3> &triangulation,
                        const bool        face_orientation,
                        const bool        face_flip,
                        const bool        face_rotation,
                        const bool        manipulate_first_cube)
  {
    std::vector<Point<3>>    vertices;
    const unsigned int       n_cells = 2;
    std::vector<CellData<3>> cells(n_cells);

    const Point<3> p(1, 0, 0);

    // Corner points of the cube [0,1]^3
    static const std::array<Point<3>, 12> hexahedron = {{{0, 0, 0},   // 0
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
      vertices.push_back(hexahedron[i]);

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
          //        for (unsigned int vertex_index = 0;
          //        GeometryInfo<3>::vertices_per_cell;
          //             ++vertex_index)
          {
            cells[cell_index].vertices[vertex_index] =
              cell_vertices[cell_index][vertex_index];
            cells[cell_index].material_id = 0;
          }
      }

    //    GridReordering<3>::invert_all_cells_of_negative_grid(vertices, cells);
    triangulation.create_triangulation(vertices, cells, SubCellData());
  }


  void
  generate_test_mesh_2D(Triangulation<2> &tria,
                        const bool        rotate_left_square,
                        const bool        rotate_right_square)
  {
    const unsigned int       n_cells = 2;
    std::vector<CellData<2>> cells(n_cells);

    // Corner points of the cube [0,1]^2
    const std::vector<Point<2>> vertices = {Point<2>(0, 0),  // 0
                                            Point<2>(1, 0),  // 1
                                            Point<2>(0, 1),  // 2
                                            Point<2>(1, 1),  // 3
                                            Point<2>(2, 0),  // 4
                                            Point<2>(2, 1)}; // 5


    // consistent orientation
    unsigned int cell_vertices[n_cells][4] = {{0, 1, 2, 3},  // unit cube
                                              {1, 4, 3, 5}}; // shifted cube

    // all 4 true-false combinations of (rotate_left_square | rotate_right_square) to a number 0..3
    unsigned int this_case = 2 * rotate_left_square + rotate_right_square;

    switch (this_case)
      {
          case /* rotate only right square */ 1: {
            cell_vertices[1][0] = 4;
            cell_vertices[1][1] = 5;
            cell_vertices[1][2] = 1;
            cell_vertices[1][3] = 3;
            break;
          }

          case /* rotate only left square */ 2: {
            cell_vertices[0][0] = 1;
            cell_vertices[0][1] = 3;
            cell_vertices[0][2] = 0;
            cell_vertices[0][3] = 2;
            break;
          }

          case /* rotate both squares (again consistent orientation) */ 3: {
            cell_vertices[0][0] = 1;
            cell_vertices[0][1] = 3;
            cell_vertices[0][2] = 0;
            cell_vertices[0][3] = 2;

            cell_vertices[1][0] = 4;
            cell_vertices[1][1] = 5;
            cell_vertices[1][2] = 1;
            cell_vertices[1][3] = 3;
            break;
          }

        default /* 0 */:
          break;
      } // switch

    cells.resize(n_cells, CellData<2>());

    for (unsigned int cell_index = 0; cell_index < n_cells; ++cell_index)
      {
        for (const unsigned int vertex_index :
             GeometryInfo<2>::vertex_indices())
          {
            cells[cell_index].vertices[vertex_index] =
              cell_vertices[cell_index][vertex_index];
            cells[cell_index].material_id = 0;
          }
      }

    tria.create_triangulation(vertices, cells, SubCellData());
  }
} // namespace Step20