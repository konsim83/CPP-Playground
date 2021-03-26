#include <deal.II/fe/fe_abf.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_bernardi_raugel.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_values.h>


// STL
#include <fstream>
#include <iostream>

// My test headers
#include <fe_conformty_test.hpp>

#define PRECISION 8

int
main(int, char **)
{
#ifdef DEBUG
  dealii::MultithreadInfo::set_thread_limit(1);
#endif

  std::ofstream logfile("output");
  dealii::deallog << std::setprecision(PRECISION);
  dealii::deallog << std::fixed;
  logfile << std::setprecision(PRECISION);
  logfile << std::fixed;
  dealii::deallog.attach(logfile);

  try
    {
      using namespace FEConforimityTest;
      constexpr int      dim       = 2;
      const unsigned int fe_degree = 3;

      // H1 conformal
      // FE_Q<dim> fe(fe_degree);

      // H(div) conformal
      // FE_BDM<dim> fe(fe_degree);
      FE_ABF<dim> fe(fe_degree);
      // FE_BernardiRaugel<dim> fe(fe_degree);
      // FE_RaviartThomas<dim> fe(fe_degree);

      // H(curl) conformal
      // FE_Nedelec<dim> fe(fe_degree);
      // FE_NedelecSZ<dim> fe(fe_degree);

      {
        for (unsigned int this_switch = 0; this_switch < (dim == 2 ? 4 : 8);
             ++this_switch)
          {
            FEConformityTest<dim> fe_conformity_tester(fe, this_switch);
            fe_conformity_tester.run();
          }
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
