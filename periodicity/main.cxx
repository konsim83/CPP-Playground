#include <iostream>
#include <vector>
#include <cmath>

///////////////////////////////////

int main(int argc, char **argv) {

	const double L_period = 3.0;
	double intpart;

	std::vector<double> xv = {-3.2, -0.1, 0.0, -0.0, 1.5, 3.0, 4.5};

	std::cout << "----------- Period [0, " << L_period << "] -----------" << std::endl << std::endl;

	for (const auto & x : xv)
	{
		std::cout << "x                       = " << x << std::endl
				  << "signbit(x)              = " << std::signbit(x) << std::endl
				  << "Scaled to unit interval = " << x/L_period << std::endl
				  << "Scaled fractional part  = " << modf (x/L_period , &intpart) << std::endl
				  << "Point in [0, " << L_period << "]: " << (modf (x/L_period , &intpart) + 1.0) * L_period << std::endl
				  << "Result: " << x << " % " << L_period << " = "
				  << std::fmod((modf (x/L_period , &intpart) + 1.0) * L_period, L_period) << std::endl << std::endl;
	}

	return 0;
}
