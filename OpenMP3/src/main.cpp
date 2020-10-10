#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>

double func(double x)
{
  return sin(x);
}

double calc_chunk(double a, double b, double dx) {
    double res = 0;
    double i = a;
    while (i < b) {
        double x = i + dx / 2;
        res += dx * func(x);
        i += dx;
    }
    return res;
}

double calc(double x0, double x1, double dx, uint32_t num_threads) {
    double dx_thread = (x1 - x0) / num_threads;
    double res = 0;
#pragma omp parallel for num_threads(num_threads) reduction(+:res)
    {
        for (uint32_t i = 0; i < num_threads; i++) {
            res += calc_chunk(x0 + i * dx_thread, x0 + (i + 1) * dx_thread, dx);
        }
    }
    return round(res);
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

  // Read arguments from input
  double x0 = 0.0, x1 =0.0, dx = 0.0;
  uint32_t num_threads = 0;
  input >> x0 >> x1 >> dx >> num_threads;

  // Calculation
  double res = calc(x0, x1, dx, num_threads);

  // Write result
  output << std::setprecision(13) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
