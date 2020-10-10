#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>

double calc(uint32_t x_last, uint32_t num_threads) {
    int fct = 1;
    double *facts = (double *) malloc(sizeof(double) * x_last);
    facts[0] = 1;
    for (uint32_t i = 1; i <= x_last; i++) {
        fct *= i;
        facts[i] = fct;
    }
    double res = 0;
#pragma omp parallel for num_threads(num_threads) reduction(+:res)
    {
        for (uint32_t i = x_last - 1; i >= 0; i--) {
            res += 1 / (double) facts[i];
        }
    }
    free(facts);
    return res;
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
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;

  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
