#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{
  if (size > 0)
  {
      MPI_Bcast(&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

      unsigned int numCount = ySize * xSize;
      unsigned int pSize = numCount/size;
      double* pArr = (double *)malloc(pSize * sizeof(double));
      MPI_Scatter(arr, pSize, MPI_DOUBLE,
                  pArr, pSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      for(unsigned int i = 0; i < pSize; i++) {
          pArr[i] = sin(0.00001*pArr[i]);
      }

      MPI_Gather(pArr, pSize, MPI_DOUBLE,
                 arr, pSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (rank == 0 && numCount % size > 0) {
          //если нацело распараллелить не удается, то досчитываем последний столбец
          for(unsigned int i = pSize * size; i < numCount; i++) {
              arr[i] = sin(0.00001*arr[i]);
          }
      }

      free(pArr);
  }
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
  double* arr = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
