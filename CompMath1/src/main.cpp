#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

double acceleration(double t) {
    return sin(t);
}

void calc(double *trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size) {
    if (size <= 0) {
        return;
    }

    MPI_Bcast(&traceSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    uint32_t from = traceSize * rank / size;
    uint32_t to = traceSize * (rank + 1) / size;
    uint32_t len = to - from;

    double *mytrace = (double *) malloc(len * sizeof(double));


    double a_start = 0, a_end = 0;
    double v_start = 0, v_end = 0;
    double a_prev = 0, v_prev = 0;
    if (rank == 0) {
        a_start = y0;
    }

    double tau = traceSize / size * dt;
    double t_start = tau * rank;
    t0 += t_start;

    double v0 = 0;
    mytrace[0] = a_start;
    mytrace[1] = y0 + dt * v0;

    for (uint32_t i = 2; i < len; i++) {
        mytrace[i] = dt * dt * acceleration(t0 + (i - 1) * dt)
                     + 2 * mytrace[i - 1] - mytrace[i - 2];
    }
    a_end = mytrace[len - 1];
    v_end = (mytrace[len - 1] - mytrace[len - 2]) / dt;


    MPI_Status status;
    if (size > 0) {

        //теперь надо отослать
        if(rank==0){
            MPI_Send(&a_end, 1, MPI_DOUBLE,  1, 0, MPI_COMM_WORLD);
            MPI_Send(&v_end, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&a_prev, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &status);
            v0 = (y1 - a_prev) / (tau * size);
        }
        else {
            //получаем конечные a,v от предыдущего
            MPI_Recv(&a_prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&v_prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
            a_start = a_prev;
            v_start = v_prev;

            a_end += a_start + v_start * tau;
            v_end += v_start;

            if (rank == size - 1) {
                MPI_Send(&a_end, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            } else {
                MPI_Send(&a_end, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Send(&v_end, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Bcast(&v0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        a_start += v0 * t_start;
        v_start += v0;
    } else {
        v0 = (y1 - a_end) / (tau * size);
        v_start = v0;
    }

    mytrace[0] = a_start;
    mytrace[1] = a_start + dt * v_start;
    for (uint32_t i = 2; i < len; i++)
    {
        mytrace[i] = dt * dt * acceleration (t0 + (i - 1) * dt)
                         + 2 * mytrace[i - 1] - mytrace[i - 2];
    }

    if(rank>0){
        MPI_Send (mytrace, len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }else{
        memcpy (trace, mytrace, len * sizeof (double));
        for (int i = 1; i < size; i++)
        {
            uint32_t from = traceSize * i / size;
            uint32_t to = traceSize * (i + 1) / size;
            uint32_t len = to - from;
            MPI_Recv (trace + from, len, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        }
    }

    free(mytrace);
}

int main(int argc, char **argv) {
    int rank = 0, size = 0, status = 0;
    uint32_t traceSize = 0;
    double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
    double *trace = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Check arguments
        if (argc != 3) {
            std::cout << "[Error] Usage <inputfile> <output file>\n";
            status = 1;
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return 1;
        }

        // Prepare input file
        std::ifstream input(argv[1]);
        if (!input.is_open()) {
            std::cout << "[Error] Can't open " << argv[1] << " for write\n";
            status = 1;
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return 1;
        }

        // Read arguments from input
        input >> t0 >> t1 >> dt >> y0 >> y1;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        traceSize = (t1 - t0) / dt;
        trace = new double[traceSize];

        input.close();
    } else {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (status != 0) {
            return 1;
        }
    }

    calc(trace, traceSize, t0, dt, y0, y1, rank, size);

    if (rank == 0) {
        // Prepare output file
        std::ofstream output(argv[2]);
        if (!output.is_open()) {
            std::cout << "[Error] Can't open " << argv[2] << " for read\n";
            delete trace;
            return 1;
        }

        for (uint32_t i = 0; i < traceSize; i++) {
            output << " " << trace[i];
        }
        output << std::endl;
        output.close();
        delete trace;
    }

    MPI_Finalize();
    return 0;
}
