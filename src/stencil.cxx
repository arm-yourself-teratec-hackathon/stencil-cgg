#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <vector>

// Functions forward-declarations
[[nodiscard]] auto main(int32_t argc, char** argv) -> int32_t;
auto init() -> void;
auto one_iteration() -> void;
[[nodiscard]] auto DIMXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t;
[[nodiscard]] auto MATXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t;
[[nodiscard]] auto dml_micros() -> double;

// Constant expressions declarations
constexpr uint64_t order = 8;
constexpr double ONE_MILLION = 1000000.0;

// Global variables declarations
uint64_t DIMX, DIMY, DIMZ, iters;
uint64_t MAXX, MAXY, MAXZ;
uint64_t xyplane, MATsize;
std::vector<double> matA;
std::vector<double> matB;
std::vector<double> matC;

/// Get current time in microseconds.
[[nodiscard]] auto dml_micros() -> double {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec * ONE_MILLION + tv.tv_usec;
}

/// Returns an offset in the center of a matrix of linear dimensions [0:DIM-1].
[[nodiscard]] auto DIMXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t {
    return ((z + order) * xyplane + (y + order) * MAXX + x + order);
}

/// Returns an offset in a matrix of linear dimensions [-order:DIM+order-1] but
/// in indices of [0:DIM+order*2-1].
[[nodiscard]] auto MATXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t {
    return (x + y * MAXX + z * xyplane);
}

/// The initialization isn't part of the exercise but can be optimized nonetheless.
/// It is however not accounted for in the performance evaluation as it artificially
/// fills the matrices with constant data and does not influence performance.
///
/// The A and C matrices are initialized to zero, with A the input and C the output.
auto init() -> void {
    // Dynamically allocate memory of size DIMX * DIMY * DIMZ + ghost region on 6 faces
    matA = std::vector<double>(MATsize, 0.0);
    matB = std::vector<double>(MATsize, 0.0);
    matC = std::vector<double>(MATsize, 0.0);

    // Center and edges initialization, B is a constant stencil for the run
    for (uint64_t z = 0; z < MAXZ; ++z) {
        for (uint64_t y = 0; y < MAXY; ++y) {
            for (uint64_t x = 0; x < MAXX; ++x) {
                matB[MATXYZ(x, y, z)] = sin(z * cos(x + 0.311) * cos(y + 0.817) + 0.613);
            }
        }
    }

    // Initialize the center of A, which is the data matrix
    for (uint64_t z = 0; z < DIMZ; ++z) {
        for (uint64_t y = 0; y < DIMY; ++y) {
            for (uint64_t x = 0; x < DIMX; ++x) {
                matA[DIMXYZ(x, y, z)] = 1.0;
            }
        }
    }
}

auto one_iteration() -> void {
    #pragma omp parallel for schedule(guided)
    for (uint64_t z = 0; z < DIMZ; ++z) {
        for (uint64_t y = 0; y < DIMY; ++y) {
            #pragma omp simd
            for (uint64_t x = 0; x < DIMX; ++x) {
                uint64_t xyz = DIMXYZ(x, y, z);
                matC[xyz] = matA[xyz] * matB[xyz];

                for (uint64_t o = 1; o <= order; ++o) {
                    const double exponent = 1.0 / pow(17.0, o);
                    const uint64_t xpo_yz = DIMXYZ(x + o, y, z);
                    const uint64_t xmo_yz = DIMXYZ(x - o, y, z);
                    const uint64_t x_ypo_z = DIMXYZ(x, y + o, z);
                    const uint64_t x_ymo_z = DIMXYZ(x, y - o, z);
                    const uint64_t xy_zpo = DIMXYZ(x, y, z + o);
                    const uint64_t xy_zmo = DIMXYZ(x, y, z - o);

                    matC[xyz] += matA[xpo_yz] * matB[xpo_yz] * exponent;
                    matC[xyz] += matA[xmo_yz] * matB[xmo_yz] * exponent;
                    matC[xyz] += matA[x_ypo_z] * matB[x_ypo_z] * exponent;
                    matC[xyz] += matA[x_ymo_z] * matB[x_ymo_z] * exponent;
                    matC[xyz] += matA[xy_zpo] * matB[xy_zpo] * exponent;
                    matC[xyz] += matA[xy_zmo] * matB[xy_zmo] * exponent;
                }
            }
        }
    }

    // A = C
    memcpy(matA.data(), matC.data(), MATsize * sizeof(double));
}

[[nodiscard]] auto main(int32_t argc, char** argv) -> int32_t {
    try {
        DIMX = std::stoi(argv[1]);
        DIMY = std::stoi(argv[2]);
        DIMZ = std::stoi(argv[3]);
        iters = std::stoi(argv[4]);
        MAXX = DIMX + 2 * order;
        MAXY = DIMY + 2 * order;
        MAXZ = DIMZ + 2 * order;
        xyplane = MAXX * MAXY;
        MATsize = MAXX * MAXY * MAXZ;
    } catch (...) {
        std::cout << argv[0] << " siseX sizeY sizeZ iters" << std::endl;
        return -1;
    }

    init();
    for (uint64_t i = 0; i < iters; ++i) {
        // Compute one iteration of Jacobi: C = B@A
        double t1 = dml_micros();
        one_iteration();
        double t2 = dml_micros();

        printf("_0_ ");
        for (uint64_t j = 0; j < 5; ++j) {
            printf("%18.15lf ", matA[DIMXYZ(DIMX / 2 + j, DIMY / 2 + j, DIMZ / 2 + j)]);
        }
        double ns_point = (t2 - t1) * 1000.0 / DIMX / DIMY / DIMZ;
        printf("\033[1m%10.0lf\033[0m %10.3lf %lu %lu %lu\n", t2 - t1, ns_point, DIMX, DIMY, DIMZ);
    }

    return 0;
}
