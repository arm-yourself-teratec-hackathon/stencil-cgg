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
uint64_t BLOCK_SIZE;
std::vector<double> matA;
std::vector<double> matB;
std::vector<double> matC;
std::vector<double> exponents;

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

    // Initialize the exponents array
    for (uint64_t o = 1; o <= order; ++o) {
        exponents.push_back(1.0 / pow(17.0, o));
    }
}

auto one_iteration() -> void {
    #pragma omp parallel for schedule(guided)
    for (uint64_t z = 0; z < DIMZ; z += BLOCK_SIZE) {
        for (uint64_t y = 0; y < DIMY; y += BLOCK_SIZE) {
            for (uint64_t x = 0; x < DIMX; x += BLOCK_SIZE) {
                for (uint64_t zz = z; zz < z + BLOCK_SIZE && zz < DIMZ; ++zz) {
                    for (uint64_t yy = y; yy < y + BLOCK_SIZE && yy < DIMY; ++yy) {
                        for (uint64_t xx = x; xx < x + BLOCK_SIZE && xx < DIMX; ++xx) {
                            // Pre-compute planes
                            const uint64_t xyz = DIMXYZ(xx, yy, zz);
                            const uint64_t yz = (zz + order) * xyplane + (yy + order) * MAXX + order;
                            const uint64_t xz = (zz + order) * xyplane + xx + order;
                            const uint64_t xy = (yy + order) * MAXX + xx + order;

                            // Ensure compiler pre-loads exponents with `ld1rd` in SVE regs only once
                            const double exp0 = exponents[0];
                            const double exp1 = exponents[1];
                            const double exp2 = exponents[2];
                            const double exp3 = exponents[3];
                            const double exp4 = exponents[4];
                            const double exp5 = exponents[5];
                            const double exp6 = exponents[6];
                            const double exp7 = exponents[7];

                            // Compute for current cell (o = 0)
                            matC[xyz] = matA[xyz] * matB[xyz];

                            // Compute all orders on the x axis (first positive direction, then negative one)
                            matC[xyz] += matA[(xx + 1) + yz] * matB[xx + 1 + yz] * exp0;
                            matC[xyz] += matA[(xx + 2) + yz] * matB[xx + 2 + yz] * exp1;
                            matC[xyz] += matA[(xx + 3) + yz] * matB[xx + 3 + yz] * exp2;
                            matC[xyz] += matA[(xx + 4) + yz] * matB[xx + 4 + yz] * exp3;
                            matC[xyz] += matA[(xx + 5) + yz] * matB[xx + 5 + yz] * exp4;
                            matC[xyz] += matA[(xx + 6) + yz] * matB[xx + 6 + yz] * exp5;
                            matC[xyz] += matA[(xx + 7) + yz] * matB[xx + 7 + yz] * exp6;
                            matC[xyz] += matA[(xx + 8) + yz] * matB[xx + 8 + yz] * exp7;
                            matC[xyz] += matA[(xx - 1) + yz] * matB[xx - 1 + yz] * exp0;
                            matC[xyz] += matA[(xx - 2) + yz] * matB[xx - 2 + yz] * exp1;
                            matC[xyz] += matA[(xx - 3) + yz] * matB[xx - 3 + yz] * exp2;
                            matC[xyz] += matA[(xx - 4) + yz] * matB[xx - 4 + yz] * exp3;
                            matC[xyz] += matA[(xx - 5) + yz] * matB[xx - 5 + yz] * exp4;
                            matC[xyz] += matA[(xx - 6) + yz] * matB[xx - 6 + yz] * exp5;
                            matC[xyz] += matA[(xx - 7) + yz] * matB[xx - 7 + yz] * exp6;
                            matC[xyz] += matA[(xx - 8) + yz] * matB[xx - 8 + yz] * exp7;

                            // Compute all orders on the y axis (first positive direction, then negative one)
                            matC[xyz] += matA[((yy + 1 + order) * MAXX) + xz] * matB[((yy + 1 + order) * MAXX) + xz] * exp0;
                            matC[xyz] += matA[((yy + 2 + order) * MAXX) + xz] * matB[((yy + 2 + order) * MAXX) + xz] * exp1;
                            matC[xyz] += matA[((yy + 3 + order) * MAXX) + xz] * matB[((yy + 3 + order) * MAXX) + xz] * exp2;
                            matC[xyz] += matA[((yy + 4 + order) * MAXX) + xz] * matB[((yy + 4 + order) * MAXX) + xz] * exp3;
                            matC[xyz] += matA[((yy + 5 + order) * MAXX) + xz] * matB[((yy + 5 + order) * MAXX) + xz] * exp4;
                            matC[xyz] += matA[((yy + 6 + order) * MAXX) + xz] * matB[((yy + 6 + order) * MAXX) + xz] * exp5;
                            matC[xyz] += matA[((yy + 7 + order) * MAXX) + xz] * matB[((yy + 7 + order) * MAXX) + xz] * exp6;
                            matC[xyz] += matA[((yy + 8 + order) * MAXX) + xz] * matB[((yy + 8 + order) * MAXX) + xz] * exp7;
                            matC[xyz] += matA[((yy - 1 + order) * MAXX) + xz] * matB[((yy - 1 + order) * MAXX) + xz] * exp0;
                            matC[xyz] += matA[((yy - 2 + order) * MAXX) + xz] * matB[((yy - 2 + order) * MAXX) + xz] * exp1;
                            matC[xyz] += matA[((yy - 3 + order) * MAXX) + xz] * matB[((yy - 3 + order) * MAXX) + xz] * exp2;
                            matC[xyz] += matA[((yy - 4 + order) * MAXX) + xz] * matB[((yy - 4 + order) * MAXX) + xz] * exp3;
                            matC[xyz] += matA[((yy - 5 + order) * MAXX) + xz] * matB[((yy - 5 + order) * MAXX) + xz] * exp4;
                            matC[xyz] += matA[((yy - 6 + order) * MAXX) + xz] * matB[((yy - 6 + order) * MAXX) + xz] * exp5;
                            matC[xyz] += matA[((yy - 7 + order) * MAXX) + xz] * matB[((yy - 7 + order) * MAXX) + xz] * exp6;
                            matC[xyz] += matA[((yy - 8 + order) * MAXX) + xz] * matB[((yy - 8 + order) * MAXX) + xz] * exp7;

                            // Compute all orders on the z axis (first positive direction, then negative one)
                            matC[xyz] += matA[((zz + 1 + order) * xyplane) + xy] * matB[((zz + 1 + order) * xyplane) + xy] * exp0;
                            matC[xyz] += matA[((zz + 2 + order) * xyplane) + xy] * matB[((zz + 2 + order) * xyplane) + xy] * exp1;
                            matC[xyz] += matA[((zz + 3 + order) * xyplane) + xy] * matB[((zz + 3 + order) * xyplane) + xy] * exp2;
                            matC[xyz] += matA[((zz + 4 + order) * xyplane) + xy] * matB[((zz + 4 + order) * xyplane) + xy] * exp3;
                            matC[xyz] += matA[((zz + 5 + order) * xyplane) + xy] * matB[((zz + 5 + order) * xyplane) + xy] * exp4;
                            matC[xyz] += matA[((zz + 6 + order) * xyplane) + xy] * matB[((zz + 6 + order) * xyplane) + xy] * exp5;
                            matC[xyz] += matA[((zz + 7 + order) * xyplane) + xy] * matB[((zz + 7 + order) * xyplane) + xy] * exp6;
                            matC[xyz] += matA[((zz + 8 + order) * xyplane) + xy] * matB[((zz + 8 + order) * xyplane) + xy] * exp7;
                            matC[xyz] += matA[((zz - 1 + order) * xyplane) + xy] * matB[((zz - 1 + order) * xyplane) + xy] * exp0;
                            matC[xyz] += matA[((zz - 2 + order) * xyplane) + xy] * matB[((zz - 2 + order) * xyplane) + xy] * exp1;
                            matC[xyz] += matA[((zz - 3 + order) * xyplane) + xy] * matB[((zz - 3 + order) * xyplane) + xy] * exp2;
                            matC[xyz] += matA[((zz - 4 + order) * xyplane) + xy] * matB[((zz - 4 + order) * xyplane) + xy] * exp3;
                            matC[xyz] += matA[((zz - 5 + order) * xyplane) + xy] * matB[((zz - 5 + order) * xyplane) + xy] * exp4;
                            matC[xyz] += matA[((zz - 6 + order) * xyplane) + xy] * matB[((zz - 6 + order) * xyplane) + xy] * exp5;
                            matC[xyz] += matA[((zz - 7 + order) * xyplane) + xy] * matB[((zz - 7 + order) * xyplane) + xy] * exp6;
                            matC[xyz] += matA[((zz - 8 + order) * xyplane) + xy] * matB[((zz - 8 + order) * xyplane) + xy] * exp7;
                        }
                    }
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
        BLOCK_SIZE = 8;
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
