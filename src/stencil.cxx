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
constexpr uint64_t BLOCK_SIZE = 125;

// Global variables declarations
uint64_t DIMX, DIMY, DIMZ, iters;
uint64_t MAXX, MAXY, MAXZ;
uint64_t xyplane, MATsize;
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
    for (uint64_t z = 0; z < DIMZ; ++z) {
        for (uint64_t y = 0; y < DIMY; ++y) {
            for (uint64_t x = 0; x < DIMX; ++x) {
                for (uint64_t zz = z; zz < z + BLOCK_SIZE; zz += BLOCK_SIZE) {
                    for (uint64_t yy = y; yy < y + BLOCK_SIZE; yy += BLOCK_SIZE) {
#pragma omp simd
                        for (uint64_t xx = x; xx < x + BLOCK_SIZE; xx += BLOCK_SIZE) {
                            const uint64_t xxyyzz = DIMXYZ(xx, yy, zz);

                            const uint64_t yyzz =
                                (zz + order) * xyplane + (yy + order) * MAXX + order;
                            const uint64_t xxzz = (zz + order) * xyplane + xx + order;
                            const uint64_t xxyy = (yy + order) * MAXX + xx + order;
                            matC[xxyyzz] = matA[xxyyzz] * matB[xxyyzz];

                            matC[xxyyzz] +=
                                matA[xx + 1 + yyzz] * matB[xx + 1 + yyzz] * exponents[0];
                            matC[xxyyzz] +=
                                matA[xx + 2 + yyzz] * matB[xx + 2 + yyzz] * exponents[1];
                            matC[xxyyzz] +=
                                matA[xx + 3 + yyzz] * matB[xx + 3 + yyzz] * exponents[2];
                            matC[xxyyzz] +=
                                matA[xx + 4 + yyzz] * matB[xx + 4 + yyzz] * exponents[3];
                            matC[xxyyzz] +=
                                matA[xx + 5 + yyzz] * matB[xx + 5 + yyzz] * exponents[4];
                            matC[xxyyzz] +=
                                matA[xx + 6 + yyzz] * matB[xx + 6 + yyzz] * exponents[5];
                            matC[xxyyzz] +=
                                matA[xx + 7 + yyzz] * matB[xx + 7 + yyzz] * exponents[6];
                            matC[xxyyzz] +=
                                matA[xx + 8 + yyzz] * matB[xx + 8 + yyzz] * exponents[7];
                            matC[xxyyzz] +=
                                matA[xx - 1 + yyzz] * matB[xx - 1 + yyzz] * exponents[0];
                            matC[xxyyzz] +=
                                matA[xx - 2 + yyzz] * matB[xx - 2 + yyzz] * exponents[1];
                            matC[xxyyzz] +=
                                matA[xx - 3 + yyzz] * matB[xx - 3 + yyzz] * exponents[2];
                            matC[xxyyzz] +=
                                matA[xx - 4 + yyzz] * matB[xx - 4 + yyzz] * exponents[3];
                            matC[xxyyzz] +=
                                matA[xx - 5 + yyzz] * matB[xx - 5 + yyzz] * exponents[4];
                            matC[xxyyzz] +=
                                matA[xx - 6 + yyzz] * matB[xx - 6 + yyzz] * exponents[5];
                            matC[xxyyzz] +=
                                matA[xx - 7 + yyzz] * matB[xx - 7 + yyzz] * exponents[6];
                            matC[xxyyzz] +=
                                matA[xx - 8 + yyzz] * matB[xx - 8 + yyzz] * exponents[7];

                            matC[xxyyzz] += matA[((yy + 1 + order) * MAXX) + xxzz] *
                                            matB[((yy + 1 + order) * MAXX) + xxzz] * exponents[0];
                            matC[xxyyzz] += matA[((yy + 2 + order) * MAXX) + xxzz] *
                                            matB[((yy + 2 + order) * MAXX) + xxzz] * exponents[1];
                            matC[xxyyzz] += matA[((yy + 3 + order) * MAXX) + xxzz] *
                                            matB[((yy + 3 + order) * MAXX) + xxzz] * exponents[2];
                            matC[xxyyzz] += matA[((yy + 4 + order) * MAXX) + xxzz] *
                                            matB[((yy + 4 + order) * MAXX) + xxzz] * exponents[3];
                            matC[xxyyzz] += matA[((yy + 5 + order) * MAXX) + xxzz] *
                                            matB[((yy + 5 + order) * MAXX) + xxzz] * exponents[4];
                            matC[xxyyzz] += matA[((yy + 6 + order) * MAXX) + xxzz] *
                                            matB[((yy + 6 + order) * MAXX) + xxzz] * exponents[5];
                            matC[xxyyzz] += matA[((yy + 7 + order) * MAXX) + xxzz] *
                                            matB[((yy + 7 + order) * MAXX) + xxzz] * exponents[6];
                            matC[xxyyzz] += matA[((yy + 8 + order) * MAXX) + xxzz] *
                                            matB[((yy + 8 + order) * MAXX) + xxzz] * exponents[7];
                            matC[xxyyzz] += matA[((yy - 1 + order) * MAXX) + xxzz] *
                                            matB[((yy - 1 + order) * MAXX) + xxzz] * exponents[0];
                            matC[xxyyzz] += matA[((yy - 2 + order) * MAXX) + xxzz] *
                                            matB[((yy - 2 + order) * MAXX) + xxzz] * exponents[1];
                            matC[xxyyzz] += matA[((yy - 3 + order) * MAXX) + xxzz] *
                                            matB[((yy - 3 + order) * MAXX) + xxzz] * exponents[2];
                            matC[xxyyzz] += matA[((yy - 4 + order) * MAXX) + xxzz] *
                                            matB[((yy - 4 + order) * MAXX) + xxzz] * exponents[3];
                            matC[xxyyzz] += matA[((yy - 5 + order) * MAXX) + xxzz] *
                                            matB[((yy - 5 + order) * MAXX) + xxzz] * exponents[4];
                            matC[xxyyzz] += matA[((yy - 6 + order) * MAXX) + xxzz] *
                                            matB[((yy - 6 + order) * MAXX) + xxzz] * exponents[5];
                            matC[xxyyzz] += matA[((yy - 7 + order) * MAXX) + xxzz] *
                                            matB[((yy - 7 + order) * MAXX) + xxzz] * exponents[6];
                            matC[xxyyzz] += matA[((yy - 8 + order) * MAXX) + xxzz] *
                                            matB[((yy - 8 + order) * MAXX) + xxzz] * exponents[7];

                            matC[xxyyzz] += matA[((zz + 1 + order) * xyplane) + xxyy] *
                                            matB[((zz + 1 + order) * xyplane) + xxyy] *
                                            exponents[0];
                            matC[xxyyzz] += matA[((zz + 2 + order) * xyplane) + xxyy] *
                                            matB[((zz + 2 + order) * xyplane) + xxyy] *
                                            exponents[1];
                            matC[xxyyzz] += matA[((zz + 3 + order) * xyplane) + xxyy] *
                                            matB[((zz + 3 + order) * xyplane) + xxyy] *
                                            exponents[2];
                            matC[xxyyzz] += matA[((zz + 4 + order) * xyplane) + xxyy] *
                                            matB[((zz + 4 + order) * xyplane) + xxyy] *
                                            exponents[3];
                            matC[xxyyzz] += matA[((zz + 5 + order) * xyplane) + xxyy] *
                                            matB[((zz + 5 + order) * xyplane) + xxyy] *
                                            exponents[4];
                            matC[xxyyzz] += matA[((zz + 6 + order) * xyplane) + xxyy] *
                                            matB[((zz + 6 + order) * xyplane) + xxyy] *
                                            exponents[5];
                            matC[xxyyzz] += matA[((zz + 7 + order) * xyplane) + xxyy] *
                                            matB[((zz + 7 + order) * xyplane) + xxyy] *
                                            exponents[6];
                            matC[xxyyzz] += matA[((zz + 8 + order) * xyplane) + xxyy] *
                                            matB[((zz + 8 + order) * xyplane) + xxyy] *
                                            exponents[7];
                            matC[xxyyzz] += matA[((zz - 1 + order) * xyplane) + xxyy] *
                                            matB[((zz - 1 + order) * xyplane) + xxyy] *
                                            exponents[0];
                            matC[xxyyzz] += matA[((zz - 2 + order) * xyplane) + xxyy] *
                                            matB[((zz - 2 + order) * xyplane) + xxyy] *
                                            exponents[1];
                            matC[xxyyzz] += matA[((zz - 3 + order) * xyplane) + xxyy] *
                                            matB[((zz - 3 + order) * xyplane) + xxyy] *
                                            exponents[2];
                            matC[xxyyzz] += matA[((zz - 4 + order) * xyplane) + xxyy] *
                                            matB[((zz - 4 + order) * xyplane) + xxyy] *
                                            exponents[3];
                            matC[xxyyzz] += matA[((zz - 5 + order) * xyplane) + xxyy] *
                                            matB[((zz - 5 + order) * xyplane) + xxyy] *
                                            exponents[4];
                            matC[xxyyzz] += matA[((zz - 6 + order) * xyplane) + xxyy] *
                                            matB[((zz - 6 + order) * xyplane) + xxyy] *
                                            exponents[5];
                            matC[xxyyzz] += matA[((zz - 7 + order) * xyplane) + xxyy] *
                                            matB[((zz - 7 + order) * xyplane) + xxyy] *
                                            exponents[6];
                            matC[xxyyzz] += matA[((zz - 8 + order) * xyplane) + xxyy] *
                                            matB[((zz - 8 + order) * xyplane) + xxyy] *
                                            exponents[7];
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
