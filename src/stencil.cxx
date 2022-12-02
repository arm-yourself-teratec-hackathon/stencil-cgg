#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <vector>

// Functions forward-declarations
// [[nodiscard]] auto main(int32_t argc, char** argv) -> int32_t;
[[nodiscard]] auto main() -> int32_t;
auto init() -> void;
auto one_iteration() -> void;
[[nodiscard]] auto DIMXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t;
[[nodiscard]] auto MATXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t;
[[nodiscard]] auto dml_micros() -> double;

// Pre-processor defines so users can override at compile-time
#ifndef iters
#    define iters 5
#endif
#ifndef DIMX
#    define DIMX 500UL
#endif
#ifndef DIMY
#    define DIMY 500UL
#endif
#ifndef DIMZ
#    define DIMZ 500UL
#endif
#ifndef BLOCK_SIZE
#    define BLOCK_SIZE 20UL
#endif

// Constant expressions declarations
constexpr double ONE_MILLION = 1000000.0;
constexpr uint64_t order = 8;
constexpr uint64_t MAXX = DIMX + 2 * order;
constexpr uint64_t MAXY = DIMY + 2 * order;
constexpr uint64_t MAXZ = DIMZ + 2 * order;
constexpr uint64_t xyplane = MAXX * MAXY;
constexpr uint64_t MATsize = MAXX * MAXY * MAXZ;
// constexpr uint64_t BLOCK_SIZE = 20;

// Global variables declarations
// Dynamically allocate memory of size DIMX * DIMY * DIMZ + ghost region on 6 faces
std::vector<double> matA(MATsize, 0.0);
std::vector<double> matB(MATsize, 0.0);
std::vector<double> matAB(MATsize, 0.0);
std::vector<double> matC(MATsize, 0.0);
std::vector<double> exponents;

/// Get current time in microseconds.
[[nodiscard]] inline auto dml_micros() -> double {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec * ONE_MILLION + tv.tv_usec;
}

/// Returns an offset in the center of a matrix of linear dimensions [0:DIM-1].
[[nodiscard]] inline auto DIMXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t {
    return ((z + order) * xyplane + (y + order) * MAXX + x + order);
}

/// Returns an offset in a matrix of linear dimensions [-order:DIM+order-1] but
/// in indices of [0:DIM+order*2-1].
[[nodiscard]] inline auto MATXYZ(uint64_t x, uint64_t y, uint64_t z) -> uint64_t {
    return (x + y * MAXX + z * xyplane);
}

/// The initialization isn't part of the exercise but can be optimized nonetheless.
/// It is however not accounted for in the performance evaluation as it artificially
/// fills the matrices with constant data and does not influence performance.
///
/// The A and C matrices are initialized to zero, with A the input and C the output.
auto init() -> void {
    // Center and edges initialization, B is a constant stencil for the run
    #pragma omp parallel
    {
        #pragma omp for
        for (uint64_t z = 0; z < MAXZ; ++z) {
            for (uint64_t y = 0; y < MAXY; ++y) {
                #pragma omp simd
                for (uint64_t x = 0; x < MAXX; ++x) {
                    matB[MATXYZ(x, y, z)] = sin(z * cos(x + 0.311) * cos(y + 0.817) + 0.613);
                }
            }
        }

        // Initialize the center of A, which is the data matrix
        #pragma omp for
        for (uint64_t z = 0; z < DIMZ; ++z) {
            for (uint64_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (uint64_t x = 0; x < DIMX; ++x) {
                    matA[DIMXYZ(x, y, z)] = 1.0;
                }
            }
        }
    }

    // Initialize the exponents array
    for (uint64_t o = 1; o <= order; ++o) {
        exponents.push_back(1.0 / pow(17.0, o));
    }
}

auto one_iteration() -> void {

    // #pragma omp parallel for schedule(dynamic)
    // for (uint64_t bz = 0; bz < DIMZ; bz += BLOCK_SIZE) {
    //     for (uint64_t by = 0; by < DIMY; by += BLOCK_SIZE) {
    //         for (uint64_t bx = 0; bx < DIMX; bx += BLOCK_SIZE) {
    //             for (uint64_t z = bz; z < bz + BLOCK_SIZE; ++z) {

    //                 for (uint64_t y = by; y < by + BLOCK_SIZE; ++y) {
    //                     // #pragma omp simd
    //                     for (uint64_t x = bx; x < bx + BLOCK_SIZE; ++x) {
    //                         const uint64_t xyz = DIMXYZ(x, y, z);
    //                         matAB[xyz] = matA[xyz] * matB[xyz];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    #pragma omp parallel for schedule(dynamic)
    for (uint64_t z = 0; z < DIMZ; ++z) {
        for (uint64_t y = 0; y < DIMY; ++y) {
            #pragma omp simd
            for (uint64_t x = 0; x < DIMX; ++x) {
                const uint64_t xyz = DIMXYZ(x, y, z);
                matAB[xyz] = matA[xyz] * matB[xyz];
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (uint64_t bz = 0; bz < DIMZ; bz += 8) {
        for (uint64_t by = 0; by < DIMY; by += 8) {
            for (uint64_t bx = 0; bx < DIMX; bx += 20) {
                // #pragma omp simd
                for (uint64_t z = bz; z < bz + 8; ++z) {
                    for (uint64_t y = by; y < by + 8; ++y) {
                        for (uint64_t x = bx; x < bx + 20; ++x) {
                            // Pre-compute planes
                            const uint64_t xyz = DIMXYZ(x, y, z);
                            const uint64_t yz = (z + order) * xyplane + (y + order) * MAXX + order;
                            const uint64_t xz = (z + order) * xyplane + x + order;
                            const uint64_t xy = (y + order) * MAXX + x + order;

                            // Ensure compiler pre-loads exponents with `ld1rd` in SVE regs only
                            // once
                            const double exp0 = exponents[0];
                            const double exp1 = exponents[1];
                            const double exp2 = exponents[2];
                            const double exp3 = exponents[3];
                            const double exp4 = exponents[4];
                            const double exp5 = exponents[5];
                            const double exp6 = exponents[6];
                            const double exp7 = exponents[7];

                            // Get `matC[xyz]` into temporary
                            double matC_xyz = matC[xyz];

                            // Compute for current cell (o = 0)
                            matC_xyz = matAB[xyz];

                            // Compute all orders on the x axis (first positive direction, then
                            // negative one)
                            matC_xyz += matAB[x + 1 + yz] * exp0;
                            matC_xyz += matAB[x + 2 + yz] * exp1;
                            matC_xyz += matAB[x + 3 + yz] * exp2;
                            matC_xyz += matAB[x + 4 + yz] * exp3;
                            matC_xyz += matAB[x + 5 + yz] * exp4;
                            matC_xyz += matAB[x + 6 + yz] * exp5;
                            matC_xyz += matAB[x + 7 + yz] * exp6;
                            matC_xyz += matAB[x + 8 + yz] * exp7;
                            matC_xyz += matAB[x - 1 + yz] * exp0;
                            matC_xyz += matAB[x - 2 + yz] * exp1;
                            matC_xyz += matAB[x - 3 + yz] * exp2;
                            matC_xyz += matAB[x - 4 + yz] * exp3;
                            matC_xyz += matAB[x - 5 + yz] * exp4;
                            matC_xyz += matAB[x - 6 + yz] * exp5;
                            matC_xyz += matAB[x - 7 + yz] * exp6;
                            matC_xyz += matAB[x - 8 + yz] * exp7;

                            // Compute all orders on the y axis (first positive direction, then
                            // negative one)
                            matC_xyz += matAB[((y + 1 + order) * MAXX) + xz] * exp0;
                            matC_xyz += matAB[((y + 2 + order) * MAXX) + xz] * exp1;
                            matC_xyz += matAB[((y + 3 + order) * MAXX) + xz] * exp2;
                            matC_xyz += matAB[((y + 4 + order) * MAXX) + xz] * exp3;
                            matC_xyz += matAB[((y + 5 + order) * MAXX) + xz] * exp4;
                            matC_xyz += matAB[((y + 6 + order) * MAXX) + xz] * exp5;
                            matC_xyz += matAB[((y + 7 + order) * MAXX) + xz] * exp6;
                            matC_xyz += matAB[((y + 8 + order) * MAXX) + xz] * exp7;
                            matC_xyz += matAB[((y - 1 + order) * MAXX) + xz] * exp0;
                            matC_xyz += matAB[((y - 2 + order) * MAXX) + xz] * exp1;
                            matC_xyz += matAB[((y - 3 + order) * MAXX) + xz] * exp2;
                            matC_xyz += matAB[((y - 4 + order) * MAXX) + xz] * exp3;
                            matC_xyz += matAB[((y - 5 + order) * MAXX) + xz] * exp4;
                            matC_xyz += matAB[((y - 6 + order) * MAXX) + xz] * exp5;
                            matC_xyz += matAB[((y - 7 + order) * MAXX) + xz] * exp6;
                            matC_xyz += matAB[((y - 8 + order) * MAXX) + xz] * exp7;

                            // Compute all orders on the z axis (first positive direction, then
                            // negative one)
                            matC_xyz += matAB[((z + 1 + order) * xyplane) + xy] * exp0;
                            matC_xyz += matAB[((z + 2 + order) * xyplane) + xy] * exp1;
                            matC_xyz += matAB[((z + 3 + order) * xyplane) + xy] * exp2;
                            matC_xyz += matAB[((z + 4 + order) * xyplane) + xy] * exp3;
                            matC_xyz += matAB[((z + 5 + order) * xyplane) + xy] * exp4;
                            matC_xyz += matAB[((z + 6 + order) * xyplane) + xy] * exp5;
                            matC_xyz += matAB[((z + 7 + order) * xyplane) + xy] * exp6;
                            matC_xyz += matAB[((z + 8 + order) * xyplane) + xy] * exp7;
                            matC_xyz += matAB[((z - 1 + order) * xyplane) + xy] * exp0;
                            matC_xyz += matAB[((z - 2 + order) * xyplane) + xy] * exp1;
                            matC_xyz += matAB[((z - 3 + order) * xyplane) + xy] * exp2;
                            matC_xyz += matAB[((z - 4 + order) * xyplane) + xy] * exp3;
                            matC_xyz += matAB[((z - 5 + order) * xyplane) + xy] * exp4;
                            matC_xyz += matAB[((z - 6 + order) * xyplane) + xy] * exp5;
                            matC_xyz += matAB[((z - 7 + order) * xyplane) + xy] * exp6;
                            matC_xyz += matAB[((z - 8 + order) * xyplane) + xy] * exp7;

                            matC[xyz] = matC_xyz;
                        }
                    }
                }
            }
        }
    }

    // #pragma omp parallel for schedule(dynamic)
    // for (uint64_t z = 0; z < DIMZ; ++z) {
    //     for (uint64_t y = 0; y < DIMY; ++y) {
    //         #pragma omp simd
    //         for (uint64_t x = 0; x < DIMX; ++x) {
    //             // Pre-compute planes
    //             const uint64_t xyz = DIMXYZ(x, y, z);
    //             const uint64_t yz = (z + order) * xyplane + (y + order) * MAXX + order;
    //             const uint64_t xz = (z + order) * xyplane + x + order;
    //             const uint64_t xy = (y + order) * MAXX + x + order;

    //             // Ensure compiler pre-loads exponents with `ld1rd` in SVE regs only once
    //             const double exp0 = exponents[0];
    //             const double exp1 = exponents[1];
    //             const double exp2 = exponents[2];
    //             const double exp3 = exponents[3];
    //             const double exp4 = exponents[4];
    //             const double exp5 = exponents[5];
    //             const double exp6 = exponents[6];
    //             const double exp7 = exponents[7];

    //             // Get `matC[xyz]` into temporary
    //             double matC_xyz = matC[xyz];

    //             // Compute for current cell (o = 0)
    //             matC_xyz = matAB[xyz];

    //             // Compute all orders on the x axis (first positive direction, then negative one)
    //             matC_xyz += matAB[x + 1 + yz] * exp0;
    //             matC_xyz += matAB[x + 2 + yz] * exp1;
    //             matC_xyz += matAB[x + 3 + yz] * exp2;
    //             matC_xyz += matAB[x + 4 + yz] * exp3;
    //             matC_xyz += matAB[x + 5 + yz] * exp4;
    //             matC_xyz += matAB[x + 6 + yz] * exp5;
    //             matC_xyz += matAB[x + 7 + yz] * exp6;
    //             matC_xyz += matAB[x + 8 + yz] * exp7;
    //             matC_xyz += matAB[x - 1 + yz] * exp0;
    //             matC_xyz += matAB[x - 2 + yz] * exp1;
    //             matC_xyz += matAB[x - 3 + yz] * exp2;
    //             matC_xyz += matAB[x - 4 + yz] * exp3;
    //             matC_xyz += matAB[x - 5 + yz] * exp4;
    //             matC_xyz += matAB[x - 6 + yz] * exp5;
    //             matC_xyz += matAB[x - 7 + yz] * exp6;
    //             matC_xyz += matAB[x - 8 + yz] * exp7;

    //             // Compute all orders on the y axis (first positive direction, then negative one)
    //             matC_xyz += matAB[((y + 1 + order) * MAXX) + xz] * exp0;
    //             matC_xyz += matAB[((y + 2 + order) * MAXX) + xz] * exp1;
    //             matC_xyz += matAB[((y + 3 + order) * MAXX) + xz] * exp2;
    //             matC_xyz += matAB[((y + 4 + order) * MAXX) + xz] * exp3;
    //             matC_xyz += matAB[((y + 5 + order) * MAXX) + xz] * exp4;
    //             matC_xyz += matAB[((y + 6 + order) * MAXX) + xz] * exp5;
    //             matC_xyz += matAB[((y + 7 + order) * MAXX) + xz] * exp6;
    //             matC_xyz += matAB[((y + 8 + order) * MAXX) + xz] * exp7;
    //             matC_xyz += matAB[((y - 1 + order) * MAXX) + xz] * exp0;
    //             matC_xyz += matAB[((y - 2 + order) * MAXX) + xz] * exp1;
    //             matC_xyz += matAB[((y - 3 + order) * MAXX) + xz] * exp2;
    //             matC_xyz += matAB[((y - 4 + order) * MAXX) + xz] * exp3;
    //             matC_xyz += matAB[((y - 5 + order) * MAXX) + xz] * exp4;
    //             matC_xyz += matAB[((y - 6 + order) * MAXX) + xz] * exp5;
    //             matC_xyz += matAB[((y - 7 + order) * MAXX) + xz] * exp6;
    //             matC_xyz += matAB[((y - 8 + order) * MAXX) + xz] * exp7;

    //             // Compute all orders on the z axis (first positive direction, then negative one)
    //             matC_xyz += matAB[((z + 1 + order) * xyplane) + xy] * exp0;
    //             matC_xyz += matAB[((z + 2 + order) * xyplane) + xy] * exp1;
    //             matC_xyz += matAB[((z + 3 + order) * xyplane) + xy] * exp2;
    //             matC_xyz += matAB[((z + 4 + order) * xyplane) + xy] * exp3;
    //             matC_xyz += matAB[((z + 5 + order) * xyplane) + xy] * exp4;
    //             matC_xyz += matAB[((z + 6 + order) * xyplane) + xy] * exp5;
    //             matC_xyz += matAB[((z + 7 + order) * xyplane) + xy] * exp6;
    //             matC_xyz += matAB[((z + 8 + order) * xyplane) + xy] * exp7;
    //             matC_xyz += matAB[((z - 1 + order) * xyplane) + xy] * exp0;
    //             matC_xyz += matAB[((z - 2 + order) * xyplane) + xy] * exp1;
    //             matC_xyz += matAB[((z - 3 + order) * xyplane) + xy] * exp2;
    //             matC_xyz += matAB[((z - 4 + order) * xyplane) + xy] * exp3;
    //             matC_xyz += matAB[((z - 5 + order) * xyplane) + xy] * exp4;
    //             matC_xyz += matAB[((z - 6 + order) * xyplane) + xy] * exp5;
    //             matC_xyz += matAB[((z - 7 + order) * xyplane) + xy] * exp6;
    //             matC_xyz += matAB[((z - 8 + order) * xyplane) + xy] * exp7;

    //             matC[xyz] = matC_xyz;
    //         }
    //     }
    // }
}

[[nodiscard]] auto main() -> int32_t {
    // No arguments as we defined them with the pre-processor

    init();
    for (uint64_t i = 0; i < iters; ++i) {
        // Compute one iteration of Jacobi: C = B@A
        double t1 = dml_micros();
        one_iteration();
        double t2 = dml_micros();

        // Avoid copying C into A with a simple pointer swap (zero-cost)
        matC.swap(matA);
        printf("_0_ ");
        for (uint64_t j = 0; j < 5; ++j) {
            printf("%18.15lf ", matA[DIMXYZ(DIMX / 2 + j, DIMY / 2 + j, DIMZ / 2 + j)]);
        }
        double ns_point = (t2 - t1) * 1000.0 / DIMX / DIMY / DIMZ;
        printf("\033[1m%10.0lf\033[0m %10.3lf %lu %lu %lu\n", t2 - t1, ns_point, DIMX, DIMY, DIMZ);
    }

    return 0;
}
