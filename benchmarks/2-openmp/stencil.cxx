#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/time.h>
#include <vector>

using float64_t = double;
using MicroSeconds = float64_t;
using Instant = struct timespec;

/// Preset dimensions of the problem.
enum Preset: size_t {
    Small = 100,
    Medium = 500,
    Big = 1000,
};

#if !defined(PRESET)
#    define PRESET Small
#endif
#if !defined(NB_ITERATIONS)
#    define NB_ITERATIONS 5
#endif

static constexpr float64_t ONE_THOUSAND = 1.0e+3;
static constexpr float64_t ONE_MILLION = 1.0e+6;

static constexpr float64_t EXPONENT = 17.0;
static constexpr size_t HALF_ORDER = 8;
static constexpr size_t ORDER = 16;

static constexpr size_t DIMX = PRESET;
static constexpr size_t DIMY = PRESET;
static constexpr size_t DIMZ = PRESET;
static constexpr size_t MAXX = DIMX + ORDER;
static constexpr size_t MAXY = DIMY + ORDER;
static constexpr size_t MAXZ = DIMZ + ORDER;

static constexpr size_t XY_PLANE = MAXX * MAXY;
static constexpr size_t TENSOR_SIZE = MAXX * MAXY * MAXZ;

namespace instant {
    /// Returns an instant in time corresponding to "now".
    [[nodiscard]] static inline
    auto now() -> Instant {
        Instant now;
        clock_gettime(CLOCK_MONOTONIC_RAW, &now);    
        return now;
    }

    /// Returns the number of microseconds elapsed since an earlier point in time.
    [[nodiscard]] static inline
    auto elapsed_since(Instant const& start) -> MicroSeconds {
        Instant const stop = now();
        return static_cast<MicroSeconds>(
            (stop.tv_sec - start.tv_sec) * ONE_MILLION
            + (stop.tv_nsec - start.tv_nsec) / ONE_THOUSAND
        );
    }
} // namespace instant

/// Returns an offset in the center of the tensor of dimensions [0, DIM).
[[nodiscard]] static inline
auto dim_xyz(size_t x, size_t y, size_t z) -> size_t {
    size_t const z_offset = (z + HALF_ORDER) * XY_PLANE;
    size_t const y_offset = (y + HALF_ORDER) * MAXX;
    size_t const x_offset = x + HALF_ORDER;
    return z_offset + y_offset + x_offset;
}

/// Returns an offset in the center of the tensor of dimensions [-HALF_ORDER, DIM + HALF_ORDER) but
/// in indices of [0, DIM + ORDER).
[[nodiscard]] static inline
auto tensor_xyz(size_t x, size_t y, size_t z) -> size_t {
    size_t const z_offset = z * XY_PLANE;
    size_t const y_offset = y * MAXX;
    size_t const x_offset = x;
    return z_offset + y_offset + x_offset;
}

/// Initializes the tensors for the problem to solve.
///
/// Optimizing this function is not part of the exercise. It can be optimized but is not part of the
/// measured code section and does not influence performance.
/// 
/// @param A The input data tensor, with it scenter cells initialized to 1.0 and its ghost cells
///          initialized to 0.0.
/// @param B The input constant tensor, with all its cells initialized as follows:
///          sin(z * cos(y + 0.817) * cos(x + 0.311) + 0.613)
auto initialize_tensors(
    std::vector<float64_t>& A,
    std::vector<float64_t>& B
) -> void {
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t z = 0; z < MAXZ; ++z) {
            for (size_t y = 0; y < MAXY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < MAXX; ++x) {
                    B[tensor_xyz(x, y, z)] = sin(z * cos(y + 0.817) * cos(x + 0.311) + 0.613);
                }
            }
        }

        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    A[dim_xyz(x, y, z)] = 1.0;
                }
            }
        }
    } // pragma omp parallel
}

/// Performs the stencil operation on the tensors.
auto jacobi_iteration(
    std::vector<float64_t>& A,
    std::vector<float64_t> const& B,
    std::vector<float64_t>& C
) -> void {
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    C[dim_xyz(x, y, z)] = A[dim_xyz(x, y, z)] * B[dim_xyz(x, y, z)];
                    for (size_t o = 1; o <= HALF_ORDER; ++o) {
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x - o, y, z)] * B[dim_xyz(x - o, y, z)] / pow(EXPONENT, o);
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x + o, y, z)] * B[dim_xyz(x + o, y, z)] / pow(EXPONENT, o);
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x, y - o, z)] * B[dim_xyz(x, y - o, z)] / pow(EXPONENT, o);
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x, y + o, z)] * B[dim_xyz(x, y + o, z)] / pow(EXPONENT, o);
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x, y, z - o)] * B[dim_xyz(x, y, z - o)] / pow(EXPONENT, o);
                        C[dim_xyz(x, y, z)] +=
                            A[dim_xyz(x, y, z + o)] * B[dim_xyz(x, y, z + o)] / pow(EXPONENT, o);
                    }
                }
            }
        }

        // Tensor copy: A <- C
        #pragma omp for
        for (size_t z = 0; z < DIMZ; ++z) {
            for (size_t y = 0; y < DIMY; ++y) {
                #pragma omp simd
                for (size_t x = 0; x < DIMX; ++x) {
                    A[dim_xyz(x, y, z)] = C[dim_xyz(x, y, z)];
                }
            }
        }
    } // pragma omp parallel
}

auto main() -> int32_t {
    std::vector<float64_t> A(TENSOR_SIZE, 0.0);
    std::vector<float64_t> B(TENSOR_SIZE, 0.0);
    std::vector<float64_t> C(TENSOR_SIZE, 0.0);

    initialize_tensors(A, B);

    for (size_t i = 0; i < NB_ITERATIONS; ++i) {
        Instant const start = instant::now();
        jacobi_iteration(A, B, C);
        MicroSeconds elapsed = instant::elapsed_since(start);

        printf("_0_");
        for (size_t j = 0; j < 5; j++) {
            printf(" %18.15lf", A[dim_xyz(DIMX / 2 + j, DIMY / 2 + j, DIMZ / 2 + j)]);
        }
        float64_t ns_point = elapsed * ONE_THOUSAND / DIMX / DIMY / DIMZ;
        printf(" %10.0lf %10.3lf %zu %zu %zu\n", elapsed, ns_point, DIMX, DIMY, DIMZ);
    }

    return 0;
}