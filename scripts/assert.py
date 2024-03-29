#!/usr/bin/python

import argparse
import errno
import functools
import math
import os
import sys


# Parse program arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "assert")
    parser.add_argument("-d", "--dimensions",
                        nargs = 3,
                        metavar = ("DIM_X", "DIM_Y", "DIM_Z"),
                        type = int,
                        default = [1000, 1000, 1000],
                        dest = "dims",
                        help = "specify the dimensions to use (default is 1000x1000x1000)")
    parser.add_argument("-i", "--iterations",
                        metavar = "ITER",
                        type = int,
                        default = 5,
                        dest = "iters",
                        help = "specify the number of iterations to use (default is 5)")
    parser.add_argument("-a", "--accuracy",
                        metavar = "MIN_TOL",
                        type = float,
                        default = 1e-13,
                        dest = "accuracy",
                        help = "specify the minimum accuracy required when comparing coefficients (default is 1e-13)")
    parser.add_argument("-r", "--rerun",
                        action = "store_true",
                        default = False,
                        dest = "rerun",
                        help = "re-run reference code (uses specified dimensions, default ones otherwise)")
    parser.add_argument("-p", "--preset",
                        choices = ["small", "medium", "big", "official"],
                        default = None,
                        dest = "preset",
                        help = "specify a preset run to compare against: small (100x100x100), medium (500x500x500), big (1000x1000x1000) or official (1000x1000x1000)"),
    parser.add_argument("-o", "--output",
                        type = str,
                        default = None,
                        dest = "output",
                        help = "specify an output file of the current code (avoids running it)")
    parser.add_argument("-f", "--fail",
                        action = "store_true",
                        dest = "fail",
                        help = "specify if a run should fail if one or more coefficient do not reach the required accuracy, or in case of mismatched dimensions")
    return parser


# Strip bold text formatting from strings
def colorstrip(data: str) -> str:
    return data.replace('\x1b[1m', '').replace('\x1b[0m', '')


# Run command
def run(command: str) -> list:
    bin = command.split(maxsplit = 1)[0]
    args = command.split(maxsplit = 1)[1:]
    if not os.path.exists(bin):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), bin)
    if args[0] != 1000 or args[1] != 1000 or args[2] != 1000:
        # TODO: re-compile with appropriate dimensions
        pass
    run = os.popen(command)
    res = run.read().strip('\n')
    run.close()

    cleaned_contents = []
    for line in colorstrip(res).split('\n'):
        if line.startswith("_0_"):
            cleaned_contents.append(line)
    return cleaned_contents


def parse_output(file: str) -> list:
    with open(file, "r") as f:
        contents = f.read().strip('\n')
        cleaned_contents = []
        for line in colorstrip(contents).split('\n'):
            if line.startswith("_0_"):
                cleaned_contents.append(line)
    return cleaned_contents


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Get dimensions
    dimx = args.dims[0]
    dimy = args.dims[1]
    dimz = args.dims[2]
    iter = args.iters

    # Override default values if using a preset
    if args.preset == "small":
        dimx = dimy = dimz = 100
        iter = 5
    elif args.preset == "medium":
        dimx = dimy = dimz = 500
        iter = 5
    elif args.preset == "big" or args.preset == "official":
        dimx = dimy = dimz = 1000
        iter = 5

    print(f"Dimensions: \033[1;34m{dimx}\033[0mx\033[1;34m{dimy}\033[0mx\033[1;34m{dimz}\033[0m")
    print(f"Iterations: \033[1;34m{iter}\033[0m")
    print(f"  Accuracy: \033[1;34m{args.accuracy}\033[0m\n")

    # Get reference code output, either by re-running or by looking up existing
    # reference output files
    if args.rerun is True:
        print("Re-running reference code...", end = ' ')
        sys.stdout.flush()
        ref = run(f"ref/stencil {dimx} {dimy} {dimz} {iter}")
        print(f"done{chr(10) if args.output is not None else ''}")
    else:
        if args.preset == "medium":
            ref = parse_output("ref/ref500.out")
        elif args.preset == "big":
            ref = parse_output("ref/ref1000.out")
        elif args.preset == "official":
            ref = parse_output("ref/ref_official.out")
        else:
            ref = parse_output("ref/ref.out")

    if args.output is not None:
        if os.path.exists(args.output) and os.path.isfile(args.output):
            cur = parse_output(args.output)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.output)
    else:
        print("Running current code...", end = ' ')
        sys.stdout.flush()
        cur = run(f"./stencil {dimx} {dimy} {dimz} {iter}")
        print("done\n")

    accuracy_errors = 0
    dimension_errors = 0
    # Warn user if number of iterations in reference is greater than current
    if len(ref) > len(cur):
        print(f"\033[1;33mwarning:\033[0m number of iterations don't match")
        print(f"    -> reference is \033[35m{len(ref)}\033[0m, but current is only \033[35m{len(cur)}\033[0m")
        dimension_errors += 1

    ref_times = []
    cur_times = []
    for i, (r, c) in enumerate(zip(ref, cur)):
        r = [float(x) for x in r.split()[1:]]
        c = [float(x) for x in c.split()[1:]]

        # Check that dimensions match (only at first iteration)
        if i == 0:
            for d, (dim_r, dim_c) in enumerate(zip(r[7:10], c[7:10])):
                if dim_r != dim_c:
                    print(f"\033[1;33mwarning:\033[0m detected different dimensions on the {'x' if d == 0 else 'y' if d == 1 else 'z'} axis")
                    print(f"    -> reference is \033[35m{int(dim_r)}\033[0m, but current is \033[35m{int(dim_c)}\033[0m")
                    dimension_errors += 1
            # Fail run if flag is enabled
            if args.fail is True and dimension_errors != 0:
                print(f"\033[1;31merror:\033[0m run failed because of {dimension_errors} incoherent dimension{'s' if dimension_errors > 1 else ''}")
                exit(dimension_errors)

        # Assert that result of the current version is comparable to reference
        # to the specified accuracy
        for j, (val_r, val_c) in enumerate(zip(r[:5], c[:5])):
            if not math.isclose(val_r, val_c, rel_tol = args.accuracy):
                print(f"\033[1;33mwarning:\033[0m at iteration \033[1m#{i + 1}\033[0m coefficients \033[1m#{j}\033[0m are incoherent")
                print(f"    -> reference is \033[35m{val_r}\033[0m, but current is \033[35m{val_c}\033[0m")
                accuracy_errors += 1

        # Store the iteration time
        ref_times.append(r[5])
        cur_times.append(c[5])

    # Fail run if flag is enabled
    if args.fail is True and accuracy_errors != 0:
        print(f"\033[1;31merror:\033[0m run failed because of {accuracy_errors} incoherent coefficient{'s' if accuracy_errors > 1 else ''}")
        exit(accuracy_errors)
    else:
        print(f"{chr(10) if accuracy_errors + dimension_errors != 0 else ''}\033[1;32mSuccess!\033[0m Run passed all checks")

    # Sort data
    ref_times.sort()
    cur_times.sort()

    # Get min/max
    ref_min = ref_times[0]
    ref_max = ref_times[-1]
    cur_min = cur_times[0]
    cur_max = cur_times[-1]

    # Compute average
    real_iters = min(len(ref), len(cur))
    ref_avg = functools.reduce(lambda sum, x: sum + x, ref_times, 0.0) / real_iters
    cur_avg = functools.reduce(lambda sum, x: sum + x, cur_times, 0.0) / real_iters

    # Compute standard deviation
    ref_std = math.sqrt(functools.reduce(lambda sum, x: sum + (x - ref_avg)**2, ref_times, 0.0) / (real_iters - 1))
    cur_std = math.sqrt(functools.reduce(lambda sum, x: sum + (x - cur_avg)**2, cur_times, 0.0) / (real_iters - 1))

    print("\n\033[1mBenchmark 1:\033[m reference stencil")
    print(f"  Time (\033[1;32mmean\033[0m +/- \033[32mσ\033[0m):  \033[1;32m{ref_avg / 1000.0:3.2f} ms\033[0m +/- \033[32m{ref_std / 1000.0:3.2f} ms\033[0m")
    print(f"  Range (\033[36mmin\033[0m … \033[35mmax\033[0m):  \033[36m{ref_min / 1000.0:3.2f} ms\033[0m ... \033[35m{ref_max / 1000.0:3.2f} ms\033[0m")

    print("\n\033[1mBenchmark 2:\033[m current stencil")
    print(f"  Time (\033[1;32mmean\033[0m +/- \033[32mσ\033[0m):  \033[1;32m{cur_avg / 1000.0:3.2f} ms\033[0m +/- \033[32m{cur_std / 1000.0:3.2f} ms\033[0m")
    print(f"  Range (\033[36mmin\033[0m … \033[35mmax\033[0m):  \033[36m{cur_min / 1000.0:3.2f} ms\033[0m ... \033[35m{cur_max / 1000.0:3.2f} ms\033[0m\n")

    print("\033[1mSummary:\033[0m")
    print(f"  \033[1m'current'\033[0m is \033[1;32m~{ref_max / cur_min:.2f}\033[0m times faster than \033[1m'reference'\033[0m")

    exit(accuracy_errors)


if __name__ == "__main__":
    main()
