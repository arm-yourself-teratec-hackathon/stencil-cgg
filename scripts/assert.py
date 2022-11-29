#!/usr/bin/python

import argparse
import functools
import math
import os
import sys


# Parse program arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "stencil-assert")
    parser.add_argument("-r", "--rerun",
                        nargs = 4,
                        metavar = ("dim_x", "dim_y", "dim_z", "iter"),
                        type = int,
                        default = [100, 100, 100, 5],
                        dest = "stencil_args",
                        help = "Re-run reference code with new dimensions.")
    parser.add_argument("-p", "--precision",
                        metavar = "eps",
                        type = float,
                        default = 1e-12,
                        dest = "precision",
                        help = "Specify a minimum precision required to validate the run.")
    parser.add_argument("-m", "--medium",
                        action = "store_true",
                        dest = "med",
                        help = "Compare against medium run (500x500x500, 5 iterations).")
    parser.add_argument("-b", "--big",
                        action = "store_true",
                        dest = "big",
                        help = "Compare against big run (1000x1000x1000, 5 iterations).")
    return parser


# Strip color/formating on strings
def colorstrip(data: str) -> str:
    find = data.find('\x03')
    while find > -1:
        done = False
        data = data[0:find] + data[find+1:]
        if len(data) <= find+1:
            done = True
        try:
            assert not done
            assert int(data[find])
            while True:
                assert int(data[find])
                data = data[0:find] + data[find+1:]
        except:
            if not done:
                if data[find] != ',': done = True

        if (not done) and (len(data) > find+1) and (data[find] == ','):
            try:
                assert not done
                assert int(data[find+1])
                data = data[0:find] + data[find+1:]
                data = data[0:find] + data[find+1:]
            except:
                done = True
            try:
                assert not done
                while True:
                    assert int(data[find])
                    data = data[0:find] + data[find+1:]
            except: pass

        find = data.find('\x03')

    data = data.replace('\x1b[1m','')
    data = data.replace('\x1b[0m','')
    data = data.replace('\x02','')
    data = data.replace('\x1d','')
    data = data.replace('\x1f','')
    data = data.replace('\x16','')
    data = data.replace('\x0f','')

    return data


def run(command: str) -> list:
    run = os.popen(command)
    res = run.read()
    res = colorstrip(res).split('\n')[:-1]
    run.close()
    return res


def get_ref(file: str) -> list:
    with open(file, "r") as f:
        c = f.read()
        c = colorstrip(c).split('\n')
    return c


def main():
    parser = build_parser()
    args = parser.parse_args()

    rerun = False
    dimx = args.stencil_args[0]
    dimy = args.stencil_args[1]
    dimz = args.stencil_args[2]
    iter = args.stencil_args[3]
    
    # Allow user to override default parameters
    if args.stencil_args != [100, 100, 100, 5]:
        rerun = True

    if rerun:
        ref = run(f"ref/stencil {dimx} {dimy} {dimz} {iter}")
    else:
        if args.med is True:
            dimx = dimy = dimz = 500
            iter = 5
            ref = get_ref("ref/ref500.out")
        elif args.big is True:
            dimx = dimy = dimz = 1000
            iter = 5
            ref = get_ref("ref/ref1000.out")
        else:
            ref = get_ref("ref/ref.out")
    cur = run(f"./stencil {dimx} {dimy} {dimz} {iter}")

    ref_ms = []
    cur_ms = []
    for i, (r, c) in enumerate(zip(ref, cur)):
        r = [float(x) for x in r.split()[1:]]
        c = [float(x) for x in c.split()[1:]]
        
        # Assert that result of the current version is still valid compared to reference
        for j, (vr, vc) in enumerate(zip(r[:5], c[:5])):
            if not math.isclose(vr, vc, rel_tol = args.precision):
                print(f"\033[1;31mfailed:\033[0m coefficients \033[1m#{j}\033[0m at iteration \033[1m#{i}\033[0m are incoherent.\n",
                      f"    -> reference is \033[34m{vr}\033[0m, but current is \033[34m{vc}\033[0m")
                exit(-1)

        # Store the iteration time
        ref_ms.append(r[6])
        cur_ms.append(c[6])

    # Compute average
    ref_avg = functools.reduce(lambda sum, x: sum + x, ref_ms, 0.0) / iter
    cur_avg = functools.reduce(lambda sum, x: sum + x, cur_ms, 0.0) / iter
    # ref_avg = 0.0
    # cur_avg = 0.0
    # for r, c in zip(ref_ms, cur_ms):
    #     ref_avg += r
    #     cur_avg += c
    # ref_avg /= iter
    # cur_avg /= iter
    
    print(f"Reference average: {ref_avg:03.2f} us")
    print(f"  Current average: {cur_avg:03.2f} us")
    print(f"\nSpeedup: \033[1;32m{ref_avg / cur_avg:.2f}x\033[0m")


if __name__ == "__main__":
    main()
