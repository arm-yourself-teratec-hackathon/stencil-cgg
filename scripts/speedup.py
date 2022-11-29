#!/usr/bin/python

import math
import os
import sys


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
    dim_x = 100
    dim_y = 100
    dim_z = 100
    iter = 5
    rerun = False
    
    # Allow user to override default parameters
    if len(sys.argv) == 6 and sys.argv[1] == "-r":
        rerun = True
        dim_x = sys.argv[1]
        dim_y = sys.argv[2]
        dim_z = sys.argv[3]
        iter = sys.argv[4]
    elif len(sys.argv) != 1:
        print("\033[1mUsage:\033[0m <dim_x> <dim_y> <dim_z> <iter>")
        exit(-1)

    if rerun:
        ref = run(f"ref/stencil {dim_x} {dim_y} {dim_z} {iter}")
    else:
        ref = get_ref("ref/ref.out")
        
    cur = run(f"./stencil {dim_x} {dim_y} {dim_z} {iter}")
    ref_ms = []
    cur_ms = []
    for i, (r, c) in enumerate(zip(ref, cur)):
        r = [float(x) for x in r.split()[1:]]
        c = [float(x) for x in c.split()[1:]]
        
        # Assert that result of the current version is still valid compared to reference
        for j in range(1, 5):
            if not math.isclose(c[j], r[j]):
                print(f"\033[1;31merror:\033[0m coefficients at iteration \033[1m#{i}\033[0m are incoherent.\n",
                      f"   -> reference is \033[34m{r[j]}\033[0m, but current is \033[34m{c[j]}\033[0m")
                exit(-1)

        ref_ms.append(r[6])
        cur_ms.append(c[6])

    # Compute average
    ref_avg = 0.0
    cur_avg = 0.0
    for r, c in zip(ref_ms, cur_ms):
        ref_avg += r
        cur_avg += c
    ref_avg /= 5
    cur_avg /= 5
    
    print(f"Reference average: {ref_avg:03.2f} us")
    print(f"  Current average: {cur_avg:03.2f} us")
    print(f"\nSpeedup: \033[1;32m{ref_avg / cur_avg:.2f}\033[0m")


if __name__ == "__main__":
    main()
