#!/usr/bin/python

import os


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


def main():
    ref = run("ref/stencil-x86 100 100 100 5")
    cur = run("./stencil 100 100 100 5")
    ref_ms = []
    cur_ms = []
    for i, (r, c) in enumerate(zip(ref, cur)):
        r = r.split()
        c = c.split()
        
        # assert c[1] == r[1] and c[2] == r[2] and c[3] == r[3], "error: coefficients aren't coherent"
        ref_ms.append(float(r[6]))
        cur_ms.append(float(c[6]))

    ref_avg = 0.0
    cur_avg = 0.0
    for r, c in zip(ref_ms, cur_ms):
        ref_avg += r
        cur_avg += c
    ref_avg /= 5
    cur_avg /= 5
    
    print(f"Average ref: {ref_avg} ms")
    print(f"Average cur: {cur_avg} ms")
    print(f"  Speedup: {ref_avg / cur_avg:.2f}")


if __name__ == "__main__":
    main()
