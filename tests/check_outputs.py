"""
Compare FTLE outputs of tests/run_cases.sh with tests/ftle_reference.json.

    python tests/check_outputs.py OUTDIR            # check, exit 1 on failure
    python tests/check_outputs.py OUTDIR --update   # rewrite the reference

For every case the reference records the md5 of the FTLE array and its min and
max.  A case passes if min and max agree with the reference to a relative
difference < --rtol (default 1e-4).  The md5 is reported but not enforced:
results are not bit-identical across compilers/CPUs (-ffast-math, FMA).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pyvista as pv

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'ftle_reference.json')


def summarise(path):
    mesh = pv.read(path)
    names = [n for n in mesh.array_names if 'ftle' in n.lower()]
    if not names:
        raise ValueError(f'no FTLE array in {mesh.array_names}')
    a = np.ascontiguousarray(mesh[names[0]])
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError('empty or non-finite FTLE')
    return dict(n=int(a.size),
                md5=hashlib.md5(a.tobytes()).hexdigest(),
                min=float(a.min()),
                max=float(a.max()))


def rel_diff(x, ref):
    return abs(x - ref) / abs(ref) if ref != 0 else abs(x)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outdir')
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--update', action='store_true',
                   help='write the current results as the new reference')
    args = p.parse_args()

    results, failed = {}, False
    for fname in sorted(os.listdir(args.outdir)):
        case, ext = os.path.splitext(fname)
        if ext not in ('.vtr', '.vts'):
            continue
        try:
            results[case] = summarise(os.path.join(args.outdir, fname))
        except ValueError as e:
            print(f'FAIL {case}: {e}')
            failed = True

    if args.update:
        with open(REFERENCE, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
            f.write('\n')
        print(f'wrote {len(results)} cases to {REFERENCE}')
        return 1 if failed else 0

    with open(REFERENCE) as f:
        reference = json.load(f)

    print(f'{"case":18s} {"min":>13s} {"rel diff":>9s} {"max":>13s} '
          f'{"rel diff":>9s}  md5')
    for case in sorted(set(reference) | set(results)):
        if case not in results:
            print(f'FAIL {case}: no output')
            failed = True
            continue
        if case not in reference:
            print(f'FAIL {case}: not in {os.path.basename(REFERENCE)} '
                  '(run with --update to add it)')
            failed = True
            continue
        got, ref = results[case], reference[case]
        if got['n'] != ref['n']:
            print(f'FAIL {case}: {got["n"]} FTLE values, reference has {ref["n"]}')
            failed = True
            continue
        dmin = rel_diff(got['min'], ref['min'])
        dmax = rel_diff(got['max'], ref['max'])
        ok = dmin < args.rtol and dmax < args.rtol
        failed |= not ok
        md5 = 'same' if got['md5'] == ref['md5'] else f'differs ({got["md5"]})'
        print(f'{case:18s} {got["min"]:13.6e} {dmin:9.1e} {got["max"]:13.6e} '
              f'{dmax:9.1e}  {md5}  {"ok" if ok else "FAIL"}')

    print('FAILED' if failed else f'all cases within rtol={args.rtol:g}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
