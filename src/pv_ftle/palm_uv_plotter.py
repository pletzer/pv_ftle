"""
Interactive U/V velocity plotter for PALM NetCDF output.

Displays a horizontal slice of the wind field coloured by speed, with
arrow glyphs showing direction.  Press 'z' to move up a z-level, 'Z'
to move down.

Usage
-----
    palm_uv_plotter <palmfile> [--time-index N] [--arrow-stride S]
                               [--fill-threshold F] [--glyph-factor G]
"""
import argparse
import sys
import os

import netCDF4
import numpy as np

# so that Python sees the shared libraries when run from source
plugin_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plugin_dir)

from pv_ftle.uvw_palm_reader import UVWPalmReader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Interactively plot the U/V velocity field from a PALM NetCDF file. "
            "Press 'z' to go up one z-level, 'Z' to go down."
        )
    )
    p.add_argument("palmfile", help="Path to PALM NetCDF file.")
    p.add_argument(
        "--time-index", type=int, default=0, metavar="N",
        help="Time step index to display (default: 0).",
    )
    p.add_argument(
        "--arrow-stride", type=int, default=4, metavar="S",
        help="Sub-sampling stride for arrow glyphs (default: 4). "
             "Increase for faster rendering on large grids.",
    )
    p.add_argument(
        "--fill-threshold", type=float, default=100.0, metavar="F",
        help="Velocity magnitude (m/s) above which cells are treated as "
             "fill/obstacle values and masked (default: 100).",
    )
    p.add_argument(
        "--glyph-factor", type=float, default=None, metavar="G",
        help="Arrow length scale in metres (default: auto = one grid cell "
             "per stride width).",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--zero-fill", dest="zero_fill", action="store_true", default=True,
        help="Replace masked fill values (e.g. -9999 building cells) with "
             "zero (default, physically correct).",
    )
    g.add_argument(
        "--no-zero-fill", dest="zero_fill", action="store_false",
        help="Preserve fill values as-is.  Reproduces pre-fix behaviour; "
             "produces artefacts at building boundaries.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- minimal NC read: time axis only ------------------------------------
    with netCDF4.Dataset(args.palmfile, "r") as nc:
        fld = UVWPalmReader._get_nc_names(nc)
        t_all = np.asarray(nc.variables[fld["time"]][:], dtype=np.float64)

    nt = t_all.size
    if not (0 <= args.time_index < nt):
        parser.error(
            f"--time-index {args.time_index} is out of range "
            f"(file has {nt} time steps, indices 0–{nt - 1})."
        )

    t_val = float(t_all[args.time_index])
    print(
        f"Plotting time index {args.time_index} / {nt - 1}  "
        f"(t = {t_val:.1f} s)  from {args.palmfile}"
    )

    if not args.zero_fill:
        print("WARNING: --no-zero-fill active; fill values (-9999) are preserved "
              "and will corrupt interpolation near building boundaries.")

    # load exactly one time step
    reader = UVWPalmReader(args.palmfile, tmin=t_val, tmax=t_val,
                           zero_fill=args.zero_fill)

    reader.plotUV(
        time_index=0,
        fill_threshold=args.fill_threshold,
        arrow_stride=args.arrow_stride,
        glyph_factor=args.glyph_factor,
    )


def cli():
    main()


if __name__ == "__main__":
    cli()
