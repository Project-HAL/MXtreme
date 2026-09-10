"""One entry point for the associative experiment: ``python -m mxtreme.experiments.associative``.

select    choose the regions from a scan and a baseline; writes a parameter file
preview   draw what a parameter file will do
run       do it, on the rig
report    read a recording back against what was meant to happen
"""

from __future__ import annotations

import argparse
import os
import sys

from mxtreme.experiments.associative.params import AssociativeParams

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mxtreme.experiments.associative",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="draw a parameter file")
    p.add_argument("--params", default=os.path.join(HERE, "params_default.json"))
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override one parameter, repeatable; e.g. --set dt_cs_us=20",
    )
    p.add_argument("--config-file", dest="cfg", help="a .cfg to draw routed electrodes from")
    p.add_argument("-o", "--out", help="png; otherwise a window")

    s = sub.add_parser("select", help="choose the regions")
    s.add_argument("--params", default=os.path.join(HERE, "params_default.json"))
    s.add_argument("--out", required=True, help="directory for the parameter file, record and figure")
    g = s.add_mutually_exclusive_group()
    g.add_argument(
        "--activity-scan", metavar="H5", help="a scan: where the culture fires, for the candidates"
    )
    g.add_argument("--scan-npz", metavar="NPZ", help="a preprocessed recording of one well instead of a scan")
    g.add_argument("--scan-with", metavar="TOML", help="run an activity scan now, into this store")
    g.add_argument("--centers", metavar="'x,y;x,y;...'", help="skip the scan: candidate centres in um")
    s.add_argument(
        "--electrodes",
        metavar="CFG_OR_H5_OR_NPZ",
        help="with --centers: reuse this file's electrode set for the routing, instead of inventing one",
    )
    b = s.add_mutually_exclusive_group()
    b.add_argument(
        "--baseline",
        metavar="H5_OR_NPZ",
        help="one continuous unstimulated recording with the candidates routed, for the "
        "coupling; a multi-round scan is refused",
    )
    b.add_argument("--record-baseline", metavar="TOML", help="record one now, into this store")
    s.add_argument("--baseline-sec", type=int, default=300)
    s.add_argument("--candidates", type=int, default=4)
    s.add_argument("--separation", type=float, help="override min_region_separation_um")
    s.add_argument("--min-active", type=int, help="absolute active-electrode floor per patch")

    r = sub.add_parser("run", help="run on the rig")
    r.add_argument("--params", required=True, help="the file select wrote, amplitudes filled in")
    r.add_argument("--config", help="mxtreme.toml naming the store; not needed when params has save_path")
    r.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override one parameter, repeatable; e.g. --set t_stim=600 --set encode_cycles=1",
    )
    r.add_argument(
        "--mode",
        choices=["conditioning", "calibration", "connectivity"],
        help="override the file's mode; calibration and connectivity are the two short gates",
    )

    t = sub.add_parser("report", help="read a recording back")
    t.add_argument("h5")
    t.add_argument("--protocol", required=True, help="the _protocol.json run wrote")
    t.add_argument("-o", "--out", help="png")
    t.add_argument("--csv", help="write every per-presentation count here, for checking by hand")

    args = parser.parse_args(argv)
    params = AssociativeParams.from_json(args.params) if hasattr(args, "params") else None
    if params is not None:
        params = params.with_overrides(getattr(args, "overrides", None))

    if args.command == "preview":
        from mxtreme.experiments.associative.preview import preview

        preview(params, args.out, args.cfg, show=args.out is None)
    elif args.command == "select":
        from mxtreme.experiments.associative.select import select

        select(
            params,
            args.out,
            activity_scan=args.activity_scan,
            scan_npz=args.scan_npz,
            scan_with=args.scan_with,
            centers=args.centers,
            electrodes=args.electrodes,
            baseline=args.baseline,
            record_baseline=args.record_baseline,
            baseline_sec=args.baseline_sec,
            candidates=args.candidates,
            separation=args.separation,
            min_active=args.min_active,
        )
    elif args.command == "run":
        from mxtreme.experiments.associative.run import run

        if args.mode:
            params.mode = args.mode
        config = None
        if args.config:
            from mxtreme.config import Config

            config = Config.from_toml(args.config)
        run(params, config)
    elif args.command == "report":
        from mxtreme.experiments.associative.report import report

        report(args.h5, args.protocol, args.out, out_csv=args.csv)


if __name__ == "__main__":
    sys.exit(main())
