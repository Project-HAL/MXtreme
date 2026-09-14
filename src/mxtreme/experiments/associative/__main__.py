"""One entry point for the associative experiment: ``python -m mxtreme.experiments.associative``.

select    choose the regions from a scan and a baseline; writes a parameter file
preview   draw what a parameter file will do
run       do it, on the rig
report    read a recording back against what was meant to happen
compare   calibration runs side by side: which stimulation pattern to condition through
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
    p.add_argument(
        "--config",
        metavar="TOML",
        help="mxtreme.toml: also print exactly where a run with these parameters would write",
    )
    p.add_argument("-o", "--out", help="png; otherwise a window")

    s = sub.add_parser("select", help="choose the regions, offline, from a baseline recording")
    s.add_argument("--params", default=os.path.join(HERE, "params_default.json"))
    s.add_argument(
        "--out",
        required=True,
        help="where to write the parameter file and figures; outside the managed store",
    )
    s.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="override one parameter before choosing, repeatable; set the culture's batch, chip, "
        "plate_date, div and well here, so the parameter file carries them",
    )
    s.add_argument(
        "--baseline",
        metavar="H5_OR_NPZ",
        help="the network scan of the culture's active set: candidates, coupling and routing",
    )
    s.add_argument(
        "--activity-scan",
        metavar="H5",
        help="optional: add the scan's active electrodes inside the chosen regions to the routing",
    )
    s.add_argument(
        "--centers",
        metavar="'x,y;x,y;...'",
        help="place the regions by hand, in um, instead of choosing them from the baseline",
    )
    s.add_argument(
        "--electrodes",
        metavar="CFG_OR_H5_OR_NPZ",
        help="with --centers and no baseline: take the routing from this file",
    )
    s.add_argument("--candidates", type=int, default=4)
    s.add_argument("--separation", type=float, help="override min_region_separation_um")
    s.add_argument("--min-active", type=int, help="absolute active-electrode floor per patch")

    r = sub.add_parser("run", help="run on the rig")
    r.add_argument("--params", required=True, help="the file select wrote, amplitudes filled in")
    r.add_argument("--config", help="mxtreme.toml naming the store; not needed when params has save_path")
    r.add_argument(
        "--work",
        metavar="DIR",
        help="outside the store: a copy of the protocol and the fire-time table go here",
    )
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
    r.add_argument(
        "--phase",
        help="record one phase of a conditioning session: baseline, encode_1..encode_N, retrieval",
    )

    t = sub.add_parser("report", help="read a recording back")
    t.add_argument("h5", nargs="+", help="one recording, or every recording of a split session")
    t.add_argument("--protocol", help="a _protocol.json, for one recording; by default read from inside it")
    t.add_argument("-o", "--out", help="draw the figure here; nothing is drawn without it")
    t.add_argument(
        "--csv",
        help="per-presentation counts, for checking by hand; default <recording>_readout.csv beside it",
    )

    c = sub.add_parser(
        "compare",
        help="set calibration runs of the same regions side by side, one per stimulation pattern",
    )
    c.add_argument("h5", nargs="+", help="two or more calibration recordings")
    c.add_argument("-o", "--out", help="draw local and remote response against amplitude here")

    args = parser.parse_args(argv)
    params = AssociativeParams.from_json(args.params) if hasattr(args, "params") else None
    if params is not None:
        params = params.with_overrides(getattr(args, "overrides", None))

    if args.command == "preview":
        from mxtreme.experiments.associative.preview import preview

        if args.config or params.save_path:
            from mxtreme.experiments.associative.preview import destination

            print("a run with these parameters would write:")
            for key, value in destination(params, args.config).items():
                print(f"  {key:<12} {value}")
            print()
        preview(params, args.out, args.cfg, show=args.out is None)
    elif args.command == "select":
        from mxtreme.experiments.associative.select import select

        select(
            params,
            args.out,
            baseline=args.baseline,
            activity_scan=args.activity_scan,
            centers=args.centers,
            electrodes=args.electrodes,
            candidates=args.candidates,
            separation=args.separation,
            min_active=args.min_active,
        )
    elif args.command == "run":
        from mxtreme.experiments.associative.run import run

        if args.mode:
            params.mode = args.mode
        if args.phase:
            params.phase = args.phase
        config = None
        if args.config:
            from mxtreme.config import Config

            config = Config.from_toml(args.config)
        run(params, config, work=args.work)
    elif args.command == "report":
        from mxtreme.experiments.associative.report import report

        report(args.h5, args.protocol, args.out, out_csv=args.csv)
    elif args.command == "compare":
        from mxtreme.experiments.associative.report import compare

        compare(args.h5, args.out)


if __name__ == "__main__":
    sys.exit(main())
