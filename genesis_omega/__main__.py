"""Command line (debug / local use). The intended experience is the notebook + browser.

    python -m genesis_omega run [--profile cpu] [--port 8765] [--root DIR] [--no-tunnel] [--reset]
    python -m genesis_omega info
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="genesis_omega")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="start (or resume) the cosmos, server and viewer")
    run.add_argument("--profile", default="auto")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--host", default="0.0.0.0")
    run.add_argument("--root", default=None, help="persistent storage root")
    run.add_argument("--scratch", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--reset", action="store_true", help="archive the existing cosmos and start a new one")
    run.add_argument("--no-tunnel", action="store_true")
    run.add_argument("--no-tune", action="store_true")
    run.add_argument("--checkpoint-every", type=float, default=None)
    sub.add_parser("info", help="print detected hardware and feature status")
    args = ap.parse_args(argv)
    if args.cmd == "info":
        from .core.hardware import detect
        from .core.status import feature_table
        print(json.dumps({"hardware": detect().to_dict(), "features": feature_table()}, indent=1))
        return 0
    from .runtime.launcher import launch

    h = launch(storage_root=args.root, scratch_root=args.scratch, profile=args.profile, seed=args.seed,
               reset=args.reset, port=args.port, host=args.host, tunnel="none" if args.no_tunnel else "auto",
               tune=not args.no_tune, checkpoint_interval_s=args.checkpoint_every)
    print("\nGENESIS Ω is running. Open:")
    for k, v in h.urls.items():
        print(f"  {k:14s} {v}")
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        while not stop["flag"]:
            time.sleep(1.0)
    finally:
        print("checkpointing and stopping…")
        h.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
