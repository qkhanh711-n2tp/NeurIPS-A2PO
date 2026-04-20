from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .registry import get_experiment, list_experiments
except ImportError:
    from registry import get_experiment, list_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect/export experiment specs derived from A2PO.md")
    parser.add_argument("--list", action="store_true", help="list registered experiments")
    parser.add_argument("--name", type=str, help="experiment id to inspect")
    parser.add_argument("--write", action="store_true", help="write resolved config under experiments/generated/<name>/")
    args = parser.parse_args()

    if args.list:
        for name in list_experiments():
            print(name)
        return

    if not args.name:
        raise SystemExit("Pass --list or --name <experiment_id>.")

    spec = get_experiment(args.name)
    payload = spec.to_dict()
    print(json.dumps(payload, indent=2))

    if args.write:
        root = Path(__file__).resolve().parent
        outdir = root / "generated" / args.name
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "resolved_config.json"
        outpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote: {outpath}")


if __name__ == "__main__":
    main()
