"""Command line of the Tomoz lab: ``tomoz-lab <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("tomoz_lab")


def _volumes(data: Path, exclude: set[str], split: str | None = None):
    """(name, metadata, volume) for every volume in ``data`` of the given
    split (volumes without a split are always included)."""
    import numpy as np

    for meta_path in sorted(data.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("collection", "") in exclude or (split and meta.get("split", split) != split):
            continue
        vol = np.load(meta_path.with_suffix(".npy"))
        yield meta_path.stem, meta, vol


def cmd_train(args: argparse.Namespace) -> None:
    import numpy as np

    from . import features as F_
    from .export import estimate_priors, model_bytes, model_id
    from .model import export_integer
    from .train import (
        Samples,
        TrainConfig,
        evaluate_integer,
        flat_statistics,
        integer_predictions,
        train,
        volume_samples,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(filter(None, args.exclude.split(",")))
    hidden = tuple(int(h) for h in args.hidden.split(","))
    report: dict = {
        "hidden": hidden,
        "excluded_collections": sorted(exclude),
        "per_slice": args.per_slice,
        "slices_2d": args.slices_2d,
        "steps": args.steps,
        "quant_steps": args.quant_steps,
        "seed": args.seed,
        "models": {},
    }
    rng = np.random.default_rng(args.seed)
    vols = [
        (name, meta, (v.astype(np.int64) - int(v.min())))
        for name, meta, v in _volumes(Path(args.data), exclude, "train")
    ]
    report["volumes"] = [name for name, _, _ in vols]
    log.info("%d training volumes", len(vols))
    for kind in (3, 2):
        three_d = kind == 3
        # The 2-D model codes 2-D images and only the first slice of each
        # tile of a volume: a few slices per volume represent the latter.
        max_slices = None if three_d else args.slices_2d
        parts = [
            volume_samples(v, three_d, args.per_slice, rng, i, max_slices) for i, (_, _, v) in enumerate(vols)
        ]
        samples = Samples.concat([p for p in parts if p is not None])
        order = rng.permutation(len(samples))
        n_val = min(200_000, len(samples) // 10)
        val, tr = samples.subset(order[:n_val]), samples.subset(order[n_val:])
        log.info(
            "%s model: %d training samples, %d validation", "3-D" if three_d else "2-D", len(tr), len(val)
        )
        cfg = TrainConfig(
            hidden=hidden,
            float_steps=args.steps,
            quant_steps=args.quant_steps,
            seed=args.seed,
            threads=args.threads,
        )
        net = train(tr, val, F_.INPUTS_3D if three_d else F_.INPUTS_2D, cfg)
        inet, out_shift = export_integer(net)
        mu16, ls8 = integer_predictions(inet, out_shift, val)
        flat = flat_statistics(vols, three_d)
        priors = estimate_priors(mu16, ls8, val.x, flat)
        name = f"tz1-{kind}d-h{'x'.join(map(str, hidden))}"
        data = model_bytes(kind, name, out_shift, inet, priors)
        path = out / f"tz1-{kind}d.tzm"
        path.write_bytes(data)
        bits = evaluate_integer(inet, out_shift, val)
        report["models"][f"{kind}d"] = {
            "file": path.name,
            "name": name,
            "id": model_id(data),
            "bytes": len(data),
            "validation_bits_per_sample": round(bits, 4),
            "training_samples": len(tr),
            "history": cfg.history,
        }
        log.info(
            "wrote %s (%d bytes, id %s): %.4f bits/sample on validation",
            path,
            len(data),
            model_id(data),
            bits,
        )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def cmd_select(args: argparse.Namespace) -> None:
    from .datasets.manifest import select, write_manifest

    entries = select()
    write_manifest(Path(args.out), entries)
    log.info("wrote %s: %d series", args.out, len(entries))


def cmd_fetch(args: argparse.Namespace) -> None:
    from .datasets.manifest import fetch

    path = Path(args.manifest)
    manifest = json.loads(path.read_text())

    def save() -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1) + "\n")
        tmp.replace(path)

    fetch(manifest, Path(args.out), set(args.split.split(",")), record=args.record, on_record=save)


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import markdown, run, summarize

    out = Path(args.out)
    csv_path = out / "results.csv"
    run(Path(args.data), csv_path, Path(args.models) if args.models else None, args.limit)
    summary = summarize(csv_path)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(summary))
    print(markdown(summary))


def cmd_golden(args: argparse.Namespace) -> None:
    from .golden import write

    write(Path(args.out))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tomoz-lab", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the 2-D and 3-D predictors and export them")
    t.add_argument("--data", required=True, help="directory of volumes (.npy + .json)")
    t.add_argument("--out", required=True, help="output directory for .tzm files and the report")
    t.add_argument("--exclude", default="", help="comma-separated collections held out for testing")
    t.add_argument("--hidden", default="48,48")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--quant-steps", type=int, default=6_000)
    t.add_argument("--per-slice", type=int, default=3000)
    t.add_argument("--slices-2d", type=int, default=4, help="slices of each volume used for the 2-D model")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=0, help="PyTorch threads (0: all cores)")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("select", help="select series from TCIA and write a manifest")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_select)

    f = sub.add_parser("fetch", help="download the series of a manifest")
    f.add_argument("--manifest", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--split", default="train,test")
    f.add_argument("--record", action="store_true", help="store the SHA-256 of new series in the manifest")
    f.set_defaults(func=cmd_fetch)

    e = sub.add_parser("eval", help="benchmark Tomoz and the baselines on the test split")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True, help="directory for results.csv, summary.json and summary.md")
    e.add_argument("--models", help="directory with tz1-2d.tzm and tz1-3d.tzm (default: built-in models)")
    e.add_argument("--limit", type=int)
    e.set_defaults(func=cmd_eval)

    g = sub.add_parser("golden", help="write golden test vectors for the Rust codec")
    g.add_argument("out")
    g.set_defaults(func=cmd_golden)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        stream=sys.stderr,
    )
    args.func(args)


if __name__ == "__main__":
    main()
