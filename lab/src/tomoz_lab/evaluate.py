"""Benchmark of Tomoz against classical lossless codecs on the test split.

For every test volume and codec the harness records the compressed size,
encode and decode times and whether the round trip was exact. Sizes are
reported in bits per sample; savings relative to a baseline are averaged
per volume (each volume counts once) with a bootstrap 95 % interval.
Times are single-threaded for every codec; Tomoz is also timed with all
cores, since its tiles are coded in parallel.

Projection images (radiographs, mammograms) are 2-D: a test "volume" of such
a modality is a stack of different views, which Tomoz codes one tile per
image (slab 1) exactly as its DICOM archives do, so no image is predicted
from another, as with the 2-D baselines.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import baselines

log = logging.getLogger(__name__)

#: Modalities whose images are projections, never slices of one volume.
PROJECTION_MODALITIES = {"CR", "DX", "MG", "XA", "RF", "IO", "PX"}

FIELDS = [
    "id",
    "collection",
    "modality",
    "split",
    "depth",
    "height",
    "width",
    "bits_stored",
    "slice_spacing",
    "codec",
    "bytes",
    "bits_per_sample",
    "encode_s",
    "decode_s",
    "lossless",
]


def _tomoz(volume: np.ndarray, models, threads: int | None, slab: int = 32) -> baselines.CodecResult:
    import tomoz

    t0 = time.perf_counter()
    data = tomoz.encode(volume, slab=slab, models=models, threads=threads)
    t1 = time.perf_counter()
    back = tomoz.decode(data, models=models, threads=threads)
    t2 = time.perf_counter()
    return baselines.CodecResult(len(data), t1 - t0, t2 - t1, bool(np.array_equal(back, volume)))


def run(data_dir: Path, out_csv: Path, models_dir: Path | None, limit: int | None = None) -> None:
    import tomoz

    models = None
    if models_dir is not None:
        models = tomoz.Models(
            (models_dir / "tz1-2d.tzm").read_bytes(), (models_dir / "tz1-3d.tzm").read_bytes()
        )
    done: set[tuple[str, str]] = set()
    if out_csv.exists():
        with out_csv.open() as f:
            done = {(r["id"], r["codec"]) for r in csv.DictReader(f)}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        metas = [json.loads(p.read_text()) for p in sorted(data_dir.glob("*.json"))]
        metas = [m for m in metas if m.get("split") == "test"][:limit]
        for m in metas:
            if (m["id"], "tomoz") in done:
                continue
            vol = np.load(data_dir / f"{m['id']}.npy")
            n = vol.size
            log.info("%s %s %s", m["id"], m["modality"], vol.shape)
            slab = 1 if m["modality"] in PROJECTION_MODALITIES else 32
            results = {"tomoz": _tomoz(vol, models, 1, slab), "tomoz-mt": _tomoz(vol, models, None, slab)}
            results.update(baselines.run(vol))
            for codec, r in results.items():
                if r is None:
                    continue
                w.writerow(
                    {
                        "id": m["id"],
                        "collection": m["collection"],
                        "modality": m["modality"],
                        "split": m["split"],
                        "depth": vol.shape[0],
                        "height": vol.shape[1],
                        "width": vol.shape[2],
                        "bits_stored": m.get("bits_stored"),
                        "slice_spacing": m.get("slice_spacing"),
                        "codec": codec,
                        "bytes": r.bytes,
                        "bits_per_sample": round(8 * r.bytes / n, 5),
                        "encode_s": round(r.encode_s, 4),
                        "decode_s": round(r.decode_s, 4),
                        "lossless": r.lossless,
                    }
                )
            f.flush()


def bootstrap_ci(values: np.ndarray, reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(csv_path: Path) -> dict:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["codec"]] = r
    codecs = sorted({r["codec"] for r in rows})
    groups = ["all", *sorted({r["modality"] for r in rows})]
    summary: dict = {
        "volumes": len(by),
        "codecs": codecs,
        "lossless_failures": [(r["id"], r["codec"]) for r in rows if r["lossless"] != "True"],
        "groups": {},
    }
    for g in groups:
        vols = [v for v in by.values() if "tomoz" in v and (g == "all" or v["tomoz"]["modality"] == g)]
        if not vols:
            continue
        entry: dict = {"volumes": len(vols), "codecs": {}}
        for c in codecs:
            have = [v for v in vols if c in v]
            if not have:
                continue
            bps = np.array([float(v[c]["bits_per_sample"]) for v in have])
            raw_mb = np.array(
                [2 * int(v[c]["depth"]) * int(v[c]["height"]) * int(v[c]["width"]) / 1e6 for v in have]
            )
            enc = np.array([float(v[c]["encode_s"]) for v in have])
            dec = np.array([float(v[c]["decode_s"]) for v in have])
            e = {
                "mean_bits_per_sample": round(float(bps.mean()), 4),
                "encode_mb_per_s": round(float(raw_mb.sum() / enc.sum()), 2),
                "decode_mb_per_s": round(float(raw_mb.sum() / dec.sum()), 2),
            }
            if c not in ("tomoz", "tomoz-mt"):
                paired = [v for v in have if "tomoz" in v]
                ratio = np.array([float(v["tomoz"]["bytes"]) / float(v[c]["bytes"]) for v in paired])
                saving = 100 * (1 - ratio)
                lo, hi = bootstrap_ci(saving)
                e["tomoz_saving_percent"] = round(float(saving.mean()), 2)
                e["tomoz_saving_ci95"] = [round(lo, 2), round(hi, 2)]
                e["tomoz_smaller_on"] = f"{int((ratio < 1).sum())}/{len(ratio)}"
            entry["codecs"][c] = e
        summary["groups"][g] = entry
    summary["machine"] = {
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    summary["software"] = _versions()
    return summary


def _versions() -> dict:
    versions: dict = {}
    try:
        import tomoz

        versions["tomoz_models"] = {k: v[1] for k, v in tomoz.builtin_models().items()}
    except ImportError:
        pass
    try:
        import imagecodecs

        versions["imagecodecs"] = imagecodecs.__version__
    except ImportError:
        pass
    return versions


def markdown(summary: dict) -> str:
    lines = []
    for g, entry in summary["groups"].items():
        lines.append(f"### {g} ({entry['volumes']} volumes)\n")
        lines.append(
            "| Codec | Bits/sample | Tomoz saving | 95 % CI | Tomoz smaller | Encode MB/s | Decode MB/s |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c, e in sorted(entry["codecs"].items(), key=lambda kv: kv[1]["mean_bits_per_sample"]):
            ci = e.get("tomoz_saving_ci95")
            lines.append(
                f"| {c} | {e['mean_bits_per_sample']:.3f} | "
                + (
                    f"{e['tomoz_saving_percent']:+.1f} % | [{ci[0]:+.1f}, {ci[1]:+.1f}] | "
                    f"{e['tomoz_smaller_on']}"
                    if ci
                    else "– | – | –"
                )
                + f" | {e['encode_mb_per_s']:.1f} | {e['decode_mb_per_s']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)
