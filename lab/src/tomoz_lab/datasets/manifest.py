"""Dataset manifests: which public series are used for training and testing.

Selection is deterministic: within each collection, series that pass the
size filters are ordered by the SHA-256 of their Series Instance UID and the
first ones from distinct patients are taken. The resulting manifest pins the
exact series and the SHA-256 of their samples, so later changes in the
archive cannot silently change the data.

Training and test series come from disjoint collections (different sites,
scanners and protocols), so the test measures generalisation to data the
models have never seen. Collections under non-commercial licences are not
used: the released models must be usable by anyone.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import tcia
from .volumes import build, volume_sha256

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rule:
    collection: str
    modality: str
    split: str  # "train" or "test"
    count: int
    min_images: int = 1
    max_images: int = 100_000
    #: Training series keep a window of this many central slices.
    window: int | None = None


TRAIN_WINDOW = 40

#: The TZ1 dataset. Collections are split by site: none appears in both.
RULES: tuple[Rule, ...] = (
    # CT, training
    *(
        Rule(c, "CT", "train", 4, 60, 1000, TRAIN_WINDOW)
        for c in (
            "Pancreas-CT",
            "CT COLONOGRAPHY",
            "LCTSC",
            "Pediatric-CT-SEG",
            "HCC-TACE-Seg",
            "C4KC-KiTS",
            "CT Lymph Nodes",
            "FDG-PET-CT-Lesions",
            "Lung-PET-CT-Dx",
            "COVID-19-NY-SBU",
        )
    ),
    # CT, test
    *(
        Rule(c, "CT", "test", 2, 60, 350)
        for c in (
            "LIDC-IDRI",
            "Spine-Mets-CT-SEG",
            "StageII-Colorectal-CT",
            "Colorectal-Liver-Metastases",
            "CPTAC-PDA",
            "PSMA-PET-CT-Lesions",
        )
    ),
    # MR
    *(
        Rule(c, "MR", "train", 4, 20, 400, TRAIN_WINDOW)
        for c in ("PROSTATEx", "UPENN-GBM", "ACRIN-6698", "Prostate-3T", "MRI-DIR", "ReMIND")
    ),
    *(
        Rule(c, "MR", "test", 3, 20, 400)
        for c in ("Prostate-MRI-US-Biopsy", "QIN-BREAST", "Vestibular-Schwannoma-SEG")
    ),
    # PET
    *(Rule(c, "PT", "train", 4, 60, 1000, TRAIN_WINDOW) for c in ("FDG-PET-CT-Lesions", "Lung-PET-CT-Dx")),
    *(Rule(c, "PT", "test", 3, 60, 1000) for c in ("ACRIN-NSCLC-FDG-PET", "PSMA-PET-CT-Lesions")),
    # Projection radiography and mammography (2-D)
    Rule("COVID-19-NY-SBU", "CR", "train", 4),
    Rule("COVID-19-NY-SBU", "DX", "train", 4),
    Rule("CBIS-DDSM", "MG", "train", 6),
    Rule("COVID-19-AR", "DX", "test", 3),
    Rule("COVID-19-AR", "CR", "test", 3),
    Rule("CMMD", "MG", "test", 4),
)


@dataclass
class Entry:
    id: str
    split: str
    collection: str
    modality: str
    series_uid: str
    license: str
    collection_doi: str
    window: int | None
    sha256: str | None = None
    shape: list[int] | None = None


def _order(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


def select(rules: tuple[Rule, ...] = RULES) -> list[Entry]:
    """Queries TCIA and applies the selection rules."""
    entries: list[Entry] = []
    seen: set[str] = set()
    for rule in rules:
        candidates = [
            s
            for s in tcia.series(rule.collection, rule.modality)
            if rule.min_images <= int(s.get("ImageCount", 0)) <= rule.max_images
            and "NonCommercial" not in s.get("LicenseName", "")
            and s["SeriesInstanceUID"] not in seen
        ]
        candidates.sort(key=lambda s: _order(s["SeriesInstanceUID"]))
        patients: set[str] = set()
        taken = 0
        for s in candidates:
            if s.get("PatientID") in patients:
                continue
            patients.add(s.get("PatientID"))
            seen.add(s["SeriesInstanceUID"])
            slug = rule.collection.lower().replace(" ", "-")
            entries.append(
                Entry(
                    id=f"{rule.modality.lower()}-{slug}-{_order(s['SeriesInstanceUID'])[:8]}",
                    split=rule.split,
                    collection=rule.collection,
                    modality=rule.modality,
                    series_uid=s["SeriesInstanceUID"],
                    license=s.get("LicenseName", ""),
                    collection_doi=s.get("CollectionURI", ""),
                    window=rule.window,
                )
            )
            taken += 1
            if taken == rule.count:
                break
        log.info(
            "%s %s %s: %d of %d candidates",
            rule.collection,
            rule.modality,
            rule.split,
            taken,
            len(candidates),
        )
    return entries


def window(pixels: np.ndarray, depth: int | None) -> np.ndarray:
    """The ``depth`` central slices (all of them if None)."""
    if depth is None or pixels.shape[0] <= depth:
        return pixels
    lo = (pixels.shape[0] - depth) // 2
    return pixels[lo : lo + depth]


def fetch(manifest: dict, out: Path, splits: set[str], record: bool = False, on_record=None) -> None:
    """Downloads the series of a manifest into ``out`` as ``<id>.npy`` and
    ``<id>.json`` and checks their SHA-256.

    With ``record``, the SHA-256 and shape of each series are stored in the
    manifest instead, and a series without a usable grayscale image (TCIA
    metadata cannot tell, e.g. RGB screen captures filed as CT) is marked
    ``excluded`` with the reason; ``on_record`` is called after every change
    so an interrupted run keeps what it recorded."""
    out.mkdir(parents=True, exist_ok=True)
    for e in manifest["series"]:
        if e["split"] not in splits or e.get("excluded"):
            continue
        npy, meta_path = out / f"{e['id']}.npy", out / f"{e['id']}.json"
        if npy.exists() and meta_path.exists():
            if record and not e.get("sha256"):
                stored = json.loads(meta_path.read_text())
                e["sha256"], e["shape"] = stored["sha256"], stored["shape"]
                if on_record:
                    on_record()
            continue
        log.info("fetching %s (%s)", e["id"], e["collection"])
        vol = build(tcia.download_series(e["series_uid"]))
        if vol is None:
            if not record:
                raise RuntimeError(f"{e['id']}: no usable image in series {e['series_uid']}")
            log.warning("%s: no 8/16-bit grayscale image, excluded", e["id"])
            e["excluded"] = "no 8- or 16-bit grayscale image"
            if on_record:
                on_record()
            continue
        pixels = window(vol.pixels, e.get("window"))
        digest = volume_sha256(pixels, vol.signed)
        if e.get("sha256") and e["sha256"] != digest:
            raise RuntimeError(f"{e['id']}: samples differ from the manifest (SHA-256 {digest})")
        meta = {
            **e,
            "bits_stored": vol.bits_stored,
            "signed": vol.signed,
            "slice_thickness": vol.slice_thickness,
            "slice_spacing": vol.slice_spacing,
            "pixel_spacing": vol.pixel_spacing,
            "manufacturer": vol.manufacturer,
            "description": vol.description,
            "sha256": digest,
            "shape": list(pixels.shape),
        }
        np.save(npy, pixels)
        meta_path.write_text(json.dumps(meta, indent=1) + "\n")
        if record:
            e["sha256"], e["shape"] = digest, list(pixels.shape)
            if on_record:
                on_record()


def write_manifest(path: Path, entries: list[Entry]) -> None:
    doc = {
        "name": "tz1",
        "source": "The Cancer Imaging Archive (TCIA), public collections, NBIA API v1",
        "api": tcia.API,
        "selection": "lab/src/tomoz_lab/datasets/manifest.py (RULES)",
        "series": [asdict(e) for e in entries],
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
