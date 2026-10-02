"""Turning the DICOM files of a series into one volume of stored values."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

log = logging.getLogger(__name__)


@dataclass
class SeriesVolume:
    """Stored pixel values (no rescale) of a series, slices sorted along the
    slice normal, with the attributes the evaluation reports."""

    pixels: np.ndarray  # (Z, H, W) int16, uint16 or uint8
    bits_stored: int
    signed: bool
    modality: str
    slice_thickness: float | None
    slice_spacing: float | None
    pixel_spacing: tuple[float, float] | None
    manufacturer: str
    description: str


def volume_sha256(pixels: np.ndarray, signed: bool) -> str:
    """SHA-256 of the samples as 16-bit little-endian integers: the digest
    stored in Tomoz containers (``Volume::sha256``)."""
    return hashlib.sha256(
        np.ascontiguousarray(pixels).astype("<i2" if signed else "<u2").tobytes()
    ).hexdigest()


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(files: dict[str, bytes]) -> SeriesVolume | None:
    """The volume of a series, or None if it has no grayscale image of 8 or
    16 bits. Multi-frame instances provide their frames; otherwise the largest
    group of single-frame images sharing size and orientation is stacked."""
    images = []
    for name, data in sorted(files.items()):
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
        except (InvalidDicomError, EOFError, ValueError):
            continue
        if "PixelData" not in ds or int(getattr(ds, "SamplesPerPixel", 1)) != 1:
            continue
        if int(getattr(ds, "BitsAllocated", 0)) not in (8, 16):
            continue
        images.append((name, ds))
    if not images:
        return None
    multi = [(n, d) for n, d in images if int(getattr(d, "NumberOfFrames", 1) or 1) > 1]
    if multi:
        _, ref = max(multi, key=lambda nd: int(nd[1].NumberOfFrames))
        pixels = ref.pixel_array
        stack = [ref]
    else:

        def key(d):
            o = getattr(d, "ImageOrientationPatient", None)
            return (int(d.Rows), int(d.Columns), tuple(round(float(v), 3) for v in o) if o else None)

        groups: dict = {}
        for _, d in images:
            groups.setdefault(key(d), []).append(d)
        stack = max(groups.values(), key=len)
        ref = stack[0]
        o = getattr(ref, "ImageOrientationPatient", None)
        if o is not None and all(hasattr(d, "ImagePositionPatient") for d in stack):
            o = np.array([float(v) for v in o])
            normal = np.cross(o[:3], o[3:])
            stack.sort(
                key=lambda d: (
                    float(np.dot(normal, [float(v) for v in d.ImagePositionPatient])),
                    int(getattr(d, "InstanceNumber", 0) or 0),
                )
            )
        else:
            stack.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
        pixels = np.stack([d.pixel_array for d in stack])
    if pixels.ndim == 2:
        pixels = pixels[None]
    signed = int(getattr(ref, "PixelRepresentation", 0)) == 1
    bits = int(getattr(ref, "BitsStored", ref.BitsAllocated))
    dtype = np.uint8 if int(ref.BitsAllocated) == 8 and not signed else (np.int16 if signed else np.uint16)
    pixels = pixels.astype(dtype)
    spacing = None
    if len(stack) > 1 and all(hasattr(d, "ImagePositionPatient") for d in stack[:2]):
        p0, p1 = (np.array([float(v) for v in d.ImagePositionPatient]) for d in stack[:2])
        spacing = float(np.linalg.norm(p1 - p0))
    ps = getattr(ref, "PixelSpacing", None) or getattr(ref, "ImagerPixelSpacing", None)
    return SeriesVolume(
        pixels=pixels,
        bits_stored=bits,
        signed=signed,
        modality=str(getattr(ref, "Modality", "")),
        slice_thickness=_float(getattr(ref, "SliceThickness", None)),
        slice_spacing=spacing,
        pixel_spacing=(float(ps[0]), float(ps[1])) if ps else None,
        manufacturer=str(getattr(ref, "Manufacturer", "")),
        description=str(getattr(ref, "SeriesDescription", "")),
    )
