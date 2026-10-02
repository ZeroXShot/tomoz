"""Synthetic DICOM series for tests that must not depend on downloads.

The images are a smooth phantom with noise, stored like CT slices (16 bits
allocated, 12 stored, signed, explicit VR little endian), so that archives
and the gateway's compaction see realistic structure.
"""

from __future__ import annotations

import io
import uuid

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

_NAMESPACE = uuid.UUID("5b0a4d43-3c1e-4c38-9a8e-2b3f4f1d6c21")


def _uid(name: str) -> str:
    """A UUID-derived UID (ISO/IEC 9834-8 "2.25" arc), stable for ``name``."""
    return f"2.25.{uuid.uuid5(_NAMESPACE, name).int}"


def _phantom(z: int, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    cy, cx = rows / 2, cols / 2
    body = ((y - cy) / (0.42 * rows)) ** 2 + ((x - cx) / (0.46 * cols)) ** 2 <= 1
    organ = ((y - cy * 0.9) / (0.15 * rows)) ** 2 + ((x - cx * 1.1) / (0.12 * cols + z % 3)) ** 2 <= 1
    img = np.where(body, 40, -1000) + np.where(organ, 60, 0) + (x // 16) * body
    noise = rng.normal(0, 12, size=img.shape) * body
    return np.clip(img + noise, -1024, 2047).astype(np.int16)


def ct_series(
    count: int, rows: int = 128, cols: int = 128, series: int = 1, seed: int = 0
) -> list[tuple[str, bytes]]:
    """``count`` slices of one synthetic CT series as (file name, DICOM bytes)."""
    rng = np.random.default_rng(seed)
    study_uid = _uid(f"study {series}")
    series_uid = _uid(f"series {series}")
    files = []
    for n in range(count):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = _uid(f"instance {series} {n}")
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientName = "Synthetic^Phantom"
        ds.PatientID = "SYNTHETIC"
        ds.InstanceNumber = n + 1
        ds.ImagePositionPatient = [0.0, 0.0, 2.5 * n]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [0.8, 0.8]
        ds.SliceThickness = 2.5
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelData = _phantom(n, rows, cols, rng).tobytes()
        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, enforce_file_format=True)
        files.append((f"slice{n + 1:04d}.dcm", buf.getvalue()))
    return files
