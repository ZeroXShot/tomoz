"""Learned lossless compression for medical image volumes.

>>> import numpy as np, tomoz
>>> v = (np.arange(4 * 32 * 32) % 97).astype(np.uint16).reshape(4, 32, 32)
>>> bool(np.array_equal(tomoz.decode(tomoz.encode(v)), v))
True
"""

from ._tomoz import Models, __version__, builtin_models, decode, encode, info, pack_dicom, unpack_dicom

__all__ = ["Models", "__version__", "builtin_models", "decode", "encode", "info", "pack_dicom", "unpack_dicom"]
