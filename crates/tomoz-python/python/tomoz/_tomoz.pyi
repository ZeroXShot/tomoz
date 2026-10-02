from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

class Models:
    def __init__(self, model_2d: bytes, model_3d: bytes) -> None: ...
    @property
    def ids(self) -> tuple[str, str]: ...

def encode(
    array: npt.NDArray[np.integer[Any]],
    *,
    bits: int | None = None,
    slab: int = 32,
    stripe: int = 512,
    packing: bool = True,
    metadata: bytes | None = None,
    models: Models | None = None,
    threads: int | None = None,
) -> bytes: ...
def decode(
    data: bytes, *, models: Models | None = None, verify: bool = True, threads: int | None = None
) -> npt.NDArray[np.integer[Any]]: ...
def info(data: bytes) -> dict[str, Any]: ...
def pack_dicom(
    files: list[tuple[str, bytes]], *, zstd_level: int = 19, models: Models | None = None
) -> tuple[bytes, dict[str, Any]]: ...
def unpack_dicom(data: bytes, *, models: Models | None = None) -> list[tuple[str, bytes]]: ...
def builtin_models() -> dict[str, tuple[str, str]]: ...
