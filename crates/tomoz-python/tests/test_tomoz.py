import numpy as np
import pytest

import tomoz


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint16, np.int16])
def test_roundtrip(dtype):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    base = np.linspace(info.min, info.max, 3 * 20 * 30).reshape(3, 20, 30)
    v = np.clip(base + rng.integers(-3, 4, base.shape), info.min, info.max).astype(dtype)
    data = tomoz.encode(v)
    back = tomoz.decode(data)
    assert back.dtype == v.dtype
    assert np.array_equal(back, v)
    assert tomoz.info(data)["shape"] == v.shape


def test_two_dimensional_and_metadata():
    v = (np.arange(40 * 50) % 1000).astype(np.uint16).reshape(40, 50)
    data = tomoz.encode(v, bits=12, metadata=b"hello")
    assert np.array_equal(tomoz.decode(data)[0], v)
    assert tomoz.info(data)["metadata"] == b"hello"


def test_errors():
    with pytest.raises(TypeError):
        tomoz.encode(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        tomoz.encode(np.zeros((2, 2, 2, 2), np.uint16))
    with pytest.raises(ValueError):
        tomoz.decode(b"not a tomoz file")
