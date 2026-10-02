# tomoz (Python)

Python bindings of [Tomoz](https://github.com/ZeroXShot/tomoz), the learned
lossless codec for medical image volumes.

```python
import numpy as np
import tomoz

volume = np.load("ct.npy")            # (slices, rows, columns), int16/uint16/uint8/int8
data = tomoz.encode(volume)
assert np.array_equal(tomoz.decode(data), volume)
print(tomoz.info(data)["bits_per_sample"])
```

DICOM archives restore every file byte for byte:

```python
archive, report = tomoz.pack_dicom([("1.dcm", open("1.dcm", "rb").read())])
files = tomoz.unpack_dicom(archive)   # [(name, bytes)]
```
