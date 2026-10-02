# Built-in models (TZ1)

| File | Kind | Identifier | Network | Size |
|---|---|---|---|---|
| `tz1-2d.tzm` | 2-D: 2-D images and the first slice of each tile | `78f954e342dbbd3eef09fc52ab780e42` | 12-48-48-6 | 4,401 B |
| `tz1-3d.tzm` | 3-D: every other slice | `fe411d9822c41fac88240c959e0e48ea` | 30-48-48-6 | 5,265 B |

The identifier is the first 16 bytes of the file's SHA-256; containers name
the models they were coded with, so these files can never change without
changing the identifiers (a test pins them).

## Provenance

Trained with the lab on the training split of the TZ1 manifest
(`lab/manifests/tz1.json`: 81 volumes from 17 public TCIA collections under
CC BY 3.0/4.0; attribution in [NOTICE](../../../NOTICE)):

```sh
cd lab
uv run tomoz-lab fetch --manifest manifests/tz1.json --out ../data/tz1 --split train
uv run --extra train tomoz-lab train --data ../data/tz1 --out models --threads 3
```

with the defaults recorded in `training-report.json`: hidden layers 48 and
48, 20,000 float steps then 6,000 quantisation-aware steps, batches of
8,192, seed 0. Validation (held-out samples of the training volumes):

| Model | Float | Integer (shipped) |
|---|---|---|
| 3-D | 5.156 bits/sample | 5.207 bits/sample |
| 2-D | 7.009 bits/sample | 7.030 bits/sample |

PyTorch on CPU is deterministic for a fixed thread count; retraining on
another machine gives models of equivalent quality but not necessarily the
same bytes. Containers record model identifiers, so a decoder with the
shipped models always decodes containers made with them.
