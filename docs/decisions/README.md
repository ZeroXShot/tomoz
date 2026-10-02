# Architecture decision records

Each record states a decision, the context that forced it, the alternatives
that were considered and what the decision costs. Records are not edited
after the fact; a later decision supersedes an earlier one explicitly.

| # | Decision |
|---|---|
| [0001](0001-integer-network.md) | The predictor runs on integers only |
| [0002](0002-tiles.md) | Volumes are coded as independent tiles |
| [0003](0003-relative-inputs.md) | The network sees relative, contrast-normalised context (and what did not work) |
| [0004](0004-hybrid-coding.md) | Learned mean and scale, adaptive classical entropy coding |
| [0005](0005-s3-gateway.md) | Integrate through an S3 gateway, not a PACS plugin |
| [0006](0006-data.md) | Public CC BY data, split by site, pinned by hash |
