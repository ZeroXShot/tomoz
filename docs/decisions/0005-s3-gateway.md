# 0005 — Integrate through an S3 gateway, not a PACS plugin

## Context

To save storage, Tomoz has to sit where DICOM is stored. The obvious
integration points are plugins for open-source PACS (Orthanc's storage area
plugins, dcm4chee's storage SPI). A plugin would tie Tomoz to one PACS, its
release cycle and its licensing terms (Orthanc is GPLv3), and every other
system would need its own integration.

## Decision

Tomoz provides an S3-compatible server. Orthanc (S3 storage plugin),
dcm4chee (S3 storage), cloud-native DICOM stores and research pipelines
already write to S3; pointing them at the gateway requires configuration,
not code. The gateway stores objects as they arrive and compacts each quiet
series in the background. Series are found from the objects' content (the
Series Instance UID in the DICOM header), not from key names, so any client's
key layout works — Orthanc, for example, uses opaque attachment UUIDs.

## Alternatives

- *PACS plugins*: tighter integration (e.g. compressing at ingest), but one
  per PACS, each with its own licence and release cycle.
- *A filesystem (FUSE) layer*: universal, but random writes, renames and
  partial reads are hard to map onto immutable archives, and FUSE is not
  available everywhere.
- *Transcoding inside DICOM (a new transfer syntax)*: the clean long-term
  solution, but requires standardisation and support in every viewer;
  Tomoz's archives restore the original files instead, so nothing downstream
  changes.

## Consequences

- Clients get byte-exact objects back, ETags included; compaction is
  invisible to them except for the `x-tomoz-storage` header.
- The gateway must implement enough of S3 (SigV4, aws-chunked uploads,
  multipart, listing) to satisfy real SDKs; compatibility is tested with
  boto3 in CI.
- Compression happens after ingest, so storage briefly holds the raw
  objects; the delay is configurable (`compaction.quiet_seconds`).
