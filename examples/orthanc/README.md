# Orthanc storing through Tomoz

[Orthanc](https://www.orthanc-server.com/) is an open-source PACS. With its
S3 storage plugin it keeps every DICOM file as an object; here those objects
go to the Tomoz gateway, which compacts each study's series after it goes
quiet and serves Orthanc the original bytes whenever it reads them back.

```text
modality / upload ──DICOM──▶ Orthanc ──S3 (SigV4)──▶ tomoz serve ──▶ raw objects
                                                        │  after 30 s quiet
                                                        └──▶ verified Tomoz archives
```

## Run

Requires Docker with Compose v2 and `openssl`.

```sh
./init.sh                       # random credentials in .env and ./credentials (git-ignored)
docker compose up -d --build    # builds the Tomoz image from this repository
```

- Orthanc: <http://127.0.0.1:8042>, user `admin`, password `ORTHANC_PASSWORD`
  from `.env`; DICOM port 4242.
- Tomoz metrics: <http://127.0.0.1:9100/_tomoz/metrics>.

Every port is bound to `127.0.0.1`. Set `ORTHANC_HTTP_PORT`,
`ORTHANC_DICOM_PORT` or `TOMOZ_PORT` in `.env` to use other ports.

## Try it

Upload a series through Orthanc's REST API (any uncompressed DICOM files):

```sh
. ./.env
for f in /path/to/series/*.dcm; do
  curl -fsS -u "admin:$ORTHANC_PASSWORD" --data-binary @"$f" http://127.0.0.1:8042/instances > /dev/null
done
```

Thirty seconds after the last file, the gateway compacts the series:

```sh
curl -s http://127.0.0.1:9100/_tomoz/metrics | grep -E '^tomoz_(archived_object_bytes|archive_bytes|compactions_total) '
docker compose logs tomoz | grep "compacted series"
```

The storage saving is `1 − tomoz_archive_bytes / tomoz_archived_object_bytes`.
Downloading an instance from Orthanc (`/instances/<id>/file`) returns the
uploaded bytes exactly.

Tested with Orthanc 1.13.0 (image 26.9.1, AWS S3 storage plugin 2.5.4): a
LIDC-IDRI CT series of 205 instances (107.9 MB) uploaded through Orthanc was
compacted 30 s later into one 46.1 MB archive; after restarting Orthanc (to
empty its own cache), all 205 instances read back through the gateway were
identical to the uploaded files, and Orthanc rendered them normally.

## Notes

- Orthanc must store files as received: `StorageCompression` is off and no
  ingest transcoding is configured, otherwise the gateway would receive
  zlib-compressed or transcoded objects that it can only store as they are.
- The credentials file is world-readable inside this example directory
  because Compose bind-mounts it with its host permissions and the gateway
  runs as an unprivileged user in its container; in production use your
  orchestrator's secrets.
- `docker compose down -v` removes the containers and both volumes.
