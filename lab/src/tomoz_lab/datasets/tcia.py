"""Minimal client for the public NBIA REST API of The Cancer Imaging Archive.

Only public collections are used; no account or token is needed.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
USER_AGENT = "tomoz-lab/0.1 (+https://github.com/ZeroXShot/tomoz)"

log = logging.getLogger(__name__)


def _get(path: str, timeout: float = 600.0, retries: int = 4, **params: str) -> bytes:
    url = API + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt + 1 == retries:
                raise
            wait = 5 * 2**attempt
            log.warning("TCIA request failed (%s), retrying in %ds", e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def series(collection: str, modality: str | None = None) -> list[dict]:
    """Series of a collection, with ImageCount, PatientID, LicenseName, ..."""
    params = {"Collection": collection, "format": "json"}
    if modality:
        params["Modality"] = modality
    data = _get("getSeries", **params)
    return json.loads(data) if data.strip() else []


def download_series(series_uid: str) -> dict[str, bytes]:
    """All files of a series, by name inside the zip returned by the API."""
    data = _get("getImage", SeriesInstanceUID=series_uid)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {name: z.read(name) for name in z.namelist() if not name.endswith("/")}
