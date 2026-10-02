"""Compatibility check of the Tomoz gateway with a real S3 client (boto3).

Usage: python gateway_compat.py ENDPOINT ACCESS_KEY SECRET (DICOM_DIR | synthetic:N)

Uploads the DICOM files of DICOM_DIR (a series), or N slices of a synthetic
series, exercises the S3 operations the gateway supports, waits for the
series to be compacted and checks that every object still reads back byte
for byte. The gateway must have a bucket named "dicom".
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def main() -> None:
    endpoint, key, secret, folder = sys.argv[1:5]
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 1}),
    )
    if folder.startswith("synthetic:"):
        from tomoz_lab.synthetic import ct_series

        originals = {f"series/{name}": data for name, data in ct_series(int(folder.split(":")[1]))}
    else:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix == ".dcm")[:40]
        originals = {f"series/{p.name}": p.read_bytes() for p in files}

    assert "dicom" in [b["Name"] for b in s3.list_buckets()["Buckets"]]
    s3.create_bucket(Bucket="scratch")
    for k, data in originals.items():
        s3.put_object(Bucket="dicom", Key=k, Body=data, ContentType="application/dicom")
    print(f"put {len(originals)} objects")

    k0 = next(iter(originals))
    head = s3.head_object(Bucket="dicom", Key=k0)
    assert head["ContentLength"] == len(originals[k0])
    assert head["ETag"].strip('"') == hashlib.md5(originals[k0]).hexdigest()
    assert s3.get_object(Bucket="dicom", Key=k0)["Body"].read() == originals[k0]
    part = s3.get_object(Bucket="dicom", Key=k0, Range="bytes=128-131")["Body"].read()
    assert part == b"DICM", part

    keys, token = [], None
    while True:
        kw = {"Bucket": "dicom", "Prefix": "series/", "MaxKeys": 7}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in page.get("Contents", [])]
        if not page["IsTruncated"]:
            break
        token = page["NextContinuationToken"]
    assert keys == sorted(originals), (len(keys), len(originals))
    top = s3.list_objects_v2(Bucket="dicom", Delimiter="/")
    assert [p["Prefix"] for p in top.get("CommonPrefixes", [])] == ["series/"]
    print("listing ok")

    big = os.urandom(12 << 20)
    from boto3.s3.transfer import TransferConfig

    s3.upload_fileobj(
        __import__("io").BytesIO(big),
        "scratch",
        "big.bin",
        Config=TransferConfig(multipart_threshold=5 << 20, multipart_chunksize=5 << 20),
    )
    got = s3.get_object(Bucket="scratch", Key="big.bin")
    assert got["Body"].read() == big and got["ETag"].endswith('-3"'), got["ETag"]
    s3.copy_object(Bucket="scratch", Key="copy.bin", CopySource={"Bucket": "scratch", "Key": "big.bin"})
    assert s3.get_object(Bucket="scratch", Key="copy.bin")["Body"].read() == big
    r = s3.delete_objects(
        Bucket="scratch", Delete={"Objects": [{"Key": "big.bin"}, {"Key": "copy.bin"}, {"Key": "missing"}]}
    )
    assert len(r["Deleted"]) == 3
    print("multipart, copy and batch delete ok")

    try:
        s3.get_object(Bucket="dicom", Key="series/none")
        raise AssertionError("expected NoSuchKey")
    except ClientError as e:
        assert e.response["Error"]["Code"] == "NoSuchKey"
    bad = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key="wrong-secret-0000",
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 1}),
    )
    try:
        bad.list_buckets()
        raise AssertionError("expected SignatureDoesNotMatch")
    except ClientError as e:
        assert e.response["Error"]["Code"] == "SignatureDoesNotMatch", e.response["Error"]
    print("errors ok")

    deadline = time.time() + 120
    while True:
        h = s3.head_object(Bucket="dicom", Key=k0)
        if h["ResponseMetadata"]["HTTPHeaders"].get("x-tomoz-storage") == "archive":
            break
        if time.time() > deadline:
            raise AssertionError("series was not compacted")
        time.sleep(1)
    t0 = time.time()
    for k, data in originals.items():
        assert s3.get_object(Bucket="dicom", Key=k)["Body"].read() == data, k
    print(f"compacted: all {len(originals)} objects restored byte for byte in {time.time() - t0:.2f} s")
    s3.delete_object(Bucket="dicom", Key=k0)
    s3.put_object(Bucket="dicom", Key=k0, Body=originals[k0])
    assert s3.get_object(Bucket="dicom", Key=k0)["Body"].read() == originals[k0]
    print("overwrite after compaction ok")


if __name__ == "__main__":
    main()
