"""Tigris downloads. Objects are written only under the dashboard/ prefix."""

from __future__ import annotations

import os

import boto3
from botocore.config import Config
from flask import abort

PREFIX = "dashboard/"
FORBIDDEN = "rl/model-release/"


def configured() -> bool:
    needed = (
        "TIGRIS_ENDPOINT",
        "TIGRIS_BUCKET",
        "TIGRIS_ACCESS_KEY_ID",
        "TIGRIS_SECRET_ACCESS_KEY",
    )
    return all(os.environ.get(name) for name in needed)


def client():
    if not configured():
        abort(503, "Tigris keys are missing from dashboard/.env")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["TIGRIS_ENDPOINT"],
        aws_access_key_id=os.environ["TIGRIS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["TIGRIS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("TIGRIS_REGION", "auto"),
        config=Config(s3={"addressing_style": "virtual"}),
    )


def bucket() -> str:
    return os.environ["TIGRIS_BUCKET"]


def guard_key(key: str) -> str:
    if not key or not key.startswith(PREFIX) or FORBIDDEN in key:
        abort(400, "refusing a key outside dashboard/")
    return key


def upload(path: str, key: str) -> None:
    guard_key(key)
    if FORBIDDEN in path.replace("\\", "/"):
        abort(400, "refusing to read rl/model-release/")
    s3 = client()
    try:
        s3.upload_file(path, bucket(), key)
    except Exception as exc:
        raise RuntimeError(type(exc).__name__) from None


def open_object(key: str, range_header: str | None):
    guard_key(key)
    kwargs = {"Bucket": bucket(), "Key": key}
    if range_header:
        kwargs["Range"] = range_header
    s3 = client()
    try:
        return s3.get_object(**kwargs)
    except Exception as exc:
        raise RuntimeError(type(exc).__name__) from None
