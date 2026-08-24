"""Canonical serialization and tamper-evident logging helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


class HashChainWriter:
    """Write canonical JSONL records linked by SHA-256."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self.previous_hash = "0" * 64
        self.count = 0

    def write(self, payload: dict) -> str:
        body = dict(payload)
        if "previous_record_sha256" in body or "record_sha256" in body:
            raise ValueError("hash-chain fields are reserved")
        body["previous_record_sha256"] = self.previous_hash
        digest = sha256_bytes(canonical_json_bytes(body))
        record = {**body, "record_sha256": digest}
        self._handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
        self._handle.flush()
        self.previous_hash = digest
        self.count += 1
        return digest

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def verify_hash_chain(path: Path) -> dict:
    previous = "0" * 64
    count = 0
    errors = []
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            stored = record.pop("record_sha256", None)
            linked = record.get("previous_record_sha256")
            computed = sha256_bytes(canonical_json_bytes(record))
            if linked != previous:
                errors.append(f"line {line_number}: previous hash mismatch")
            if stored != computed:
                errors.append(f"line {line_number}: record hash mismatch")
            previous = stored or ""
            count += 1
            records.append({**record, "record_sha256": stored})
    return {
        "passed": not errors,
        "record_count": count,
        "terminal_record_sha256": previous,
        "errors": errors,
        "records": records,
    }


def verify_paths(entries: Iterable[dict], root: Path) -> list[dict]:
    checks = []
    for entry in entries:
        path = Path(root) / entry["path"]
        actual = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": entry["path"],
                "expected_sha256": str(entry["sha256"]).upper(),
                "actual_sha256": actual,
                "passed": actual == str(entry["sha256"]).upper(),
            }
        )
    return checks
