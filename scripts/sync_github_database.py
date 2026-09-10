#!/usr/bin/env python3
"""Download and verify the latest successful GitHub Actions prediction database.

The downloaded database is never copied over the working database.  A GitHub
token is required because GitHub's artifact download endpoint returns 401 even
for public repositories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import urllib.error
import urllib.request
import zipfile


API = "https://api.github.com"


def request_json(url: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "00631L-db-preflight/20260908.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def database_summary(path: Path) -> dict[str, object]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = con.execute("PRAGMA foreign_key_check").fetchall()
        count, max_id, latest = con.execute(
            "SELECT COUNT(*), MAX(id), MAX(predicted_at) FROM predictions"
        ).fetchone()
    if integrity != "ok" or foreign_keys:
        raise RuntimeError(f"downloaded database failed integrity checks: {integrity}, foreign_keys={foreign_keys}")
    return {"path": str(path), "count": count, "max_id": max_id, "latest_predicted_at": latest,
            "integrity_check": integrity, "foreign_key_violations": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="s1710931008/yfinance")
    parser.add_argument("--workflow", default="predict.yml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-database", default="predictions.sqlite3")
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    runs = request_json(
        f"{API}/repos/{args.repository}/actions/workflows/{args.workflow}/runs?status=success&per_page=1",
        token,
    ).get("workflow_runs", [])
    if not runs:
        raise RuntimeError("no successful workflow run found")
    run = runs[0]
    artifacts = request_json(run["artifacts_url"], token).get("artifacts", [])
    candidates = [a for a in artifacts if a["name"].startswith("00631L-analysis-") and not a["expired"]]
    if not candidates:
        raise RuntimeError("latest successful run has no unexpired analysis artifact")
    if not token:
        raise RuntimeError("artifact located but download requires GH_TOKEN or GITHUB_TOKEN; no local database was changed")

    output = Path(args.output_dir).resolve() / f"run-{run['id']}"
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "artifact.zip"
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
               "User-Agent": "00631L-db-preflight/20260908.1"}
    try:
        with urllib.request.urlopen(urllib.request.Request(candidates[0]["archive_download_url"], headers=headers), timeout=60) as response:
            archive.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"artifact download failed with HTTP {exc.code}; no local database was changed") from exc
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(output / "artifact")
    remote = output / "artifact" / "predictions.sqlite3"
    if not remote.exists():
        raise RuntimeError("artifact does not contain predictions.sqlite3")
    report = {"authoritative_source": "latest successful GitHub Actions analysis artifact",
              "run_id": run["id"], "run_url": run["html_url"], "head_sha": run["head_sha"],
              "remote": database_summary(remote), "working_database_replaced": False}
    local = Path(args.local_database)
    if local.exists():
        report["local"] = database_summary(local)
        report["remote_is_newer"] = (report["remote"]["max_id"] or 0) > (report["local"]["max_id"] or 0)
    report_path = output / "preflight-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

