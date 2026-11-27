"""Acquire papers and pinned upstream source with license metadata; no weights."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import io
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    last = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(15, 90), headers={"User-Agent": "RePAR-baseline-source-audit/1"})
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"{url}: {last}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--papers-only", action="store_true")
    parser.add_argument("--refresh-upstream", action="store_true",
                        help="Explicitly resolve a new upstream HEAD; never use for a frozen run")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "baselines/catalog.json").read_text(encoding="utf-8"))
    for name, spec in catalog.items():
        if not isinstance(spec, dict) or (args.methods and name not in args.methods):
            continue
        target = ROOT / "baselines" / name
        target.mkdir(parents=True, exist_ok=True)
        lock = target / "sources.json"
        previous = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {}
        manifest = {**previous, **spec, "checked_at_utc": datetime.now(timezone.utc).isoformat()}
        paper = target / "paper.pdf"
        if spec.get("arxiv"):
            url = f"https://arxiv.org/pdf/{spec['arxiv']}"
            try:
                content = paper.read_bytes() if paper.exists() else fetch(url).content
                if not content.startswith(b"%PDF"):
                    raise ValueError("Response is not a PDF")
                expected_paper = previous.get('paper', {})
                if expected_paper.get('url') == url and expected_paper.get('sha256') and sha(content) != expected_paper['sha256']:
                    raise ValueError('Paper hash differs from the locked source; inspect explicitly instead of rewriting provenance')
                paper.write_bytes(content)
                manifest["paper"] = {"url": url, "sha256": sha(content), "bytes": len(content)}
            except Exception as exc:
                manifest["paper_error"] = str(exc)
        if spec.get("github") and not args.papers_only:
            try:
                repo = spec["github"]
                frozen = previous.get("upstream", {})
                if frozen.get("repository") == f"https://github.com/{repo}" and not args.refresh_upstream:
                    commit = frozen["commit"]
                    metadata = {"license": frozen.get("license")}
                else:
                    metadata = fetch(f"https://api.github.com/repos/{repo}").json()
                    branch = metadata["default_branch"]
                    commit = fetch(f"https://api.github.com/repos/{repo}/commits/{branch}").json()["sha"]
                archive_files = None
                if frozen.get('commit') == commit and frozen.get('files') and not args.refresh_upstream:
                    tree = {'tree': [{'path': x['path'], 'type': 'blob', 'size': x['bytes']} for x in frozen['files']]}
                else:
                    try:
                        tree = fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1").json()
                    except RuntimeError:
                        # GitHub's anonymous metadata endpoint can be rate-limited.
                        # The commit-addressed official archive is the same source.
                        archive = zipfile.ZipFile(io.BytesIO(fetch(f'https://codeload.github.com/{repo}/zip/{commit}').content))
                        archive_files = {x.filename.split('/', 1)[1]: x for x in archive.infolist()
                                         if not x.is_dir() and '/' in x.filename}
                        tree = {'tree': [{'path': p, 'type': 'blob', 'size': x.file_size} for p, x in archive_files.items()]}
                if tree.get("truncated"):
                    raise ValueError("GitHub returned a truncated tree")
                excluded = {".git", "data", "datasets", "dataset", "outputs", "results", "assets", "figures", "images", "node_modules", "venv", ".venv"}
                def include(item):
                    p = Path(item["path"])
                    return (item["type"] == "blob" and not excluded.intersection(p.parts)
                            and item.get("size", 0) <= 2_000_000
                            and (p.suffix in {".py", ".toml", ".txt", ".md", ".json", ".yaml", ".yml"}
                                 or p.name.upper().startswith(("LICENSE", "NOTICE"))))
                selected = [item for item in tree["tree"] if include(item)]
                source = ROOT / "third_party/baselines" / name / commit
                source.mkdir(parents=True, exist_ok=True)
                def download(item):
                    rel = item["path"]
                    dest = source / rel
                    dest.resolve().relative_to(source.resolve())
                    content = dest.read_bytes() if dest.exists() else (archive.read(archive_files[rel]) if archive_files is not None
                        else fetch(f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}").content)
                    expected = next((x.get('sha256') for x in frozen.get('files', []) if x['path'] == rel), None)
                    if frozen.get('commit') == commit and expected and sha(content) != expected:
                        raise ValueError(f'Upstream hash mismatch: {rel}')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                    return {"path": rel, "sha256": sha(content), "bytes": len(content)}
                with ThreadPoolExecutor(max_workers=8) as pool:
                    files = list(pool.map(download, selected))
                manifest["upstream"] = {"repository": f"https://github.com/{repo}", "commit": commit,
                    "license": metadata.get("license"), "source_dir": source.relative_to(ROOT).as_posix(),
                    "scope": "Source/config/license snapshot, excludes datasets, generated outputs and weights", "files": files}
            except Exception as exc:
                manifest["upstream_error"] = str(exc)
        (target / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"method": name, "paper": manifest.get("paper", {}).get("bytes"), "commit": manifest.get("upstream", {}).get("commit"),
                          "files": len(manifest.get("upstream", {}).get("files", [])),
                          "errors": [manifest[k] for k in ("paper_error", "upstream_error") if k in manifest]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
