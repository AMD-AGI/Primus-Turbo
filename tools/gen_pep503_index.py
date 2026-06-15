#!/usr/bin/env python3
"""Generate a PEP 503 "simple" repository index for GitHub Pages.

Scans the repo's GitHub Releases, collects distribution files and emits a static
index that links to the release assets. Designed to run in CI (uses
GITHUB_TOKEN) and have its output deployed to GitHub Pages.

Layout produced (phase 1: sdist only):

    <output>/.nojekyll
    <output>/simple/index.html
    <output>/simple/primus-turbo/index.html

Wheels carry a variant local version (e.g. +rocm7.1torch2.10) and will be routed
into /whl/<rocm>/<torch>/ in phase 2 (see route_wheel TODO below); for now only
sdist (.tar.gz) is indexed under /simple/.

Yanking (PEP 592): list filenames (or exact versions) in tools/yanked.txt, one
per line; matching files get a data-yanked attribute so pip skips them for new
resolves while pinned installs still work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

API_ROOT = "https://api.github.com"


def normalize_project_name(name: str) -> str:
    # PEP 503 normalization.
    return re.sub(r"[-_.]+", "-", name).lower()


def gh_get(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def fetch_releases(repo: str, token: str | None) -> list[dict]:
    releases: list[dict] = []
    page = 1
    while True:
        url = f"{API_ROOT}/repos/{repo}/releases?per_page=100&page={page}"
        try:
            data = json.loads(gh_get(url, token))
        except urllib.error.HTTPError as exc:  # pragma: no cover
            sys.exit(f"GitHub API error fetching releases: {exc}")
        if not data:
            break
        releases.extend(data)
        page += 1
    return releases


def load_yanked(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def is_yanked(filename: str, version: str, yanked: set[str]) -> bool:
    return filename in yanked or version in yanked


def version_from_filename(filename: str) -> str:
    # primus_turbo-0.3.1.tar.gz -> 0.3.1 ; best-effort, only for yank matching.
    m = re.match(r"^[A-Za-z0-9_.]+?-([0-9][^-]*?)(?:\.tar\.gz|-.*\.whl)$", filename)
    return m.group(1) if m else ""


def collect_files(releases: list[dict], exts: tuple[str, ...]) -> list[dict]:
    files: list[dict] = []
    for rel in releases:
        if rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(exts):
                files.append({"name": name, "url": asset.get("browser_download_url", "")})
    # Stable, readable ordering.
    files.sort(key=lambda f: f["name"])
    return files


def render_project_index(project: str, files: list[dict], yanked: set[str]) -> str:
    anchors = []
    for f in files:
        name, url = f["name"], f["url"]
        ver = version_from_filename(name)
        attr = ' data-yanked=""' if is_yanked(name, ver, yanked) else ""
        anchors.append(f'    <a href="{escape(url)}"{attr}>{escape(name)}</a><br>')
    body = "\n".join(anchors)
    return (
        "<!DOCTYPE html>\n<html>\n  <head>\n"
        '    <meta name="pypi:repository-version" content="1.0">\n'
        f"    <title>Links for {escape(project)}</title>\n"
        "  </head>\n  <body>\n"
        f"    <h1>Links for {escape(project)}</h1>\n"
        f"{body}\n"
        "  </body>\n</html>\n"
    )


def render_root_index(projects: list[str]) -> str:
    anchors = "\n".join(f'    <a href="{escape(p)}/">{escape(p)}</a><br>' for p in projects)
    return (
        "<!DOCTYPE html>\n<html>\n  <head>\n"
        '    <meta name="pypi:repository-version" content="1.0">\n'
        "    <title>Simple index</title>\n"
        "  </head>\n  <body>\n"
        f"{anchors}\n"
        "  </body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PEP 503 index for GitHub Pages.")
    parser.add_argument("--output", default="_site", help="output site directory")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/repo (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--project-name", default="primus-turbo")
    parser.add_argument("--yanked-file", default="tools/yanked.txt")
    args = parser.parse_args()

    if not args.repo:
        sys.exit("Repository not set; pass --repo or set GITHUB_REPOSITORY.")

    token = os.environ.get("GITHUB_TOKEN")
    project = normalize_project_name(args.project_name)
    yanked = load_yanked(Path(args.yanked_file))

    releases = fetch_releases(args.repo, token)
    # Phase 1: index sdist only. (Phase 2: route .whl into /whl/<rocm>/<torch>/.)
    sdists = collect_files(releases, (".tar.gz",))

    out = Path(args.output)
    simple = out / "simple"
    (simple / project).mkdir(parents=True, exist_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (simple / "index.html").write_text(render_root_index([project]), encoding="utf-8")
    (simple / project / "index.html").write_text(
        render_project_index(project, sdists, yanked), encoding="utf-8"
    )

    print(f"Indexed {len(sdists)} sdist file(s) for '{project}' from {args.repo}.")
    if yanked:
        print(f"Yanked entries applied: {sorted(yanked)}")


if __name__ == "__main__":
    main()
