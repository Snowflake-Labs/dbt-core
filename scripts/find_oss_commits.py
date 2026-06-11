#!/usr/bin/env python3
"""Identify OSS rebase targets and report dependency issues against the Snowflake conda channel.

Given a dbt-core version:
  1. Find the latest dbt-snowflake commit on the stable branch (same major.minor).
  2. Read the dbt-adapters version from dbt-adapters/CHANGELOG.md at that commit.
  3. Find the latest dbt-common commit on main.
  4. Look up the tagged dbt-core commit.

All dbt package versions are treated as pre-release (e.g. 1.22.3 -> 1.22.3b0) because
Snowflake forks append a 'b' suffix before publishing to the internal conda channel.

For dependencies other than dbt-core/common/adapters/snowflake, checks availability in
https://repo.anaconda.com/pkgs/snowflake/ and reports any issues.

Usage:
    python scripts/find_oss_commits.py 1.11.5
"""

import argparse
import base64
import json
import re
import subprocess
import sys
import urllib.request
from functools import lru_cache
from urllib.parse import urlencode

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

CONDA_CHANNEL = "https://repo.anaconda.com/pkgs/snowflake"
CONDA_SUBDIRS = ["noarch", "linux-64", "linux-aarch64"]
DBT_PACKAGES = frozenset({"dbt-core", "dbt-common", "dbt-adapters", "dbt-snowflake"})


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def gh_api(endpoint: str, **params) -> object:
    url = f"{endpoint}?{urlencode(params)}" if params else endpoint
    r = subprocess.run(["gh", "api", url], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"GitHub API error ({url}): {r.stderr.strip()}")
    return json.loads(r.stdout)


def get_file_at_commit(repo: str, path: str, sha: str) -> str | None:
    try:
        data = gh_api(f"repos/{repo}/contents/{path}", ref=sha)
        return base64.b64decode(data["content"]).decode()
    except Exception:
        return None


def find_tagged_commit(repo: str, tag: str) -> str:
    ref = gh_api(f"repos/{repo}/git/ref/tags/{tag}")
    sha = ref["object"]["sha"]
    if ref["object"]["type"] == "tag":
        sha = gh_api(f"repos/{repo}/git/tags/{sha}")["object"]["sha"]
    return sha


def find_latest_commit(repo: str, branch: str, pattern: re.Pattern) -> tuple[str, str] | None:
    """Return (sha, title) for the most recent commit matching pattern."""
    for page in range(1, 11):
        try:
            commits = gh_api(f"repos/{repo}/commits", sha=branch, per_page=100, page=page)
        except RuntimeError as e:
            print(f"  Warning: {e}", file=sys.stderr)
            break
        if not commits:
            break
        for c in commits:
            title = c["commit"]["message"].split("\n")[0]
            if pattern.search(title):
                return c["sha"], title
    return None


def find_stable_branch(repo: str, major: str, minor: str) -> str | None:
    for candidate in (
        "stable",
        f"{major}.{minor}.latest",
        f"{major}.{minor}.patch",
        f"dbt-snowflake-{major}.{minor}.latest",
    ):
        try:
            gh_api(f"repos/{repo}/git/ref/heads/{candidate}")
            return candidate
        except RuntimeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def extract_version(title: str) -> str | None:
    m = re.search(r"(\d+\.\d+(?:\.\d+)?(?:[ab]\d*|rc\d+)?)", title)
    return m.group(1) if m else None


def as_prerelease(version: str) -> str:
    """Return the pre-release (beta) form of a version string.
    '1.22.3' -> '1.22.3b0', '1.22.3b' -> '1.22.3b0', '1.22.3b1' -> '1.22.3b1'.
    """
    if re.search(r"[ab]\d+$|rc\d+$", version):
        return version  # already has numeric pre-release suffix
    v = re.sub(r"b$", "b0", version)
    v = re.sub(r"a$", "a0", v)
    if not re.search(r"[ab]|rc", v):
        v += "b0"
    return v


def display_ver(version: str) -> str:
    """Strip the internal b0 suffix for human-readable output."""
    return re.sub(r"b0$", "", version)


# ---------------------------------------------------------------------------
# pyproject.toml / CHANGELOG.md parsing
# ---------------------------------------------------------------------------


def parse_deps(content: str) -> list[str]:
    try:
        return tomllib.loads(content).get("project", {}).get("dependencies", [])
    except Exception:
        pass
    m = re.search(r"\[project\].*?dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def read_version_from_changelog(content: str) -> str | None:
    """Extract the first version header from a CHANGELOG.md."""
    m = re.search(r"^##+[^#\n]*?(\d+\.\d+\.\d+(?:[ab]\d*|rc\d+)?)", content, re.MULTILINE)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Conda channel
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_conda_packages() -> dict[str, list[str]]:
    pkgs: dict[str, list[str]] = {}
    for subdir in CONDA_SUBDIRS:
        url = f"{CONDA_CHANNEL}/{subdir}/repodata.json"
        print(f"  Fetching {url} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
        except Exception as e:
            print(f"  Warning: {e}", file=sys.stderr)
            continue
        for section in ("packages", "packages.conda"):
            for entry in data.get(section, {}).values():
                name = entry.get("name", "").lower().replace("_", "-")
                ver = entry.get("version", "")
                if name and ver:
                    pkgs.setdefault(name, []).append(ver)
    return pkgs


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------


def check_dep(spec: str, known: dict[str, str]) -> str | None:
    """Return a failure message, or None if the dep is satisfied."""
    try:
        req = Requirement(spec)
    except Exception:
        return None
    if not req.specifier:
        return None

    pkg = req.name.lower().replace("_", "-")

    if pkg in DBT_PACKAGES:
        if pkg not in known:
            return None  # deploying it; skip
        if not req.specifier.contains(known[pkg], prereleases=True):
            return f"{pkg}=={display_ver(known[pkg])} does not satisfy '{spec}'"
        return None

    conda = get_conda_packages()
    available = conda.get(pkg, [])
    if not available:
        return f"'{pkg}' not found in conda channel"
    if not any(Version(v) in req.specifier for v in available):
        recent = sorted(set(available))[-3:]
        return f"no conda version satisfies '{spec}' (newest available: {recent})"
    return None


def report_deps(label: str, dep_specs: list[str], known: dict[str, str]) -> bool:
    """Print per-dep results; return True if any issues found."""
    issues = [msg for spec in dep_specs if (msg := check_dep(spec, known))]
    if issues:
        print(f"  {label}: {len(issues)} issue(s)")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  {label}: OK")
    return bool(issues)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="dbt-core version, e.g. 1.11.5")
    args = parser.parse_args()

    version = args.version.lstrip("v")
    major, minor = version.split(".")[:2]

    print(f"\nRebase targets for dbt-core=={version}\n")
    print("Fetching conda repodata (cached for this run):")
    get_conda_packages()

    # 1. Latest dbt-snowflake on stable branch
    stable_branch = find_stable_branch("dbt-labs/dbt-adapters", major, minor)
    sf_branch = stable_branch or "main"
    if stable_branch:
        sf_pattern = re.compile(rf"\[Automated\] Generate changelog for {major}\.{minor}\.", re.I)
    else:
        sf_pattern = re.compile(rf"\[Automated\] Publish dbt-snowflake=={major}\.{minor}\.", re.I)
    print(f"\n=== dbt-snowflake ({sf_branch}) ===")
    sf_result = find_latest_commit("dbt-labs/dbt-adapters", sf_branch, sf_pattern)
    if not sf_result:
        print(
            f"ERROR: no dbt-snowflake {major}.{minor}.x commit on {sf_branch!r}", file=sys.stderr
        )
        sys.exit(1)
    sf_sha, sf_title = sf_result
    sf_ver = as_prerelease(extract_version(sf_title) or "")
    print(f"  {sf_sha}  {sf_title!r}")
    print(f"  dbt-snowflake=={display_ver(sf_ver)}")
    print(f"  https://github.com/dbt-labs/dbt-adapters/commit/{sf_sha}")

    # 2. dbt-adapters version from CHANGELOG.md at the snowflake commit
    adapters_ver: str | None = None
    changelog = get_file_at_commit("dbt-labs/dbt-adapters", "dbt-adapters/CHANGELOG.md", sf_sha)
    if changelog:
        raw = read_version_from_changelog(changelog)
        if raw:
            adapters_ver = as_prerelease(raw)
    if adapters_ver:
        print(f"  dbt-adapters=={display_ver(adapters_ver)}  (from dbt-adapters/CHANGELOG.md)")
    else:
        print("  Warning: could not read dbt-adapters version from CHANGELOG.md")

    # 3. Latest dbt-common on main
    print("\n=== dbt-common (main) ===")
    common_result = find_latest_commit(
        "dbt-labs/dbt-common",
        "main",
        re.compile(r"Bumping version to \d+\.\d+\.\d+ and generate changelog", re.I),
    )
    if not common_result:
        print("ERROR: no dbt-common commit found.", file=sys.stderr)
        sys.exit(1)
    common_sha, common_title = common_result
    common_ver = as_prerelease(extract_version(common_title) or "")
    print(f"  {common_sha}  {common_title!r}")
    print(f"  dbt-common=={display_ver(common_ver)}")
    print(f"  https://github.com/dbt-labs/dbt-common/commit/{common_sha}")

    # 4. dbt-core tagged commit
    print(f"\n=== dbt-core v{version} ===")
    core_sha = find_tagged_commit("dbt-labs/dbt-core", f"v{version}")
    print(f"  {core_sha}")
    print(f"  https://github.com/dbt-labs/dbt-core/commit/{core_sha}")

    # Version snapshot used for dep checks
    known: dict[str, str] = {"dbt-common": common_ver, "dbt-snowflake": sf_ver}
    if adapters_ver:
        known["dbt-adapters"] = adapters_ver

    # 5. Check deps for all four packages and report issues
    print(f"\n{'=' * 60}")
    print("Version snapshot:")
    for k, v in sorted(known.items()):
        print(f"  {k}=={display_ver(v)}")

    checks = [
        ("dbt-common", common_sha, "dbt-labs/dbt-common", "pyproject.toml"),
        ("dbt-adapters", sf_sha, "dbt-labs/dbt-adapters", "dbt-adapters/pyproject.toml"),
        ("dbt-snowflake", sf_sha, "dbt-labs/dbt-adapters", "dbt-snowflake/pyproject.toml"),
        ("dbt-core", core_sha, "dbt-labs/dbt-core", "core/pyproject.toml"),
    ]

    print("\nDependency check:")
    any_issues = False
    for label, sha, repo, path in checks:
        content = get_file_at_commit(repo, path, sha)
        if not content:
            print(f"  {label}: could not read {path}")
            continue
        any_issues |= report_deps(label, parse_deps(content), known)

    print(f"\n{'=' * 60}")
    print(f"dbt-common:    https://github.com/dbt-labs/dbt-common/commit/{common_sha}")
    print(f"dbt-snowflake: https://github.com/dbt-labs/dbt-adapters/commit/{sf_sha}")
    print(f"dbt-core:      https://github.com/dbt-labs/dbt-core/commit/{core_sha}")
    if any_issues:
        print("\n** Some dependencies have issues — see above. **")
    else:
        print("\nAll dependencies satisfied.")


if __name__ == "__main__":
    main()
