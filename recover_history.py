#!/usr/bin/env python3
"""
recover_history.py — Reconstruct pre-git file history from Claude Code file-history
                     and VS Code Local History.

Usage:
    python3 recover_history.py --list                    # show tracked files + version counts
    python3 recover_history.py --diffs                   # coloured diffs in terminal
    python3 recover_history.py --diffs --file src/model.py
    python3 recover_history.py --report                  # write history_report/history.html + .txt
    python3 recover_history.py --scrub-check             # preview what would be redacted
    python3 recover_history.py --merge-commits           # build clean git history (asks to confirm)
"""

import os
import re
import sys
import json
import difflib
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

HISTORY_ROOT   = Path.home() / ".claude/file-history"
VSCODE_HISTORY = Path.home() / ".config/Code/User/History"
PROJECT_ROOT   = Path("/home/superws/Downloads/Papers/Codes/self-medrag-production")
REPORT_DIR     = PROJECT_ROOT / "history_report"

# Sensitive patterns: (regex, replacement)
SENSITIVE_PATTERNS = [
    (r'sk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{80,}', 'sk-ant-REDACTED'),
    (r'hf_[A-Za-z0-9]{30,}',                     'hf_REDACTED'),
    (r'OmxpkRUGf6[A-Za-z0-9]+',                  'MISTRAL_KEY_REDACTED'),
    (r'ghp_[A-Za-z0-9]{30,}',                     'ghp_REDACTED'),
    (r'sk-proj-[A-Za-z0-9_\-]{40,}',             'sk-proj-REDACTED'),
]
_COMPILED = [(re.compile(p), r) for p, r in SENSITIVE_PATTERNS]

# content-hash → project-relative filename (verified by reading file content)
HASH_TO_FILE = {
    "800c3c98e41ae699": "src/model.py",
    "5a2334b5209211ce": "src/trainer.py",
    "c0992b01d60df6f2": "src/retrieval.py",
    "20fad3caf3884d6f": "src/pipeline.py",
    "9b584c2977469b46": "config.yaml",
    "f38ec96fa879fd21": ".env.template",
    "1655afbc28d77069": "SETUP_GUIDE.md",
    "a92e5e4bd0ec89f2": "scripts/query.py",
    "ce34f88a06219c1f": "scripts/diagnose_generation.py",
    "d6e99a89166d478e": "memory/project_state.md",
    "41b4d580ad400c35": "models/qwen2.5-1.5b-4bit/config.json",
    "7ea40321a8610b31": "chat.py",
}

# Claude Code file-history UUID folders that belong to this project
PROJECT_UUIDS = {
    "a196d865-ac1b-4e31-94e9-2a845c91c703",
    "da45bad6-968a-4874-8a16-bf654ee7be07",
}

# Never commit these into recovered history (local tooling / generated files)
COMMIT_EXCLUDE_PREFIXES = (
    "history_report/",
    "venv/",
    "models/",
    ".vscode/",
)

# ── Sensitive-data scrubbing ──────────────────────────────────────────────────

def scrub(text: str) -> str:
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text


def find_sensitive(text: str) -> list[tuple[str, str]]:
    hits = []
    for pattern, replacement in _COMPILED:
        for m in pattern.finditer(text):
            hits.append((m.group(), replacement))
    return hits

# ── Data loading ──────────────────────────────────────────────────────────────

def load_claude_history() -> dict:
    files: dict = {}
    for uuid_dir in HISTORY_ROOT.iterdir():
        if not uuid_dir.is_dir() or uuid_dir.name not in PROJECT_UUIDS:
            continue
        for entry in uuid_dir.iterdir():
            if "@" not in entry.name:
                continue
            hash_id = entry.name.split("@")[0]
            if hash_id not in HASH_TO_FILE or entry.stat().st_size == 0:
                continue
            rel = HASH_TO_FILE[hash_id]
            files.setdefault(rel, []).append((entry.stat().st_mtime, entry))
    return {f: sorted(v, key=lambda x: x[0]) for f, v in files.items()}


def load_vscode_history() -> dict:
    files: dict = {}
    if not VSCODE_HISTORY.exists():
        return files
    for folder in VSCODE_HISTORY.iterdir():
        entry_json = folder / "entries.json"
        if not entry_json.exists():
            continue
        try:
            data = json.loads(entry_json.read_text())
        except Exception:
            continue
        resource = data.get("resource", "")
        if str(PROJECT_ROOT) not in resource:
            continue
        rel = resource.replace(f"file://{PROJECT_ROOT}/", "")
        for entry in data.get("entries", []):
            vf = folder / entry["id"]
            if vf.exists() and vf.stat().st_size > 0:
                files.setdefault(rel, []).append((entry["timestamp"] / 1000, vf))
    return {f: sorted(v, key=lambda x: x[0]) for f, v in files.items()}


def merge_histories(claude: dict, vscode: dict) -> dict:
    merged: dict = {}
    for rel in set(claude) | set(vscode):
        versions = claude.get(rel, []) + vscode.get(rel, [])
        versions.sort(key=lambda x: x[0])
        deduped, prev = [], None
        for ts, path in versions:
            content = path.read_text(errors="replace")
            if content != prev:
                deduped.append((ts, path, content))
                prev = content
        merged[rel] = deduped
    return merged

# ── Diff helpers ──────────────────────────────────────────────────────────────

def make_diff(old, new, filename, old_label, new_label):
    return list(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"{filename} ({old_label})", tofile=f"{filename} ({new_label})",
        lineterm="",
    ))


def diff_stats(diff_lines):
    add = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    rm  = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
    return add, rm

# ── CLI actions ───────────────────────────────────────────────────────────────

def action_list(merged, claude, vscode):
    print(f"\n{'FILE':<42} {'VER':>4}  {'FIRST':>16}  {'LAST':>16}  SOURCE")
    print("-" * 92)
    for rel in sorted(merged):
        v = merged[rel]
        src = ("Claude+VSCode" if (rel in claude and rel in vscode)
               else "Claude" if rel in claude else "VSCode")
        print(f"  {rel:<40} {len(v):>4}  {fmt(v[0][0]):>16}  {fmt(v[-1][0]):>16}  {src}")
    print(f"\nTotal: {len(merged)} files, {sum(len(v) for v in merged.values())} versions")


def action_scrub_check(merged):
    print("\n── Sensitive data that will be REDACTED in --merge-commits ──\n")
    found_any = False
    for rel, versions in sorted(merged.items()):
        for ts, _, content in versions:
            hits = find_sensitive(content)
            if hits:
                found_any = True
                print(f"  {rel}  [{fmt(ts)}]")
                for raw, replacement in hits:
                    print(f"    {raw[:60]}...  →  {replacement}")
    if not found_any:
        print("  None found.")


def action_diffs(merged, filter_file=None):
    C = {"+": "\033[32m", "-": "\033[31m", "@": "\033[36m", "=": "\033[1m", "0": "\033[0m"}
    for rel, versions in sorted(merged.items()):
        if filter_file and rel != filter_file:
            continue
        print(f"\n{'='*70}\n  FILE: {rel}  ({len(versions)} versions)\n{'='*70}")
        if len(versions) < 2:
            print(f"  Only 1 version ({fmt(versions[0][0])}) — nothing to diff.")
            continue
        for i in range(1, len(versions)):
            old_ts, _, old_c = versions[i-1]
            new_ts, _, new_c = versions[i]
            diff = make_diff(old_c, new_c, rel, fmt(old_ts), fmt(new_ts))
            add, rm = diff_stats(diff)
            print(f"\n  ── v{i}→v{i+1}  {fmt(old_ts)} → {fmt(new_ts)}  [+{add}/-{rm}] ──")
            if not diff:
                print("  (no textual changes)")
                continue
            for line in diff:
                if   line.startswith("+++") or line.startswith("---"): c = C["="]
                elif line.startswith("+"):  c = C["+"]
                elif line.startswith("-"):  c = C["-"]
                elif line.startswith("@@"): c = C["@"]
                else:                       c = ""
                print(f"{c}{line}{C['0'] if c else ''}")


def action_report(merged):
    REPORT_DIR.mkdir(exist_ok=True)

    txt = REPORT_DIR / "history.txt"
    with open(txt, "w") as f:
        f.write("Self-MedRAG — Pre-git File History\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*70}\n\n")
        for rel, versions in sorted(merged.items()):
            f.write(f"\n{'='*70}\nFILE: {rel}  ({len(versions)} version(s))\n{'='*70}\n")
            for i, (ts, path, _) in enumerate(versions, 1):
                f.write(f"  v{i}: {fmt(ts)}  ({path})\n")
            for i in range(1, len(versions)):
                old_ts, _, old_c = versions[i-1]
                new_ts, _, new_c = versions[i]
                diff = make_diff(old_c, new_c, rel, fmt(old_ts), fmt(new_ts))
                add, rm = diff_stats(diff)
                f.write(f"\n--- v{i}→v{i+1}  {fmt(old_ts)} → {fmt(new_ts)}  [+{add}/-{rm}] ---\n")
                f.write("\n".join(diff) + "\n")

    html = REPORT_DIR / "history.html"
    with open(html, "w") as f:
        f.write(_html_head())
        for rel, versions in sorted(merged.items()):
            f.write(f'<h2>{rel} <span class="badge">{len(versions)} version(s)</span></h2>\n')
            f.write('<table class="meta"><tr>')
            for i, (ts, _, _) in enumerate(versions, 1):
                f.write(f'<td>v{i}: {fmt(ts)}</td>')
            f.write('</tr></table>\n')
            for i in range(1, len(versions)):
                old_ts, _, old_c = versions[i-1]
                new_ts, _, new_c = versions[i]
                diff = make_diff(old_c, new_c, rel, fmt(old_ts), fmt(new_ts))
                add, rm = diff_stats(diff)
                f.write(f'<details open><summary>v{i}→v{i+1} &nbsp; {fmt(old_ts)} → {fmt(new_ts)} '
                        f'&nbsp; <span class="add">+{add}</span> <span class="del">-{rm}</span>'
                        f'</summary><pre class="diff">')
                for line in diff:
                    cls = (' class="add"' if line.startswith("+") and not line.startswith("+++") else
                           ' class="del"' if line.startswith("-") and not line.startswith("---") else
                           ' class="hunk"' if line.startswith("@@") else "")
                    f.write(f"<span{cls}>{_he(line)}\n</span>")
                f.write("</pre></details>\n")
        f.write("</body></html>")

    print(f"\nReport written:\n  {txt}\n  {html}")


def action_merge_commits(merged):
    """
    Build a clean linear git history:
      [recovered historical commits, scrubbed]
      → [existing commits replayed with original timestamps + messages]

    Strategy:
      1. Tag current HEAD as a safety backup.
      2. Snapshot exact content of every file in existing commits (git show).
      3. Checkout --orphan branch, clear index.
      4. Apply each historical version (scrubbed, backdated).
      5. Replay existing commits (scrubbed, original timestamps/messages).
      6. Rename branch back to main.
    """
    # ── preflight ──
    print("\n── Preflight check ──")
    status = _git("status", "--porcelain").stdout.strip()
    if status:
        print("ERROR: Working tree has uncommitted changes. Commit or stash them first.")
        sys.exit(1)

    existing = _get_existing_commits()
    print(f"  Existing commits to replay : {len(existing)}")
    print(f"  Historical versions to add : {sum(len(v) for v in merged.values())}")

    # Preview sensitive hits
    hits_total = sum(len(find_sensitive(c)) for vs in merged.values() for _, _, c in vs)
    print(f"  Sensitive values to redact : {hits_total}")

    print()
    print("This will REWRITE the repository history on branch 'main'.")
    print("A backup tag 'pre-recovery-backup' will be created first.")
    ans = input("Type YES to continue: ").strip()
    if ans != "YES":
        print("Aborted.")
        return

    # ── step 1: backup tag ──
    _git("tag", "-f", "pre-recovery-backup", "HEAD")
    print("\n  [1/5] Backup tag created: pre-recovery-backup")

    # ── step 2: snapshot existing commits ──
    print("  [2/5] Snapshotting existing commits …")
    existing_snapshots = []
    for sha, date, msg in existing:
        files = _snapshot_commit(sha)
        existing_snapshots.append((sha, date, msg, files))
        print(f"        {sha[:7]}  {date[:16]}  {msg[:50]}  ({len(files)} files)")

    # ── step 3: collect + sort all historical versions (scrubbed, filtered) ──
    print("  [3/5] Collecting historical versions …")
    all_versions: list[tuple[float, str, str]] = []
    for rel, versions in merged.items():
        if any(rel.startswith(p) for p in COMMIT_EXCLUDE_PREFIXES):
            print(f"        skipping excluded path: {rel}")
            continue
        for ts, _, content in versions:
            all_versions.append((ts, rel, scrub(content)))
    all_versions.sort(key=lambda x: x[0])

    # ── step 4: create orphan branch and apply historical commits ──
    print("  [4/5] Building historical commits on orphan branch …")
    current_branch = _git("branch", "--show-current").stdout.strip() or "main"

    _git("checkout", "--orphan", "_recovered_history_tmp")
    _git("rm", "-rf", "--cached", ".")

    committed = 0
    for ts, rel, content in all_versions:
        dest = PROJECT_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8", errors="replace")
        _git("add", rel)

        if _git("diff", "--cached", "--quiet").returncode == 0:
            continue  # nothing new

        dt = _iso(ts)
        env = {**os.environ, "GIT_AUTHOR_DATE": dt, "GIT_COMMITTER_DATE": dt}
        _git("commit", "-m", f"[recovered] {rel} — {fmt(ts)}", env=env)
        committed += 1
        print(f"        + {fmt(ts)}  {rel}")

    print(f"        {committed} historical commits created.")

    # ── step 5: replay existing commits on top ──
    print("  [5/5] Replaying existing commits …")
    for sha, date, msg, files in existing_snapshots:
        for rel, raw_bytes in files.items():
            dest = PROJECT_ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            # scrub text files; leave binary as-is
            try:
                text = raw_bytes.decode("utf-8")
                dest.write_text(scrub(text), encoding="utf-8")
            except UnicodeDecodeError:
                dest.write_bytes(raw_bytes)

        _git("add", "-A")
        if _git("diff", "--cached", "--quiet").returncode == 0:
            print(f"        {sha[:7]} — no diff vs recovered history, skipping")
            continue

        env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        _git("commit", "-m", msg, env=env)
        print(f"        replayed {sha[:7]}  {date[:16]}  {msg[:50]}")

    # ── rename branch ──
    _git("branch", "-D", current_branch)
    _git("branch", "-m", current_branch)

    print(f"\nDone. Run:  git log --oneline --date=format:'%Y-%m-%d'")
    print("To push:   git push --force-with-lease origin main")
    print("To undo:   git reset --hard pre-recovery-backup")

# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(*args, env=None, check=False):
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT,
        capture_output=True, text=True, env=env, check=check,
    )


def _get_existing_commits() -> list[tuple[str, str, str]]:
    """Return [(sha, iso_date, message), ...] oldest-first."""
    result = _git("log", "--format=%H|||%aI|||%s", "--reverse")
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|||")
        if len(parts) == 3:
            rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def _snapshot_commit(sha: str) -> dict[str, bytes]:
    """Return {rel_path: bytes} for every file tracked in this commit."""
    ls = _git("ls-tree", "-r", "--name-only", sha)
    files = {}
    for rel in ls.stdout.strip().splitlines():
        content = subprocess.run(
            ["git", "show", f"{sha}:{rel}"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if content.returncode == 0:
            files[rel] = content.stdout
    return files


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

# ── Misc helpers ──────────────────────────────────────────────────────────────

def fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _he(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_head() -> str:
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<title>Self-MedRAG History</title><style>'
        'body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto}'
        'h2{margin-top:2em;border-bottom:2px solid #ccc;padding-bottom:.3em}'
        '.badge{background:#eee;border-radius:4px;padding:2px 7px;font-size:.8em}'
        'table.meta td{padding:2px 10px;font-size:.85em;color:#555}'
        'details{margin:.8em 0}'
        'summary{cursor:pointer;font-weight:bold;padding:.4em;background:#f5f5f5;'
        'border-radius:4px;list-style:none}'
        'summary .add{color:#2a7a2a} summary .del{color:#b00}'
        'pre.diff{margin:0;padding:.8em;background:#fafafa;border:1px solid #ddd;'
        'border-radius:0 0 4px 4px;overflow-x:auto;font-size:.82em;line-height:1.45}'
        'span.add{background:#e6ffed;display:block}'
        'span.del{background:#ffeef0;display:block}'
        'span.hunk{color:#00b;display:block}'
        f'</style></head><body>'
        f'<h1>Self-MedRAG — Pre-git File History</h1>'
        f'<p>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>\n'
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list",          action="store_true", help="List files and version counts")
    ap.add_argument("--scrub-check",   action="store_true", help="Preview sensitive data to redact")
    ap.add_argument("--diffs",         action="store_true", help="Print diffs to terminal")
    ap.add_argument("--report",        action="store_true", help="Write HTML + text report")
    ap.add_argument("--merge-commits", action="store_true",
                    help="Build clean git history (scrubbed, backdated) then replay existing commits")
    ap.add_argument("--file", help="Filter --diffs to one file, e.g. src/model.py")
    args = ap.parse_args()

    if not any(vars(args).values()):
        ap.print_help()
        sys.exit(0)

    print("Loading Claude Code file-history …")
    claude = load_claude_history()
    print("Loading VS Code local history …")
    vscode = load_vscode_history()
    print("Merging and deduplicating …")
    merged = merge_histories(claude, vscode)

    if args.list:          action_list(merged, claude, vscode)
    if args.scrub_check:   action_scrub_check(merged)
    if args.diffs:         action_diffs(merged, args.file)
    if args.report:        action_report(merged)
    if args.merge_commits: action_merge_commits(merged)


if __name__ == "__main__":
    main()
