#!/usr/bin/env python3
"""Documentation link checker.

Checks every tracked Markdown file of the repository (README.md, docs/**.md, crate/tool READMEs):

1. relative Markdown links ``[text](path)`` / ``[text](path#anchor)`` and reference definitions ``[id]: path``
   must point to an existing file or directory (http(s)/mailto links and pure ``#anchor`` links are skipped);
2. in LIVE documents (everything outside ``docs/archive/``), back-quoted repository paths such as
   ``crates/amdgpu-tile/src/fmha.rs:83`` or ``docs/host-gen.md`` must exist, and a ``:line`` / ``:a-b`` suffix must not
   exceed the file's line count. Archived documents are historical records: their path mentions are not checked.
3. no encoding damage: neither ``???`` nor U+FFFD may appear outside code blocks, and no file may start with a UTF-8 BOM.

Usage: ``python tools/check_doc_links.py`` (exit status 1 on any finding). No third-party dependencies.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG = re.compile(r"!\[[^\]\n]*\]\(([^)\s]+)\)")
REFDEF = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.M)
TICK = re.compile(r"`((?:crates|docs|tools|tests)/[A-Za-z0-9_./\-]+?)(?::(\d+)(?:[-–](\d+))?)?`")


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def strip_code_blocks(text):
    # keep line numbers stable: replace fenced code blocks by empty lines
    lines = text.split("\n")
    out, fence = [], False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else ln)
    return "\n".join(out)


def resolve(p):
    """A repository path, or a crate-relative shorthand (`tests/x.rs` = `crates/<crate>/tests/x.rs`, unique match)."""
    full = os.path.join(ROOT, p)
    if os.path.exists(full):
        return full
    if p.startswith(("tests/", "src/")):
        hits = [os.path.join(ROOT, "crates", c, p) for c in sorted(os.listdir(os.path.join(ROOT, "crates")))]
        hits = [h for h in hits if os.path.exists(h)]
        if len(hits) == 1:
            return hits[0]
    return None


def line_count(path):
    with open(path, "rb") as f:
        return f.read().count(b"\n") + 1


def main():
    findings = []
    files = tracked_markdown()
    for rel in files:
        path = os.path.join(ROOT, rel)
        with open(path, "rb") as f:
            raw = f.read()
        if raw.startswith(b"\xef\xbb\xbf"):
            findings.append(f"{rel}: UTF-8 BOM")
        text = raw.decode("utf-8", errors="replace")
        body = strip_code_blocks(text)
        base = os.path.dirname(path)
        live = not rel.startswith("docs/archive/")
        for lineno, ln in enumerate(body.split("\n"), 1):
            targets = [m.group(1) for m in LINK.finditer(ln)] + [m.group(1) for m in IMG.finditer(ln)]
            for t in targets:
                if re.match(r"^(https?:|mailto:|#)", t):
                    continue
                p = t.split("#", 1)[0]
                if not p:
                    continue
                full = os.path.normpath(os.path.join(base, p.replace("%20", " ")))
                if not os.path.exists(full):
                    findings.append(f"{rel}:{lineno}: broken link -> {t}")
            if "???" in ln or "\ufffd" in ln:
                findings.append(f"{rel}:{lineno}: '???' or U+FFFD (encoding damage?)")
            if live:
                for m in TICK.finditer(ln):
                    p, a, b = m.group(1).rstrip("."), m.group(2), m.group(3)
                    if any(ch in p for ch in "*{}<>"):
                        continue
                    full = resolve(p)
                    if full is None:
                        findings.append(f"{rel}:{lineno}: missing path `{m.group(0)[1:-1]}`")
                        continue
                    if a and os.path.isfile(full):
                        n = line_count(full)
                        hi = int(b or a)
                        if hi > n:
                            findings.append(f"{rel}:{lineno}: `{m.group(0)[1:-1]}` beyond end of file ({n} lines)")
        for m in REFDEF.finditer(body):
            t = m.group(1)
            if re.match(r"^(https?:|mailto:|#)", t):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(base, t.split("#", 1)[0]))):
                findings.append(f"{rel}: broken reference definition -> {t}")
    for f in findings:
        print(f)
    print(f"checked {len(files)} markdown files: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
