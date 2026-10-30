#!/usr/bin/env python3
"""Integrity checks for this repo (Python 3 stdlib only; no network by default)."""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZH = ROOT / "zh"
EN = ROOT / "en"
README = ROOT / "README.md"
README_EN = ROOT / "README.en.md"

QHEAD_RE = re.compile(r"^## ([A-Z]+)-(\d{3})\. ")
TOC_LINK_RE = re.compile(r"\]\((zh/[^)\s]+\.md|en/[^)\s]+\.md)\)")
URL_RE = re.compile(r"https?://[^\s<>]+")
URL_TRAIL_PUNCT = ").,;>"

LAYER_ZH = (
    "**答：**",
    "**深入展开：**",
    "**后续追问：**",
    "**扩展问题：**",
    "**推理回答：**",
)
LAYER_EN = (
    "**Answer:**",
    "**Deep dive:**",
    "**Follow-ups:**",
    "**Extended questions:**",
    "**Reasoning:**",
)
LIST_LAYERS = frozenset(("**扩展问题：**", "**Extended questions:**"))

# sha256 of normalised restricted terms
_RESTRICTED_CHAR: dict[int, frozenset[str]] = {
    13: frozenset(
        {
            "3e030b3b2b25d4f732799d0e25b055422fbbf544b333cf069e43eb007f34e007",
        }
    ),
    11: frozenset(
        {
            "37372f738428d47de0dbdc6f96b2d587dc3eec3890bea30eedc15752acdc2651",
        }
    ),
    6: frozenset(
        {
            "283d2da40049e3065345a7af8cbc9effd886650dee5fa6b137784ab3cdc99f7f",
            "97f232a13e785cb673d22d9c8f3fe1aa3f96eeff30df146873faa633dfbda75c",
        }
    ),
    5: frozenset(
        {
            "4a8bc62ad600d94375d8d732593f25721759ba41e672938d029fdbb302bff96c",
            "8c7ab71277170b4b3151fbd3cc44fb46150fb57e89f21d591720420bce2a300a",
            "8e131105bb6c2f67ce2e640cc9cbc8f11a3061a1db66d41036b8882590429e4e",
        }
    ),
    4: frozenset(
        {
            "771d26797da43f05ce800e02d8ae9f1d62535553ec8f23cb8fb0e9ff929d34b2",
            "95beb193a957daea8324eccd0d5b15ccb2ea21e9065083710d4e5d9e5a126b27",
            "f2a722d81a2fd8ca7529c11dbee909e48441ad1ec52347ca9867f00d2d58469c",
        }
    ),
    2: frozenset(
        {
            "a9ab292ea9feecdcdfd3d6dfae632d35e27923892feb884ddb36502a9e89a67d",
        }
    ),
}
_RESTRICTED_TOKEN: dict[int, frozenset[str]] = {
    20: frozenset(
        {
            "02851d8716aa2858326e26dcfeddec7724c5fbdf0a8fe103e8a9ce86c41b8a75",
            "063af572f558797d0abe75da8568e66ed5b629aa0eafbe770976a53c0396dd63",
            "fb29c4a96301c929e4ccfe880edf6ce62e97c1059af3bd6a4d924826fb49249e",
        }
    ),
    18: frozenset(
        {
            "e2ada46b25e4977795b6ea51ffa574d5dfa39d5adffc511fa172342090b3f87d",
        }
    ),
    15: frozenset(
        {
            "25105c0dc12641e003a17fa981285f9cf736cf09d8765dd3f51de80cdf5b5538",
            "810d8a17a396a5b8428585ab56555d86f9b479f4c88abdf6a603aa001a52b391",
            "c32f5c690dda4a085e5c453ed64bad09017f19ad344cc2f3f789e063474559a7",
            "f17cd1e64626a63036cb93587353d758bdb58d44f18c74bd718088286d368e44",
        }
    ),
    14: frozenset(
        {
            "8df902d456ebe95bce641935f7cad2caf25ca0cb56003071cbb4910ba82b4383",
        }
    ),
    13: frozenset(
        {
            "1e8563cee8b9a459ba3b74297ba131e515d4f07989291c961a4cafdb90bbb8de",
            "379491479cb46e2f3921a6f899104493d0fd6a17f9645266d6f37c78c0095604",
        }
    ),
    12: frozenset(
        {
            "7105bfdc0865838d98872429344c83b895f17e2b4c3a1a6313e82fe65cde267a",
            "edce0cbb78144107726a7796c7b1e5cb76176daf8fcc94f5550c38ff65774c7b",
            "f90eceab6fa3a85b8796929c1fa81a41054d28ece2a0c400a54be9a08a927db4",
        }
    ),
    8: frozenset(
        {
            "1581e27de87bffae0bd4d745cd7964e68528d7a83e2e4c259a782d275df6f558",
            "95a0293331a7e4c735c0694e75bb6c1ada9de8577148d7a82cc3d1a9f7887d90",
        }
    ),
    7: frozenset(
        {
            "882c45820c72fc317733a6fcdfd7fe258ebee7a4e9fb3cfa84232e58235b35aa",
            "af70b78aff548460e6ffa65d3ae8403b35bc99e5c6218fa4cd50eadfc056890d",
        }
    ),
}
_RESTRICTED_TOKEN_WINDOW = 4

LICENSE_PATH = ROOT / "LICENSE"
REQUIRED_HOLDER = "cluster1900"
# YEAR is exactly 4 digits; only that field may differ from the canonical MIT text.
MIT_LICENSE_CANONICAL = """\
MIT License

Copyright (c) {year} cluster1900

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
COPYRIGHT_LINE_RE = re.compile(r"^Copyright \(c\) (\d{4}) (.+)$")
README_LICENSE_HEADING = "## 许可证 / License"

EXTRA_LICENSE_PATTERNS = (
    "LICENSE-*",
    "LICENSE.*",
    "LICENCE*",
    "NOTICE*",
    "COPYING*",
)

failed: list[str] = []


def report(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not passed:
        failed.append(name)


def parse_questions(path: Path) -> list[tuple[str, int, str]]:
    """Return (qid, heading_lineno, block_text) in file order."""
    lines = path.read_text(encoding="utf-8").splitlines()
    heads: list[tuple[str, int]] = []
    for i, line in enumerate(lines, 1):
        m = QHEAD_RE.match(line)
        if m:
            heads.append((f"{m.group(1)}-{m.group(2)}", i))
    blocks: list[tuple[str, int, str]] = []
    for idx, (qid, lineno) in enumerate(heads):
        start = lineno - 1
        end = heads[idx + 1][1] - 1 if idx + 1 < len(heads) else len(lines)
        blocks.append((qid, lineno, "\n".join(lines[start:end])))
    return blocks


def iter_md(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.md"))


def read_text_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def parse_readme_contents_rows() -> list[tuple[int, str, str, int, str | None]]:
    """Contents table rows: (lineno, zh_rel, en_rel, count, code).

    Accepts the 4-column table and the 5-column table (count = last cell).
    ``code`` is the backticked chapter code when present.
    """
    if not README.is_file():
        return []
    rows: list[tuple[int, str, str, int, str | None]] = []
    for lineno, line in enumerate(README.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[-1].isdigit():
            continue
        zh_m = re.search(r"\]\((zh/[^)\s]+\.md)\)", line)
        en_m = re.search(r"\]\((en/[^)\s]+\.md)\)", line)
        if not (zh_m and en_m):
            continue
        code = None
        for cell in cells:
            cm = re.fullmatch(r"`([A-Z]+)`", cell)
            if cm:
                code = cm.group(1)
                break
        rows.append((lineno, zh_m.group(1), en_m.group(1), int(cells[-1]), code))
    return rows


def check_ids() -> None:
    problems: list[str] = []
    file_qids: dict[str, list[str]] = {}
    file_prefix: dict[str, str] = {}
    prefix_slugs: dict[str, set[str]] = defaultdict(set)
    id_slugs: dict[str, set[str]] = defaultdict(set)
    id_lang_files: dict[tuple[str, str], list[str]] = defaultdict(list)
    total = 0

    for folder in (ZH, EN):
        lang = folder.name
        for path in iter_md(folder):
            rel = str(path.relative_to(ROOT))
            lines = path.read_text(encoding="utf-8").splitlines()
            qids: list[str] = []
            prefixes: list[str] = []
            for i, line in enumerate(lines, 1):
                if not line.startswith("## "):
                    continue
                m = QHEAD_RE.match(line)
                if not m:
                    problems.append(f"{rel}:{i} invalid heading {line!r}")
                    continue
                prefix, seq = m.group(1), m.group(2)
                qid = f"{prefix}-{seq}"
                qids.append(qid)
                prefixes.append(prefix)
                id_slugs[qid].add(path.name)
                id_lang_files[(lang, qid)].append(rel)

            file_qids[rel] = qids
            total += len(qids)
            uniq = list(dict.fromkeys(prefixes))
            if not qids:
                problems.append(f"{rel} has no question headings")
                continue
            if len(uniq) > 1:
                problems.append(f"{rel} mixed prefixes {uniq} ids={qids}")
                continue
            prefix = uniq[0]
            file_prefix[rel] = prefix
            prefix_slugs[prefix].add(path.name)
            expected = [f"{prefix}-{n:03d}" for n in range(1, len(qids) + 1)]
            if qids != expected:
                problems.append(f"{rel} sequence {qids} != {expected}")

    for prefix, slugs in sorted(prefix_slugs.items()):
        if len(slugs) > 1:
            problems.append(
                f"prefix {prefix} used in multiple files: {', '.join(sorted(slugs))}"
            )

    seen_global: set[str] = set()
    for qid, slugs in sorted(id_slugs.items()):
        if len(slugs) > 1:
            problems.append(f"duplicate id {qid} in {', '.join(sorted(slugs))}")
            seen_global.add(qid)
    for (lang, qid), rels in sorted(id_lang_files.items()):
        if len(rels) > 1 and qid not in seen_global:
            problems.append(f"duplicate id {qid} in {lang}: {', '.join(rels)}")

    table_code: dict[str, str | None] = {}
    for _lineno, zh_rel, en_rel, _count, code in parse_readme_contents_rows():
        table_code[zh_rel] = code
        table_code[en_rel] = code
    for rel, prefix in sorted(file_prefix.items()):
        if rel not in table_code:
            continue
        expected_code = table_code[rel]
        if expected_code != prefix:
            problems.append(
                f"{rel} prefix {prefix} != README code {expected_code}"
            )

    for item in problems:
        print(f"    {item}")
    report(
        "ids",
        not problems,
        (
            f"{len(problems)} problem(s); offending ids listed above"
            if problems
            else f"count={total} prefixes unique and match README; sequences 001..n"
        ),
    )


def check_en_alignment() -> None:
    zh_files = {p.name: p for p in iter_md(ZH)}
    en_files = {p.name: p for p in iter_md(EN)}
    zh_only = sorted(set(zh_files) - set(en_files))
    en_only = sorted(set(en_files) - set(zh_files))
    print(f"  zh-only chapters: {', '.join(zh_only) if zh_only else 'none'}")
    print(f"  en-only chapters: {', '.join(en_only) if en_only else 'none'}")
    problems: list[str] = []
    if zh_only:
        problems.append(f"zh-only: {', '.join(zh_only)}")
    if en_only:
        problems.append(f"en-only: {', '.join(en_only)}")
    for name in sorted(set(zh_files) & set(en_files)):
        en_ids = [qid for qid, _, _ in parse_questions(en_files[name])]
        zh_ids = [qid for qid, _, _ in parse_questions(zh_files[name])]
        if en_ids != zh_ids:
            problems.append(f"{name}: question numbers differ (en {en_ids} vs zh {zh_ids})")
    report(
        "en-alignment",
        not problems,
        "; ".join(problems) if problems else "matching files have identical question-number order",
    )


def _layer_issues(block: str, markers: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    counts = {m: block.count(m) for m in markers}
    missing = [m for m in markers if counts[m] == 0]
    dups = [m for m in markers if counts[m] > 1]
    if missing:
        issues.append(f"missing {missing}")
    if dups:
        issues.append(f"duplicate {dups}")

    present_pos = [(block.find(m), m) for m in markers if counts[m] > 0]
    found_order = [m for _, m in sorted(present_pos)]
    expected_order = [m for m in markers if counts[m] > 0]
    if found_order != expected_order:
        issues.append(f"out of order {found_order}")

    lines = block.splitlines()
    line_of: dict[str, int] = {}
    for i, line in enumerate(lines):
        for m in markers:
            if m in line and m not in line_of:
                line_of[m] = i

    for m in markers:
        if counts[m] == 0:
            continue
        idx = line_of.get(m)
        if idx is None:
            issues.append(f"empty {m}")
            continue
        after = lines[idx].split(m, 1)[1].strip()
        if after:
            continue
        if m in LIST_LAYERS:
            next_idx = len(lines)
            for other in markers:
                if other == m:
                    continue
                j = line_of.get(other)
                if j is not None and j > idx:
                    next_idx = min(next_idx, j)
            if any(ln.lstrip().startswith("- ") for ln in lines[idx + 1 : next_idx]):
                continue
        issues.append(f"empty {m}")
    return issues


def check_layers() -> None:
    bad: list[str] = []
    for folder, markers in ((ZH, LAYER_ZH), (EN, LAYER_EN)):
        for path in iter_md(folder):
            rel = path.relative_to(ROOT)
            for qid, lineno, block in parse_questions(path):
                issues = _layer_issues(block, markers)
                if issues:
                    bad.append(f"{rel}:{lineno} #{qid} {'; '.join(issues)}")
    for item in bad:
        print(f"    {item}")
    report(
        "layers",
        not bad,
        f"{len(bad)} question block(s) with layer errors" if bad else "all question blocks have required layer markers",
    )


def scan_text_files() -> list[Path]:
    paths: list[Path] = []
    if README.is_file():
        paths.append(README)
    if README_EN.is_file():
        paths.append(README_EN)
    paths.extend(iter_md(ZH))
    paths.extend(iter_md(EN))
    paths.extend(sorted(p for p in ROOT.glob("LICENSE*") if p.is_file()))
    for folder_name in ("scripts", ".github"):
        folder = ROOT / folder_name
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                paths.append(p)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _iter_repo_files() -> list[tuple[str, Path]]:
    rels: list[str] | None = None
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            rels = [os.fsdecode(p) for p in proc.stdout.split(b"\0") if p]
    except OSError:
        rels = None
    if rels is None:
        rels = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fn in filenames:
                full = Path(dirpath) / fn
                rels.append(str(full.relative_to(ROOT)))
    return [(rel.replace("\\", "/"), ROOT / rel) for rel in rels]


def _read_repo_text(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            head = f.read(8192)
            if b"\x00" in head:
                return None
            data = head + f.read()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def check_restricted_terms() -> None:
    hits: set[tuple[str, int]] = set()
    char_lengths = sorted(_RESTRICTED_CHAR)
    min_char_len = char_lengths[0] if char_lengths else 0
    token_lengths = _RESTRICTED_TOKEN
    max_token_window = _RESTRICTED_TOKEN_WINDOW

    for rel, path in _iter_repo_files():
        text = _read_repo_text(path)
        if text is None:
            continue

        char_buf: list[str] = []
        char_line: list[int] = []
        tokens: list[str] = []
        token_line: list[int] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            folded = unicodedata.normalize("NFKC", line).casefold()
            for ch in folded:
                if ch.isalnum():
                    char_buf.append(ch)
                    char_line.append(lineno)
            for tok in re.findall(r"[^\W_]+", folded):
                tokens.append(tok)
                token_line.append(lineno)

        norm = "".join(char_buf)
        n = len(norm)
        if n >= min_char_len:
            for L in char_lengths:
                if n < L:
                    continue
                want = _RESTRICTED_CHAR[L]
                limit = n - L + 1
                for i in range(limit):
                    if _digest(norm[i : i + L]) in want:
                        hits.add((rel, char_line[i]))

        ntok = len(tokens)
        if ntok:
            for width in range(1, max_token_window + 1):
                if ntok < width:
                    break
                for i in range(ntok - width + 1):
                    joined = "".join(tokens[i : i + width])
                    L = len(joined)
                    want = token_lengths.get(L)
                    if want and _digest(joined) in want:
                        hits.add((rel, token_line[i]))

    ordered = sorted(hits)
    for rel, lineno in ordered:
        print(f"    {rel}:{lineno}")
    report(
        "restricted-terms",
        not ordered,
        f"{len(ordered)} hit(s)" if ordered else "no restricted terms found",
    )


def check_readme_toc() -> None:
    if not README.is_file():
        report("readme-toc", False, "README.md missing")
        return
    links = TOC_LINK_RE.findall(README.read_text(encoding="utf-8"))
    missing = [rel for rel in links if not (ROOT / rel).is_file()]
    for rel in missing:
        print(f"    missing {rel}")
    report(
        "readme-toc",
        not missing,
        f"{len(links)} link(s), missing={missing or 'none'}",
    )


def check_readme_counts() -> None:
    if not README.is_file():
        report("readme-counts", False, "README.md missing")
        return
    rows = parse_readme_contents_rows()
    table_rels: set[str] = set()
    problems: list[str] = []
    for lineno, zh_rel, en_rel, count, _code in rows:
        table_rels.add(zh_rel)
        table_rels.add(en_rel)
        zh_path = ROOT / zh_rel
        en_path = ROOT / en_rel
        zh_n = len(parse_questions(zh_path)) if zh_path.is_file() else None
        en_n = len(parse_questions(en_path)) if en_path.is_file() else None
        if zh_n != count or en_n != count:
            problems.append(
                f"README.md:{lineno} count={count} {zh_rel}={zh_n if zh_n is not None else 'missing'} "
                f"{en_rel}={en_n if en_n is not None else 'missing'}"
            )

    disk = {f"zh/{p.name}" for p in iter_md(ZH)} | {f"en/{p.name}" for p in iter_md(EN)}
    for rel in sorted(disk - table_rels):
        problems.append(f"{rel} not in README contents table")

    for item in problems:
        print(f"    {item}")
    report(
        "readme-counts",
        not problems,
        f"{len(problems)} mismatch(es)" if problems else f"{len(rows)} table row(s) match zh/en question counts",
    )


def has_origin_main() -> bool:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "origin/main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def git_show(relpath: str) -> str | None:
    if not has_origin_main():
        return None
    proc = subprocess.run(
        ["git", "show", f"origin/main:{relpath}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def curl_http_code(url: str) -> str:
    curl = subprocess.run(
        [
            "curl",
            "-sL",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "-m",
            "20",
            "-A",
            "Mozilla/5.0",
            url,
        ],
        capture_output=True,
        text=True,
    )
    code = curl.stdout.strip()
    return code if code else f"curl-exit-{curl.returncode}"


def http_ok(code: str) -> bool:
    return code.isdigit() and 200 <= int(code) < 400


def extract_external_urls(text: str) -> set[str]:
    """Markdown links, autolinks, and bare http(s) URLs; trailing punct stripped."""
    urls: set[str] = set()
    for raw in URL_RE.findall(text):
        url = raw.rstrip(URL_TRAIL_PUNCT)
        if url:
            urls.add(url)
    return urls


def iter_link_sources() -> list[Path]:
    paths: list[Path] = []
    if README.is_file():
        paths.append(README)
    if README_EN.is_file():
        paths.append(README_EN)
    paths.extend(iter_md(ZH))
    paths.extend(iter_md(EN))
    return paths


def check_external_links() -> None:
    current: dict[str, set[str]] = {}
    for path in iter_link_sources():
        rel = str(path.relative_to(ROOT))
        current[rel] = extract_external_urls(path.read_text(encoding="utf-8"))

    origin_ok = has_origin_main()
    if not origin_ok:
        print("  origin/main missing; treating all external links as new")

    new_links: list[tuple[str, str]] = []
    for rel, urls in current.items():
        old_text = None if not origin_ok else git_show(rel)
        old = set() if old_text is None else extract_external_urls(old_text)
        for url in sorted(urls - old):
            new_links.append((rel, url))

    print(f"  new external links ({len(new_links)}):")
    for rel, url in new_links:
        print(f"    {rel}\t{url}")

    if os.environ.get("CHECK_LINKS") == "1":
        bad: list[tuple[str, str, str]] = []
        for rel, url in new_links:
            code = curl_http_code(url)
            if not http_ok(code):
                code = curl_http_code(url)
            if not http_ok(code):
                bad.append((rel, url, code or f"unknown"))
        for rel, url, code in bad:
            print(f"    HTTP {code} {rel} {url}")
        report(
            "external-links",
            not bad,
            f"{len(bad)} non-2xx/3xx" if bad else f"all {len(new_links)} new link(s) returned 2xx/3xx",
        )
    else:
        report(
            "external-links",
            True,
            f"listed {len(new_links)} new external link(s); CHECK_LINKS not set, skip curl",
        )


def _normalize_license_text(text: str) -> str:
    return "\n".join(ln.rstrip() for ln in text.splitlines())


def check_license() -> None:
    problems: list[str] = []
    holder = None

    if not LICENSE_PATH.is_file():
        problems.append("LICENSE missing at repo root")
    else:
        actual = _normalize_license_text(LICENSE_PATH.read_text(encoding="utf-8"))
        year = None
        for ln in actual.split("\n"):
            m = COPYRIGHT_LINE_RE.match(ln)
            if m:
                year = m.group(1)
                holder = m.group(2)
                break
        if holder is None:
            problems.append("missing Copyright (c) YEAR holder line")
        elif holder != REQUIRED_HOLDER:
            problems.append(f"copyright holder {holder!r}, expected {REQUIRED_HOLDER!r}")
        if year is not None:
            expected = _normalize_license_text(
                MIT_LICENSE_CANONICAL.replace("{year}", year)
            )
            if actual != expected:
                problems.append(
                    "LICENSE text does not match standard MIT (year is the only variable)"
                )
        else:
            problems.append("LICENSE text does not match standard MIT (year is the only variable)")

    if not README.is_file():
        problems.append("README.md missing")
        report("license", False, "; ".join(problems))
        return

    readme = README.read_text(encoding="utf-8")
    lines = readme.splitlines()
    heading_idx = next((i for i, ln in enumerate(lines) if ln.strip() == README_LICENSE_HEADING), None)
    if heading_idx is None:
        problems.append("README missing heading " + README_LICENSE_HEADING)
        section_lines: list[str] = []
    else:
        section_lines = []
        for ln in lines[heading_idx + 1 :]:
            if ln.startswith("## "):
                break
            section_lines.append(ln)

    zh_ok = any("MIT 许可证" in ln and "(LICENSE)" in ln for ln in section_lines)
    en_ok = any("MIT License" in ln and "(LICENSE)" in ln for ln in section_lines)
    if not zh_ok:
        problems.append("README license section missing Chinese MIT sentence linking to (LICENSE)")
    if not en_ok:
        problems.append("README license section missing English MIT sentence linking to (LICENSE)")

    if holder and holder not in readme:
        problems.append(f"copyright holder {holder!r} does not appear in README")

    report(
        "license",
        not problems,
        "; ".join(problems) if problems else "MIT LICENSE present; README license section matches holder",
    )


def _is_extra_license_basename(name: str) -> bool:
    upper = name.upper()
    return any(fnmatch.fnmatch(upper, pat) for pat in EXTRA_LICENSE_PATTERNS)


def check_extra_license_files() -> None:
    bad: list[str] = []
    license_resolved = LICENSE_PATH.resolve() if LICENSE_PATH.is_file() else None
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            path = Path(dirpath) / fn
            if license_resolved is not None and path.resolve() == license_resolved:
                continue
            if _is_extra_license_basename(fn):
                bad.append(str(path.relative_to(ROOT)))
    bad.sort()
    for rel in bad:
        print(f"    {rel}")
    report(
        "extra-license-files",
        not bad,
        f"{len(bad)} extra file(s): {', '.join(bad)}" if bad else "no extra license/notice/copying files",
    )


def main() -> None:
    check_ids()
    check_en_alignment()
    check_layers()
    check_restricted_terms()
    check_readme_toc()
    check_readme_counts()
    check_license()
    check_extra_license_files()
    check_external_links()
    print()
    if failed:
        print(f"SUMMARY: FAIL ({len(failed)} check(s): {', '.join(failed)})")
        sys.exit(1)
    print("SUMMARY: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
