"""Mirror raw VinDr-Mammo (DICOM + CSV) from PhysioNet into a directory (e.g. Google Drive) with wget.

One connection, resumable: re-running skips finished files and continues partial ones, so PhysioNet does
not throttle the IP as it does with many parallel connections. Result layout:
  <dest>/breast-level_annotations.csv, <dest>/images/<study_id>/<image_id>.dicom
which is what `scripts/vindr_fetch.py download --source <dest>` reads.

Resume: -N --no-if-modified-since makes wget compare size + timestamp from a HEAD request (plain -N
sends If-Modified-Since, and an interrupted partial file, being newer than the server copy, would get
"304 not modified" and stay truncated forever); -c then continues the partial file.
--verify additionally checks every downloaded file against SHA256SUMS.txt from the mirror root and deletes
mismatches, so the next run fetches them again.

Credentials: PHYSIONET_USER / PHYSIONET_PASS (an account that signed the VinDr-Mammo DUA). They are passed
to wget through a temporary wgetrc (mode 600), never on the command line. Stdlib only; needs wget.
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://physionet.org/files/vindr-mammo/1.0.0/"
CSV_FILES = ["breast-level_annotations.csv", "finding_annotations.csv", "metadata.csv"]
SUMS_FILE = "SHA256SUMS.txt"
# wget exit codes that retrying cannot fix: generic, parse error, file I/O (e.g. disk/Drive full), auth
FATAL = {1: "generic wget error", 2: "wget command-line/wgetrc parse error",
         3: "file I/O error (is the destination full or not writable?)"}
AUTH_FAILED = 6
SERVER_ERROR = 8  # wget: "server issued an error response" (any single 404/5xx in the whole crawl)
PRINT_EVERY = 200  # files; wget -nv prints one line per file, 25k lines would flood the notebook
LISTING_EVERY = 500  # directory listings; heartbeat during the crawl phase, which saves no real files for long
SHOW_URLS = 50  # failing URLs printed at most
# -nv lines that are pure noise for a mirror of ~20k files + ~5k directory listings
QUIET = ("not modified on server", "no newer than local file", "not retrieving", "since it should be rejected",
         "already fully retrieved", "Removing ", "Last-modified header missing")
ERROR_RE = re.compile(r"ERROR (\d{3})")
URL_RE = re.compile(r"(https?://\S+?):?(?:\s|$)")
SAVED_RE = re.compile(r'URL: ?(\S+) .*-> "([^"]*)"')


def wget_cmd(dest: Path, only_csv: bool, verify: bool = False) -> list[str]:
    common = ["wget", "-N", "--no-if-modified-since", "-c", "-nv", "--tries=5", "--waitretry=30", "--timeout=60"]
    if only_csv:
        names = CSV_FILES + ([SUMS_FILE] if verify else [])
        return common + ["-P", str(dest)] + [BASE_URL + name for name in names]
    # URL path is /files/vindr-mammo/1.0.0/ -> -nH drops the host, --cut-dirs=3 the 3 path parts,
    # so files land directly in dest: dest/images/<study>/<image>.dicom
    return common + ["-r", "-l", "inf", "-np", "-nH", "--cut-dirs=3", "-e", "robots=off",
                     "--reject", "index.html*", "-P", str(dest), BASE_URL]


@dataclass
class Attempt:
    code: int = 0
    files: int = 0  # real files saved (or resumed) in this attempt; directory listings not counted
    listings: int = 0  # index.html pages wget fetched to crawl (deleted again by --reject)
    errors: list[str] = field(default_factory=list)  # URLs that got an ERROR response
    forbidden: bool = False


def run_wget(cmd: list[str], env: dict) -> Attempt:
    """Run wget, print errors and every PRINT_EVERY-th saved file, count real files vs listings."""
    r = Attempt()
    last_url = None
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace")
    for line in p.stdout:
        line = line.rstrip()
        if not line or any(q in line for q in QUIET):
            continue
        saved = SAVED_RE.search(line)  # -nv success line: '... URL:<url> [size] -> "<path>" [1]'
        if saved:
            url, path = saved.groups()
            if "index.html" in Path(path).name or url.endswith("/"):
                r.listings += 1
                if r.listings % LISTING_EVERY == 0:
                    print(f"  crawling directory listings: {r.listings} ...", flush=True)
            else:
                r.files += 1
                if r.files % PRINT_EVERY == 0:
                    print(f"  [{r.files} files fetched this run] {line}", flush=True)
            continue
        err = ERROR_RE.search(line)
        if err:  # -nv prints '<url>:' on one line and '<time> ERROR 404: Not Found.' on the next
            m = URL_RE.search(line)
            url = m.group(1) if m else last_url
            if url and url not in r.errors:
                r.errors.append(url)
            r.forbidden |= err.group(1) == "403"
            print(line, flush=True)
            continue
        m = URL_RE.fullmatch(line.strip())
        if m and line.strip().endswith(":"):  # URL header of the next ERROR line: print together with it
            last_url = m.group(1)
            if len(r.errors) < SHOW_URLS:
                print(line, flush=True)
            continue
        print(line, flush=True)
    r.code = p.wait()
    print(f"wget exit code {r.code}: {r.files} files fetched this run, {r.listings} directory listings, "
          f"{len(r.errors)} URLs with errors", flush=True)
    return r


def print_errors(errors: list[str]) -> None:
    print(f"URLs with server errors ({len(errors)}):", flush=True)
    for url in errors[:SHOW_URLS]:
        print(f"  {url}", flush=True)
    if len(errors) > SHOW_URLS:
        print(f"  ... and {len(errors) - SHOW_URLS} more", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path) -> int:
    """Check downloaded files against SHA256SUMS.txt, delete mismatches. Returns the number deleted."""
    sums = dest / SUMS_FILE
    if not sums.is_file():
        raise SystemExit(f"--verify: {sums} not found (it is fetched with the mirror; re-run without --only-csv)")
    root = dest.resolve()
    entries = []
    for line in sums.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            entries.append((parts[0].lower(), parts[1].lstrip("*").removeprefix("./")))
    ok = bad = absent = 0
    print(f"verifying sha256 of up to {len(entries)} files in {dest} ...", flush=True)
    for i, (digest, rel) in enumerate(entries, 1):
        f = dest / rel
        if not f.resolve().is_relative_to(root):
            print(f"  skipping suspicious path in {SUMS_FILE}: {rel!r}", flush=True)
            continue
        if not f.is_file():
            absent += 1
        elif sha256(f) == digest:
            ok += 1
        else:
            f.unlink()
            bad += 1
            if bad <= SHOW_URLS:
                print(f"  sha256 mismatch, deleted: {rel}", flush=True)
        if i % 1000 == 0:
            print(f"  checked {i}/{len(entries)}", flush=True)
    print(f"verify: {ok} ok, {bad} mismatched (deleted), {absent} not downloaded", flush=True)
    return bad


def report(dest: Path, only_csv: bool) -> None:
    ann = dest / "breast-level_annotations.csv"
    expected = None
    if ann.exists():
        with ann.open(newline="") as f:
            expected = sum(1 for _ in csv.reader(f)) - 1
    dicoms = list((dest / "images").glob("*/*.dicom"))
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"CSV present: {[n for n in CSV_FILES if (dest / n).exists()]}")
    print(f"DICOM files: {len(dicoms)} / {expected if expected is not None else '?'} expected"
          f"{' (CSV-only run)' if only_csv else ''}")
    print(f"total size in {dest}: {size / 1e9:.1f} GB")


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="/content/drive/MyDrive/vindr-mammo", help="target directory")
    ap.add_argument("--only-csv", action="store_true", help="download only the CSV annotations (quick test)")
    ap.add_argument("--attempts", type=positive_int, default=20, help="wget runs at most (>= 1)")
    ap.add_argument("--sleep", type=int, default=60, help="seconds between restarts (grows linearly)")
    ap.add_argument("--verify", action="store_true",
                    help=f"after download check sha256 against {SUMS_FILE}, delete mismatches (slow: reads all data)")
    args = ap.parse_args()

    if not shutil.which("wget"):
        raise SystemExit("wget not found (on Colab it is preinstalled; elsewhere: apt/brew install wget)")
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        raise SystemExit("set PHYSIONET_USER and PHYSIONET_PASS (PhysioNet username, not email)")
    if "\n" in user or "\n" in password or "\r" in user or "\r" in password:
        raise SystemExit("PHYSIONET_USER / PHYSIONET_PASS must not contain newlines")
    if user != user.strip() or password != password.strip():
        raise SystemExit("PHYSIONET_USER / PHYSIONET_PASS start or end with whitespace: wgetrc would strip it "
                         "and the login would fail; remove the spaces (or change the password)")

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    fd, rc_path = tempfile.mkstemp(prefix="wgetrc-")  # created with mode 600
    try:
        os.chmod(rc_path, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(f"user={user}\npassword={password}\n")
        env = dict(os.environ, WGETRC=rc_path)
        cmd = wget_cmd(dest, args.only_csv, args.verify)
        print("$", " ".join(cmd), flush=True)
        for attempt in range(1, args.attempts + 1):
            r = run_wget(cmd, env)
            if r.code == 0:
                break
            if r.code == AUTH_FAILED:
                raise SystemExit("wget: authentication failed: wrong PhysioNet username/password; "
                                 "use username, not email")
            if r.forbidden and not r.files:
                raise SystemExit("PhysioNet 403: no access; sign the VinDr-Mammo data use agreement "
                                 "(credentialed access) for this account")
            if r.code in FATAL:
                raise SystemExit(f"wget failed: {FATAL[r.code]}")
            if r.code == SERVER_ERROR and not r.files:
                print_errors(r.errors)
                report(dest, args.only_csv)
                raise SystemExit(f"wget: server error responses for {len(r.errors)} URL(s) (listed above) and no "
                                 "new file saved in this attempt, so restarting the crawl will not help. "
                                 "Everything else is already here; re-run later if PhysioNet had an outage")
            if attempt == args.attempts:
                if r.errors:
                    print_errors(r.errors)
                report(dest, args.only_csv)
                raise SystemExit(f"wget still failing after {attempt} attempts (code {r.code}); re-run to resume")
            wait = min(args.sleep * attempt, 600)
            print(f"attempt {attempt}/{args.attempts} failed (code {r.code}, {r.files} files saved), "
                  f"retrying in {wait} s", flush=True)
            time.sleep(wait)
    finally:
        os.unlink(rc_path)
    report(dest, args.only_csv)
    if args.verify and (bad := verify(dest)):
        raise SystemExit(f"{bad} file(s) failed the sha256 check and were deleted; re-run to download them again")


if __name__ == "__main__":
    sys.exit(main())
