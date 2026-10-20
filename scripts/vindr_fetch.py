"""VinDr-Mammo: download from PhysioNet once, keep converted PNGs in a private Hugging Face dataset.

  download  Make data/vindr complete: shards already in the private HF repo are pulled from there,
            the rest is downloaded from PhysioNet (DICOM -> breast-cropped PNG, STORE_HEIGHT px), packed
            into tar shards and uploaded. PhysioNet login (PHYSIONET_USER / PHYSIONET_PASS, an account
            that signed the VinDr-Mammo DUA) is only needed while shards are missing. Resumable.
            With --source DIR the raw files are read from a local copy instead (e.g. Google Drive filled
            by scripts/vindr_to_drive.py; same layout as the PhysioNet URL), no PhysioNet login needed.
  restore   Pull the shards from the private HF repo into data/vindr (fast; no PhysioNet needed).
  status    Cheap check (no PhysioNet, no shard downloads): how many shards are neither here nor in HF.
            Last line: 'VINDR_STATUS total_shards=N missing_shards=M hf=ok|off|unavailable' ('?' if unknown).
            Exit code 0: nothing missing, 10: shards missing (download needed), other: could not check.

The PhysioNet data use agreement forbids sharing the data: the HF repo is always created private.
"""
import argparse
import io
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mammo.config import STORE_HEIGHT, VINDR  # noqa: E402
from mammo.preprocess import crop_breast, dicom_to_gray  # noqa: E402

BASE_URL = "https://physionet.org/files/vindr-mammo/1.0.0/"
CSV_FILES = ["breast-level_annotations.csv", "finding_annotations.csv", "metadata.csv"]
PNG_DIR = VINDR / "png"
SHARD_DIR = VINDR / "shards"


def hf_repo(api, name: str) -> str:
    repo = name if "/" in name else f"{api.whoami()['name']}/{name}"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise SystemExit(f"{repo} is public: VinDr-Mammo must not be shared (PhysioNet DUA). Make it private.")
    return repo


def session() -> requests.Session:
    user, password = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASS")
    if not user or not password:
        raise SystemExit("set PHYSIONET_USER and PHYSIONET_PASS")
    s = requests.Session()
    s.auth = (user, password)
    s.headers["User-Agent"] = "Wget/1.21.4"  # PhysioNet answers 403 to the default python-requests UA
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=64, max_retries=3)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch(sess, url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=120)
            if r.status_code == 401:
                raise SystemExit(f"PhysioNet 401 for {url}: wrong PHYSIONET_USER/PHYSIONET_PASS "
                                 "(use the PhysioNet username, not email)")
            if r.status_code == 403:
                raise SystemExit(f"PhysioNet 403 for {url}: access denied: credentialed access / DUA "
                                 "for this dataset not signed for this account")
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} {url}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def make_reader(args):
    """Return read(rel_path) -> bytes: from the local --source copy, or over HTTP from PhysioNet.

    Call it from the main thread: in HTTP mode the PhysioNet session is created right here (missing
    credentials stop the run before any worker thread starts) and then shared by the workers.
    """
    if args.source:
        root = args.source
        return lambda rel: (root / rel).read_bytes()  # missing file -> FileNotFoundError -> 'failed'
    sess = session()
    return lambda rel: fetch(sess, args.base_url + rel)


def shard_name(k: int) -> tuple[str, str]:
    return f"shards/shard_{k:03d}.tar", f"shards/transforms_{k:03d}.csv"


def shard_state(k: int, shard: pd.DataFrame, remote: set) -> str:
    """'local' (done here), 'remote' (in the HF repo) or 'missing' (must be converted from DICOM)."""
    if (SHARD_DIR / f"done_{k:03d}").exists() and all((PNG_DIR / f"{i}.png").exists() for i in shard.image_id):
        return "local"
    tar_name, tf_name = shard_name(k)
    return "remote" if tar_name in remote and tf_name in remote else "missing"


def split_shards(images: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [images.iloc[i:i + size] for i in range(0, len(images), size)]


def convert(read, study: str, image: str) -> tuple:
    """Read one DICOM, save the breast-cropped PNG. Returns (image_id, width, height, x0, y0, scale, bytes)."""
    import pydicom

    data = read(f"images/{study}/{image}.dicom")
    img, (x0, y0, scale) = crop_breast(dicom_to_gray(pydicom.dcmread(io.BytesIO(data))), STORE_HEIGHT)
    img.save(PNG_DIR / f"{image}.png")
    return image, img.width, img.height, x0, y0, scale, len(data)


def select_images(limit: int, root: Path = VINDR) -> pd.DataFrame:
    df = pd.read_csv(root / "breast-level_annotations.csv").sort_values(["study_id", "image_id"])
    if limit:  # smoke tests: a few images with findings and a few healthy ones
        findings = pd.read_csv(root / "finding_annotations.csv")
        boxed = set(findings.dropna(subset=["xmin"]).image_id)
        df = pd.concat([df[df.image_id.isin(boxed)].head(limit), df[~df.image_id.isin(boxed)].head(limit)])
    return df.reset_index(drop=True)


def download(args):
    from huggingface_hub import HfApi

    VINDR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(exist_ok=True)
    SHARD_DIR.mkdir(exist_ok=True)
    reader = None  # PhysioNet is only contacted for data that is not in the HF repo yet

    def get_reader():  # always called from the main thread
        nonlocal reader
        reader = reader or make_reader(args)
        return reader

    origin = f"local copy {args.source}" if args.source else "PhysioNet"
    api = repo = None
    remote = set()
    if not args.no_upload:
        api = HfApi()
        repo = hf_repo(api, args.hf_repo)
        remote = set(api.list_repo_files(repo, repo_type="dataset"))
    for name in CSV_FILES:
        if (VINDR / name).exists():
            continue
        if name in remote:
            pull(api, repo, name, VINDR)
        else:
            if args.source and not (args.source / name).is_file():
                raise SystemExit(f"{args.source / name} not found: run scripts/vindr_to_drive.py first")
            (VINDR / name).write_bytes(get_reader()(name))
            print(f"fetched {name} from {origin}", flush=True)
        if api and name not in remote:
            api.upload_file(path_or_fileobj=str(VINDR / name), path_in_repo=name, repo_id=repo, repo_type="dataset")

    images = select_images(args.limit)
    shards = split_shards(images, args.shard_size)
    print(f"{len(images)} images in {len(shards)} shards -> {repo or 'local only'}", flush=True)

    todo = []
    for k, shard in enumerate(shards):
        state = shard_state(k, shard, remote)
        if state == "local":
            print(f"shard {k + 1}/{len(shards)}: already here", flush=True)
        elif state == "remote":
            tar_name, tf_name = shard_name(k)
            pull(api, repo, tf_name, VINDR, flatten=True)
            pull(api, repo, tar_name, VINDR, extract=True)
            (SHARD_DIR / f"done_{k:03d}").touch()
            print(f"shard {k + 1}/{len(shards)}: restored from HF", flush=True)
        else:
            todo.append(k)

    n_todo = sum(len(shards[k]) for k in todo)
    read = get_reader() if todo else None  # HTTP mode: PhysioNet session created here, in the main thread
    if todo:
        print(f"converting {n_todo} images from {origin} ({len(todo)} shards)", flush=True)
    total_done, total_bytes, t0 = 0, 0, time.time()
    lock = threading.Lock()
    for k in todo:
        shard = shards[k]
        tar_name, tf_name = shard_name(k)
        rows, failed = [], []
        with ThreadPoolExecutor(args.workers) as ex:
            futures = {ex.submit(convert, read, r.study_id, r.image_id): r.image_id
                       for r in shard.itertuples()}
            for f in as_completed(futures):
                try:
                    image, w, h, x0, y0, scale, nbytes = f.result()
                except Exception as e:  # noqa: BLE001 - one broken file must not stop 20k downloads
                    failed.append(futures[f])
                    print(f"  failed {futures[f]}: {e}", flush=True)
                    continue
                rows.append(dict(image_id=image, width=w, height=h, x0=x0, y0=y0, scale=scale))
                with lock:
                    total_done += 1
                    total_bytes += nbytes
                    if total_done % 100 == 0:
                        dt = time.time() - t0
                        left = (n_todo - total_done) * dt / total_done
                        print(f"  {total_done}/{n_todo} images, {total_bytes / dt / 1e6:.1f} MB/s, "
                              f"ETA {left / 3600:.1f} h", flush=True)
        if len(failed) > 0.01 * len(shard):
            raise SystemExit(f"shard {k}: {len(failed)} images failed, stopping (re-run to resume)")
        tf_local = VINDR / tf_name.replace("shards/", "")
        pd.DataFrame(rows).to_csv(tf_local, index=False)
        if api:
            tar_path = SHARD_DIR / f"shard_{k:03d}.tar"
            with tarfile.open(tar_path, "w") as tar:
                for r in rows:
                    tar.add(PNG_DIR / f"{r['image_id']}.png", arcname=f"png/{r['image_id']}.png")
            api.upload_file(path_or_fileobj=str(tar_path), path_in_repo=tar_name, repo_id=repo, repo_type="dataset")
            api.upload_file(path_or_fileobj=str(tf_local), path_in_repo=tf_name, repo_id=repo, repo_type="dataset")
            tar_path.unlink()
        (SHARD_DIR / f"done_{k:03d}").touch()
        print(f"shard {k + 1}/{len(shards)}: {len(rows)} images saved{' and uploaded' if api else ''}", flush=True)
    merge_transforms()


def pull(api, repo, name, dest: Path, extract=False, flatten=False):
    """Download one file of the HF dataset into dest (tar shards are extracted and removed)."""
    from huggingface_hub import hf_hub_download

    tmp = hf_hub_download(repo, name, repo_type="dataset", local_dir=str(VINDR / "_hf"), token=api.token)
    if extract:
        with tarfile.open(tmp) as t:
            t.extractall(dest, filter="data")
        Path(tmp).unlink()
    else:
        os.replace(tmp, dest / (Path(name).name if flatten else name))


def restore(args):
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    repo = hf_repo(api, args.hf_repo)
    path = snapshot_download(repo, repo_type="dataset", local_dir=str(VINDR / "_hf"), max_workers=8)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for name in CSV_FILES:
        os.replace(Path(path) / name, VINDR / name)
    for tf in sorted((Path(path) / "shards").glob("transforms_*.csv")):
        os.replace(tf, VINDR / tf.name)
    for tar in sorted((Path(path) / "shards").glob("shard_*.tar")):
        with tarfile.open(tar) as t:
            t.extractall(VINDR, filter="data")
        tar.unlink()
        print(f"extracted {tar.name}", flush=True)
    merge_transforms()


def merge_transforms():
    parts = sorted(VINDR.glob("transforms_*.csv"))
    if parts:
        df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates("image_id")
        df.to_csv(VINDR / "transforms.csv", index=False)
        print(f"{len(df)} converted images listed in {VINDR / 'transforms.csv'}", flush=True)


STATUS_COMPLETE, STATUS_MISSING, STATUS_UNKNOWN = 0, 10, 11


def status(args) -> int:
    """Count shards that are neither converted here nor in the HF repo (same split as `download`).

    Read-only for HF (never creates the repo) and never contacts PhysioNet; at most pulls the small
    annotation CSVs from HF into data/vindr, which `download` would do anyway.
    """
    remote, hf, api, repo = set(), "off", None, None
    if not args.no_upload:
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import RepositoryNotFoundError

            api = HfApi()
            repo = args.hf_repo if "/" in args.hf_repo else f"{api.whoami()['name']}/{args.hf_repo}"
            try:
                remote = set(api.list_repo_files(repo, repo_type="dataset"))
            except RepositoryNotFoundError:
                print(f"HF dataset {repo} does not exist yet: nothing uploaded", flush=True)
            hf = "ok"
        except Exception as e:  # noqa: BLE001 - no token / no network: report 'unknown', do not crash
            print(f"HF not reachable ({type(e).__name__}: {str(e)[:200]}); only local shards counted", flush=True)
            api, hf = None, "unavailable"

    need = ["breast-level_annotations.csv"] + (["finding_annotations.csv"] if args.limit else [])
    root = None
    if all((VINDR / n).exists() for n in need):
        root = VINDR
    elif api and all(n in remote for n in need):
        VINDR.mkdir(parents=True, exist_ok=True)
        for n in need:
            if not (VINDR / n).exists():
                pull(api, repo, n, VINDR)
        root = VINDR
    elif args.source and all((args.source / n).is_file() for n in need):
        root = args.source

    if root is None:  # no annotation CSV anywhere: shard count unknown
        nothing_done = hf != "unavailable"  # `download` stores the CSV here / in HF before any shard
        print(f"annotation CSV not found locally{', in HF' if hf == 'ok' else ''}"
              f"{' or in ' + str(args.source) if args.source else ''}", flush=True)
        print(f"VINDR_STATUS total_shards=? missing_shards=? hf={hf}", flush=True)
        return STATUS_MISSING if nothing_done else STATUS_UNKNOWN

    shards = split_shards(select_images(args.limit, root), args.shard_size)
    states = [shard_state(k, shard, remote) for k, shard in enumerate(shards)]
    missing = states.count("missing")
    print(f"{len(shards)} shards: {states.count('local')} here, {states.count('remote')} in HF "
          f"({repo or 'HF not checked'}), {missing} missing", flush=True)
    print(f"VINDR_STATUS total_shards={len(shards)} missing_shards={missing} hf={hf}", flush=True)
    if not missing:
        return STATUS_COMPLETE
    return STATUS_UNKNOWN if hf == "unavailable" else STATUS_MISSING


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["download", "restore", "status"])
    ap.add_argument("--hf-repo", default="mammoai-vindr", help="private HF dataset (name or user/name)")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--source", metavar="DIR", help="read CSVs/DICOMs from a local copy instead of PhysioNet")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images with findings + N healthy")
    ap.add_argument("--no-upload", action="store_true", help="keep PNGs local only (status: ignore HF)")
    args = ap.parse_args()
    if args.source:
        args.source = Path(args.source).expanduser()
        if not args.source.is_dir():
            raise SystemExit(f"--source {args.source}: no such directory")
    if not args.base_url.endswith("/"):
        args.base_url += "/"
    if args.mode == "status":
        sys.exit(status(args))
    download(args) if args.mode == "download" else restore(args)


if __name__ == "__main__":
    main()
