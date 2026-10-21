import argparse
import ssl
import sys
import urllib.request
from pathlib import Path

BASE_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
            "flat-connectome/")
FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK = 1 << 20


def human(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def remote_size(url: str, ctx: ssl.SSLContext) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def download(name: str, out_dir: Path, ctx: ssl.SSLContext) -> None:
    url = BASE_URL + name
    path = out_dir / name
    part = path.with_suffix(path.suffix + ".part")
    total = remote_size(url, ctx)
    if path.exists() and path.stat().st_size == total:
        print(f"{name}: done already ({human(total)})")
        return
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    print(f"{name}: {human(total)}, resuming at {human(have)}" if have else
          f"{name}: {human(total)}")
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        if have and resp.status != 206:
            print("  server ignored the resume request, starting over")
            have = 0
        mode = "ab" if have else "wb"
        with open(part, mode) as f:
            done = have
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                f.write(block)
                done += len(block)
                print(f"\r  {human(done)} / {human(total)} ({100 * done / total:.1f}%)", end="")
    print()
    if part.stat().st_size != total:
        sys.exit(f"{name}: size {part.stat().st_size} != {total}, run again to resume")
    part.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DATA_DIR))
    parser.add_argument("--only", choices=FILES, nargs="+")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    for name in args.only or FILES:
        download(name, out_dir, ctx)


if __name__ == "__main__":
    main()
