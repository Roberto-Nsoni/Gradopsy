"""
Download the four MNIST files into data/raw/mnist/ (ignored by git) and check them.

The files are the original IDX archives. yann.lecun.com refuses scripted downloads, so they come
from the PyTorch mirror — the same bytes, which the md5 sums below (the published ones) confirm.

    python scripts/get_mnist.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "mnist"


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = DEST / name
        if path.exists() and md5(path) == expected:
            print(f"ok    {name}")
            continue
        print(f"fetch {name}")
        urllib.request.urlretrieve(MIRROR + name, path)
        if md5(path) != expected:
            path.unlink()
            sys.exit(f"{name}: checksum mismatch, file removed")
        print(f"ok    {name}")


if __name__ == "__main__":
    main()
