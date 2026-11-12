"""Launch / monitor the free-GPU Kaggle training run from your machine.

Requires the Kaggle CLI (`pip install kaggle`) and KAGGLE_USERNAME / KAGGLE_KEY
(or ~/.kaggle/kaggle.json). Your locally collected corpus is uploaded as a
**private** Kaggle dataset (the source datasets are research-use only, so never
make it public).

    python kaggle/launch.py push      # upload corpus + start the GPU kernel
    python kaggle/launch.py status    # queued / running / complete / error
    python kaggle/launch.py fetch     # download checkpoint + reports to kaggle_output/
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle_build"
DATASET_SLUG = "editorial-real-corpus"
KERNEL_SLUG = "video-editorial-train"
CORPUS_FILES = ["ttcc_train_big.jsonl", "vret.jsonl", "youtube_transcripts.jsonl"]


def kaggle(*args, check=True):
    exe = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run([exe, *args], check=check, text=True, capture_output=False)


def username() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        user = json.loads(cfg.read_text())["username"] if cfg.exists() else None
    if not user:
        raise SystemExit("Set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json")
    return user


def push(user: str, data_dir: Path, gpu: str):
    ds_dir = BUILD / "dataset"
    shutil.rmtree(BUILD, ignore_errors=True)
    ds_dir.mkdir(parents=True)
    for name in CORPUS_FILES:
        src = data_dir / name
        if src.exists():
            shutil.copy(src, ds_dir / name)
    if not any(ds_dir.iterdir()):
        raise SystemExit(f"No corpus files found in {data_dir}; see docs/DATASETS.md")
    (ds_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}],
    }))
    exists = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(ds_dir), "-m", "update corpus", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(ds_dir), "-r", "zip")  # private by default

    k_dir = BUILD / "kernel"
    k_dir.mkdir()
    shutil.copy(ROOT / "kaggle" / "run_kaggle.py", k_dir / "run_kaggle.py")
    meta = {
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run_kaggle.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
    }
    if gpu:
        meta["machine_shape"] = gpu
    (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle("kernels", "push", "-p", str(k_dir))
    print(f"\nKernel: https://www.kaggle.com/code/{user}/{KERNEL_SLUG}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["push", "status", "fetch"])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "real"))
    p.add_argument("--gpu", default="NvidiaTeslaT4", help="Kaggle machine_shape; '' for Kaggle's default GPU")
    p.add_argument("--out", default=str(ROOT / "kaggle_output"))
    a = p.parse_args()
    user = username()
    if a.action == "push":
        push(user, Path(a.data_dir), a.gpu)
    elif a.action == "status":
        kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}")
    else:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", a.out)


if __name__ == "__main__":
    main()
