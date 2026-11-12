"""Kaggle kernel entrypoint: build the real-data corpus and train on a free Kaggle GPU.

Runs inside a Kaggle notebook/script with Internet + GPU enabled. Steps:
  1. clone this repo and install the few packages Kaggle images lack
  2. collect all English TikTok hooks (~36.5k) and as many TTCC ads as the time budget allows
  3. merge with the attached private corpus dataset (vRetention + YouTube captions + earlier TTCC), dedupe by id
  4. train DistilBERT+LoRA on CUDA, evaluate on the group-aware held-out split
Outputs land in /kaggle/working (checkpoint, reports, corpus counts). Raw data is not copied to outputs.

Tunables (env vars): REPO_URL, REPO_BRANCH, TTCC_LIMIT, TTCC_MINUTES, TTCC_WORKERS, EPOCHS, BATCH_SIZE.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Nabilhassan12345/video-editorial-intelligence.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "devin/1794396833-antigravity-recovery")
TTCC_LIMIT = int(os.environ.get("TTCC_LIMIT", "61789"))
TTCC_MINUTES = float(os.environ.get("TTCC_MINUTES", "120"))
TTCC_WORKERS = int(os.environ.get("TTCC_WORKERS", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/repo") if Path("/kaggle/temp").exists() else Path("/tmp/repo")
DATA = REPO / "data" / "real"


def sh(cmd, check=True, **kw):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    r = subprocess.run(list(map(str, cmd)), **kw)
    print(f"[exit {r.returncode} after {time.time() - t0:.0f}s]", flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return r


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(64, 64, device="cuda")
            (x @ x).sum().item()
            print("GPU:", torch.cuda.get_device_name(0), flush=True)
            return "cuda"
    except Exception as e:  # e.g. GPU arch unsupported by the installed torch build
        print("CUDA unusable, falling back to CPU:", e, flush=True)
    return "cpu"


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a truncated last line
    return out


def _rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def collect_ttcc_parallel(py, audio):
    """Collect TTCC one parquet file per subprocess, TTCC_WORKERS at a time.

    One file per process means a crashed/OOM-killed worker loses only that file (it is retried
    once); subprocesses also isolate the known datasets-streaming teardown crash.
    """
    from huggingface_hub import HfApi

    n_files = len([f for f in HfApi().list_repo_files("liangyuch/ttcc-v0_2_0", repo_type="dataset")
                   if f.endswith(".parquet")])
    env = dict(os.environ, USE_TF="0", USE_JAX="0", TOKENIZERS_PARALLELISM="false")
    logs = WORK / "ttcc_logs"
    logs.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + TTCC_MINUTES * 60
    queue = [(i, 0) for i in range(n_files)]
    running = {}
    done_rows, failed = 0, []
    print(f"\n$ ttcc collection: {n_files} parquet files, {TTCC_WORKERS} workers, "
          f"{TTCC_MINUTES:.0f} min budget", flush=True)
    t0 = time.time()
    while queue or running:
        while queue and len(running) < TTCC_WORKERS and time.time() < deadline - 60:
            i, attempt = queue.pop(0)
            out = DATA / f"ttcc_kaggle_{i:03d}.jsonl"
            remaining = max(1.0, (deadline - time.time()) / 60)
            cmd = py + ["--source", "ttcc", "--limit", TTCC_LIMIT, "--no-transcribe",
                        "--max-minutes", f"{remaining:.1f}", "--num-shards", n_files, "--shard-index", i,
                        "--out", out, *audio]
            log = open(logs / f"file_{i:03d}_try{attempt}.log", "w")
            running[i] = (subprocess.Popen(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT,
                                           env=env), attempt, out, log)
        if not running:
            break
        time.sleep(5)
        for i in list(running):
            proc, attempt, out, log = running[i]
            if proc.poll() is None:
                continue
            log.close()
            del running[i]
            n = _rows(out)
            if n == 0 and attempt == 0:
                queue.append((i, 1))
                print(f"  file {i}: 0 rows (exit {proc.returncode}), retrying", flush=True)
                continue
            if n == 0:
                failed.append(i)
            done_rows += n
            print(f"  file {i}: {n} rows (exit {proc.returncode}); total {done_rows}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    skipped = [i for i, _ in queue]
    print(f"[ttcc collection: {done_rows} rows in {time.time() - t0:.0f}s; "
          f"failed files {failed}; not started (budget) {skipped}]", flush=True)


def main():
    t_start = time.time()
    if REPO.exists():
        shutil.rmtree(REPO)
    sh(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, REPO])
    sh([sys.executable, "-m", "pip", "install", "-q", "peft>=0.10.0", "datasets>=3.0", "scikit-learn", "pydantic>=2"])
    os.chdir(REPO)
    DATA.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-m", "src.data.real.collect"]

    sh(py + ["--source", "tiktok-hooks", "--limit", "46605", "--lang", "en", "--out", DATA / "hooks_full.jsonl"])

    audio = [] if shutil.which("ffmpeg") else ["--no-audio"]
    collect_ttcc_parallel(py, audio)

    files = [DATA / "hooks_full.jsonl"] + sorted(DATA.glob("ttcc_kaggle_*.jsonl"))
    files += [Path(p) for p in sorted(glob.glob("/kaggle/input/**/*.jsonl", recursive=True))]
    merged, seen = [], set()
    for path in files:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        kept = 0
        for r in rows:
            key = r.get("id") or r.get("text")
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
            kept += 1
        print(f"{path}: {len(rows)} rows, {kept} new", flush=True)

    combined = DATA / "combined_kaggle.jsonl"
    with open(combined, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    counts = Counter(r.get("source", "unknown") for r in merged)
    hook_pos = Counter(r.get("source", "unknown") for r in merged if r.get("hook_label") == 1.0)
    corpus = {"total": len(merged), "by_source": dict(counts), "hook_positive_by_source": dict(hook_pos)}
    print(json.dumps(corpus, indent=2), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "corpus_counts.json").write_text(json.dumps(corpus, indent=2))

    device = pick_device()
    ckpt = WORK / "editorial_model_kaggle"
    t_train = time.time()
    sh([sys.executable, "-m", "src.train", "--dataset-path", combined, "--epochs", EPOCHS,
        "--batch-size", BATCH_SIZE, "--device", device, "--output-dir", ckpt])
    train_s = time.time() - t_train
    sh([sys.executable, "-m", "src.evaluate", "--checkpoint-dir", ckpt, "--dataset-path", combined,
        "--device", device, "--output-dir", WORK / "reports"])

    (WORK / "run_info.json").write_text(json.dumps({
        "device": device, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t_start, 1),
        "repo_branch": REPO_BRANCH,
    }, indent=2))
    print("Done. Outputs in", WORK, flush=True)


if __name__ == "__main__":
    main()
