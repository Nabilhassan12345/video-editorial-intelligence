# 🎬 Video Editorial Intelligence — Quick Demonstration Guide

This guide provides a verified, step-by-step walkthrough to install the environment, load the fine-tuned model, analyze a video end-to-end, inspect results, and launch the interactive demo.

---

## 1. Environment Setup

### Clone Repository & Create Environment
```bash
git clone https://github.com/Nabilhassan12345/video-editorial-intelligence.git
cd video-editorial-intelligence

# Create virtual environment (Python 3.10 - 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies
Ensure **FFmpeg** is installed on your system for audio extraction:
- **macOS:** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt-get install -y ffmpeg`
- **Windows:** `choco install ffmpeg`

---

## 2. Verify Fine-Tuned Checkpoint

The repository includes reproducible training and dry-run scripts. Verify your pipeline without downloading expensive artifacts:

```bash
# Run training dry-run verification
python -m src.train --dry-run
```

To run a quick 3-epoch local fine-tuning run:
```bash
python -m src.train --epochs 3 --batch-size 8 --output-dir checkpoints/editorial_model
```

---

## 3. Analyze Video & Transcripts

### Analyze Media File End-to-End
```bash
python -m src.cli analyze \
  --video examples/sample_video.mp4 \
  --transcript examples/sample_transcript.json \
  --output-dir outputs/reports
```

### Analyze Raw Transcript String
```bash
python -m src.cli analyze \
  --transcript "[00:00] In the next 60 seconds I will show you the exact strategy top creators use. [00:15] Let me pause here and find my notes... uh, wait a second." \
  --output-dir outputs/reports
```

---

## 4. Launch Interactive Streamlit Demonstration

Launch the full interactive demonstration dashboard:

```bash
streamlit run demo/app.py
```
Open `http://localhost:8501` in your browser.

**Features in Demo:**
- Video upload & real-time preview
- Spoken transcript inspection
- Modeled audience retention progression graph
- Strategic editorial recommendations
- Retention drop-off alert cards with grounded evidence
- 1-click JSON, Markdown, and HTML report downloads

---

## 5. Run Model Benchmark Evaluation

Benchmark the fine-tuned model against the baseline:

```bash
python -m src.evaluate --checkpoint-dir checkpoints/editorial_model
```
Outputs are written to `outputs/evaluation/evaluation_summary.md` and `outputs/evaluation/evaluation_results.json`.

---

## 6. Microsoft Azure Cloud Workflows (Optional)

### Azure Machine Learning SDK v2
```bash
# Preview job specification (Dry-run mode, no charges)
python -m azure.ml.submit_training

# Register model asset (Dry-run preview)
python -m azure.ml.register_model
```

### Azure AI Foundry
```bash
# Test Foundry model inference (with offline simulation mode fallback)
python -m azure.foundry.inference --transcript examples/sample_transcript.json
```

