"""
Unified Command-Line Interface (CLI) for Video Editorial Intelligence.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from src.models.config import EditorialModelConfig, TrainingConfig
from src.pipeline.report_generator import ReportGenerator
from src.pipeline.video_pipeline import VideoEditorialPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cmd_analyze(args):
    """Analyze a video file or transcript."""
    pipeline = VideoEditorialPipeline(
        checkpoint_dir=args.checkpoint_dir,
        base_model_name=args.base_model,
        device=args.device,
    )

    transcript_data = None
    if args.transcript:
        t_path = Path(args.transcript)
        if t_path.exists():
            with open(t_path, "r", encoding="utf-8") as f:
                transcript_data = f.read()
        else:
            transcript_data = args.transcript

    logger.info("Executing Video Editorial Intelligence pipeline...")
    result = pipeline.run(video_path=args.video, transcript_input=transcript_data)

    # Save reports
    saved = ReportGenerator.save_reports(result, args.output_dir, base_name=args.report_name)
    print("\n" + "=" * 60)
    print("🎬 VIDEO EDITORIAL INTELLIGENCE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Overall Quality Score: {result.quality_score:.1f} / 10.0")
    print(f"Optimal Hook:          {result.hook}")
    print(f"Hook Confidence:       {result.hook_confidence * 100:.1f}%")
    print(f"Retention Risks:       {len(result.drop_off_risks)} identified")
    print(f"JSON Report:           {saved['json']}")
    print(f"Markdown Report:       {saved['markdown']}")
    print(f"HTML Report:           {saved['html']}")
    print("=" * 60 + "\n")


def cmd_train(args):
    """Fine-tune the model or run dry-run verification."""
    from src.train import run_training

    model_config = EditorialModelConfig(base_model_name=args.base_model, use_peft=True)
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    run_training(model_config, train_config, dataset_path=args.dataset_path)


def cmd_evaluate(args):
    """Evaluate models and generate benchmark summaries."""
    from src.evaluate import main as eval_main
    sys.argv = [
        "evaluate.py",
        "--checkpoint-dir", args.checkpoint_dir or "checkpoints/editorial_model",
        "--base-model", args.base_model,
        "--output-dir", args.output_dir,
        "--device", args.device,
    ]
    eval_main()


def cmd_demo(args):
    """Launch the interactive Streamlit demonstration."""
    demo_app = Path(__file__).resolve().parent.parent / "demo" / "app.py"
    if not demo_app.exists():
        print(f"Demo file not found at {demo_app}")
        sys.exit(1)

    print(f"Starting Streamlit Interactive Demo on port {args.port}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_app), "--server.port", str(args.port)])


def main():
    parser = argparse.ArgumentParser(
        prog="video-editorial",
        description="Video Editorial Intelligence - Applied ML & NLP for Video Content Optimization",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze video or transcript")
    p_analyze.add_argument("--video", type=str, default=None, help="Path to input video file")
    p_analyze.add_argument("--transcript", type=str, default=None, help="Transcript text or path to transcript file")
    p_analyze.add_argument("--checkpoint-dir", type=str, default="checkpoints/editorial_model", help="Path to fine-tuned checkpoint")
    p_analyze.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model backbone")
    p_analyze.add_argument("--output-dir", type=str, default="./outputs/reports", help="Output directory for reports")
    p_analyze.add_argument("--report-name", type=str, default="editorial_analysis", help="Base filename for reports")
    p_analyze.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', 'mps', 'cuda'")
    p_analyze.set_defaults(func=cmd_analyze)

    # train
    p_train = subparsers.add_parser("train", help="Fine-tune multi-head transformer with PEFT LoRA")
    p_train.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model")
    p_train.add_argument("--dataset-path", type=str, default=None, help="Dataset path")
    p_train.add_argument("--output-dir", type=str, default="./checkpoints/editorial_model", help="Checkpoint output directory")
    p_train.add_argument("--epochs", type=int, default=3, help="Epochs")
    p_train.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p_train.add_argument("--seed", type=int, default=42, help="Seed")
    p_train.add_argument("--device", type=str, default="cpu", help="Device")
    p_train.add_argument("--dry-run", action="store_true", help="Perform dry-run verification")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model against benchmarks")
    p_eval.add_argument("--checkpoint-dir", type=str, default="checkpoints/editorial_model", help="Checkpoint directory")
    p_eval.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model")
    p_eval.add_argument("--output-dir", type=str, default="./outputs/evaluation", help="Output directory")
    p_eval.add_argument("--device", type=str, default="cpu", help="Device")
    p_eval.set_defaults(func=cmd_evaluate)

    # demo
    p_demo = subparsers.add_parser("demo", help="Launch interactive Streamlit UI")
    p_demo.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

