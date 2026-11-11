"""
Azure ML Pipeline Components for modular ML workflows.
"""

from pathlib import Path
from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes


def create_preprocess_component(environment: str, compute: str):
    """Component for audio extraction and multimodal feature engineering."""
    return command(
        name="preprocess_video_features",
        display_name="Video & Audio Feature Extraction",
        description="Extracts FFmpeg mono audio and computes lexical & acoustic signals",
        environment=environment,
        compute=compute,
        inputs={
            "video_input": Input(type=AssetTypes.URI_FILE),
            "transcript_input": Input(type=AssetTypes.URI_FILE, optional=True),
        },
        outputs={
            "preprocessed_features": Output(type=AssetTypes.URI_FOLDER),
        },
        code="../../",
        command="""
        python -m src.cli analyze \
            --video ${{inputs.video_input}} \
            --transcript ${{inputs.transcript_input}} \
            --output-dir ${{outputs.preprocessed_features}}
        """,
    )


def create_training_component(environment: str, compute: str):
    """Component for PEFT LoRA fine-tuning."""
    return command(
        name="train_video_editorial_transformer",
        display_name="PEFT LoRA Fine-Tuning",
        description="Fine-tunes multi-head transformer with PEFT LoRA",
        environment=environment,
        compute=compute,
        inputs={
            "base_model": Input(type="string", default="distilbert-base-uncased"),
            "dataset": Input(type=AssetTypes.URI_FILE, optional=True),
            "epochs": Input(type="integer", default=3),
            "batch_size": Input(type="integer", default=8),
            "learning_rate": Input(type="number", default=0.0002),
        },
        outputs={
            "model_checkpoint": Output(type=AssetTypes.CUSTOM_MODEL),
        },
        code="../../",
        command="""
        python -m src.train \
            --base-model ${{inputs.base_model}} \
            --output-dir ${{outputs.model_checkpoint}} \
            --epochs ${{inputs.epochs}} \
            --batch-size ${{inputs.batch_size}} \
            --lr ${{inputs.learning_rate}} \
            --device cuda
        """,
    )


def create_evaluation_component(environment: str, compute: str):
    """Component for model benchmarking and error analysis."""
    return command(
        name="evaluate_editorial_model",
        display_name="Quantitative Model Evaluation",
        description="Calculates ROC-AUC, F1, RMSE, and Pearson correlation",
        environment=environment,
        compute=compute,
        inputs={
            "model_checkpoint": Input(type=AssetTypes.CUSTOM_MODEL),
            "base_model": Input(type="string", default="distilbert-base-uncased"),
        },
        outputs={
            "eval_results": Output(type=AssetTypes.URI_FOLDER),
        },
        code="../../",
        command="""
        python -m src.evaluate \
            --checkpoint-dir ${{inputs.model_checkpoint}} \
            --base-model ${{inputs.base_model}} \
            --output-dir ${{outputs.eval_results}}
        """,
    )

