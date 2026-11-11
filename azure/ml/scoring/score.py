"""
Azure ML Online Endpoint Scoring Script.
"""
import json
import os
import torch
from src.pipeline.video_pipeline import VideoEditorialPipeline

pipeline = None

def init():
    global pipeline
    model_dir = os.getenv("AZUREML_MODEL_DIR", "checkpoints/editorial_model")
    pipeline = VideoEditorialPipeline(checkpoint_dir=model_dir)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        transcript = data.get("transcript", "")
        result = pipeline.run(transcript_input=transcript)
        return result.to_dict()
    except Exception as e:
        return {"error": str(e)}
