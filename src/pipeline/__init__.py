"""
End-to-end video analysis pipeline and editorial report generation.
"""

from .video_pipeline import VideoEditorialPipeline, PipelineResult
from .report_generator import ReportGenerator

__all__ = ["VideoEditorialPipeline", "PipelineResult", "ReportGenerator"]

