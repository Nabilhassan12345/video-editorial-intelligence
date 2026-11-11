"""
End-to-End Video Editorial Intelligence Pipeline.

Orchestrates:
1. Video file validation
2. FFmpeg audio extraction
3. Transcript extraction & segmentation
4. Multimodal acoustic & lexical feature extraction
5. Multi-head Transformer inference (PEFT fine-tuned)
6. Grounded editorial intelligence synthesis
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from ..audio.extractor import AudioExtractor, AudioFeatures
from ..data.feature_engineering import FeatureEngineer, MultimodalSegment
from ..data.segmenter import TranscriptSegmenter, VideoSegment
from ..models.config import EditorialModelConfig
from ..models.multi_head_transformer import VideoEditorialTransformer
from ..models.peft_adapter import load_fine_tuned_model

logger = logging.getLogger(__name__)


@dataclass
class EditorialDropOffRisk:
    timestamp: str
    start_time: float
    end_time: float
    risk_score: float
    reason: str
    explanation: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "risk_score": round(self.risk_score, 3),
            "reason": self.reason,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class PipelineResult:
    """Standardized output container for Video Editorial Intelligence results."""
    hook: str
    hook_segment_index: int
    hook_confidence: float
    drop_off_risks: List[EditorialDropOffRisk]
    quality_score: float
    improvements: List[str]
    segments: List[Dict[str, Any]]
    retention_curve: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook": self.hook,
            "hook_segment_index": self.hook_segment_index,
            "hook_confidence": round(self.hook_confidence, 3),
            "drop_off_risks": [r.to_dict() for r in self.drop_off_risks],
            "quality_score": round(self.quality_score, 2),
            "improvements": self.improvements,
            "segments": self.segments,
            "retention_curve": self.retention_curve,
            "metadata": self.metadata,
        }


class VideoEditorialPipeline:
    """Complete end-to-end video analysis and editorial intelligence engine."""

    def __init__(
        self,
        checkpoint_dir: Optional[str] = "checkpoints/editorial_model",
        base_model_name: str = "distilbert-base-uncased",
        device: str = "cpu",
    ):
        self.device = device
        self.base_model_name = base_model_name
        self.checkpoint_dir = checkpoint_dir

        self.audio_extractor = AudioExtractor()
        self.segmenter = TranscriptSegmenter()
        self.feature_engineer = FeatureEngineer()

        # Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except Exception as e:
            logger.warning(f"Could not load online tokenizer ({e}); using token fallback.")
            self.tokenizer = None

        # Load Model
        self.model, self.model_source = self._init_model()
        self.model_status = "fine_tuned" if self.model_source.startswith("fine_tuned") else "untrained_heads"
        if self.model_status != "fine_tuned":
            logger.warning(
                "No fine-tuned checkpoint loaded: prediction heads are randomly initialised and their "
                "scores are NOT meaningful. Train one with: python -m src.train --output-dir %s",
                self.checkpoint_dir,
            )

    def _init_model(self) -> Tuple[VideoEditorialTransformer, str]:
        """Load fine-tuned checkpoint or fallback to initialized architecture."""
        if self.checkpoint_dir and Path(self.checkpoint_dir).exists():
            try:
                model = load_fine_tuned_model(self.checkpoint_dir, device=self.device)
                logger.info(f"Loaded fine-tuned model checkpoint from {self.checkpoint_dir}")
                return model, f"fine_tuned_checkpoint ({self.checkpoint_dir})"
            except Exception as e:
                logger.warning(f"Failed to load checkpoint ({e}). Using initialized model.")

        cfg = EditorialModelConfig(base_model_name=self.base_model_name, use_peft=False)
        model = VideoEditorialTransformer(cfg)
        model.to(self.device)
        model.eval()
        return model, f"base_initialized ({self.base_model_name})"

    def run(
        self,
        video_path: Optional[str] = None,
        transcript_input: Optional[Union[str, List[Dict[str, Any]]]] = None,
        output_wav_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute full editorial intelligence pipeline.
        
        Args:
            video_path: Path to video file (MP4, MKV, MOV, WebM).
            transcript_input: Optional raw text or timestamped JSON transcript.
            output_wav_path: Optional path for audio extraction.
            
        Returns:
            PipelineResult dataclass.
        """
        start_time = time.time()
        metadata: Dict[str, Any] = {
            "pipeline_version": "1.0.0",
            "model_source": self.model_source,
            "model_status": self.model_status,
            "prediction_note": (
                "Scores come from a checkpoint fine-tuned on the bundled synthetic template dataset; "
                "they are editorial estimates, not measured audience retention."
                if self.model_status == "fine_tuned"
                else "UNTRAINED prediction heads: model scores are random and must not be interpreted. "
                "Only the transcript/acoustic evidence (WPM, silence, fillers) is meaningful."
            ),
            "device": self.device,
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        # Step 1 & 2: Audio extraction from video
        extracted_wav: Optional[str] = None
        if video_path:
            v_path = self.audio_extractor.validate_video_file(video_path)
            metadata["video_filename"] = v_path.name
            metadata["video_size_mb"] = round(v_path.stat().st_size / (1024 * 1024), 2)

            if self.audio_extractor.is_available:
                try:
                    wav_path = self.audio_extractor.extract_audio(str(v_path), output_wav_path=output_wav_path)
                    extracted_wav = str(wav_path)
                    metadata["audio_extracted"] = True
                    metadata["audio_path"] = extracted_wav
                except Exception as e:
                    logger.warning(f"Audio extraction failed: {e}. Continuing with text analysis.")
                    metadata["audio_extracted"] = False
                    metadata["audio_error"] = str(e)
            else:
                metadata["audio_extracted"] = False
                metadata["audio_note"] = "FFmpeg not detected; acoustic metrics estimated from transcript."

        # Step 3: Obtain Transcript
        raw_text = transcript_input
        if not raw_text and video_path:
            # Check for whisper or transcript sidecar
            sidecar = Path(video_path).with_suffix(".json")
            if sidecar.exists():
                with open(sidecar, "r") as f:
                    raw_text = json.load(f)
            else:
                raw_text = self._attempt_whisper_transcription(extracted_wav)

        if not raw_text:
            raise ValueError(
                "No transcript provided and automatic speech recognition could not transcribe the media. "
                "Please provide a transcript string or timestamped JSON."
            )

        # Step 4: Adaptive Segmentation
        segments = self.segmenter.parse_raw_input(raw_text)
        if not segments:
            raise ValueError("Segmentation produced zero valid segments from input.")

        # Step 5: Feature Extraction & Multimodal Alignment
        audio_features = self.audio_extractor.extract_features_for_segments(
            extracted_wav,
            [s.to_dict() for s in segments],
        )
        multimodal_segments = self.feature_engineer.process_timeline(segments, audio_features)

        # Step 6: Neural Multi-Head Model Inference
        inference_outputs = self._run_model_inference(multimodal_segments)

        # Step 7: Synthesize Editorial Intelligence
        result = self._synthesize_editorial_insights(
            multimodal_segments,
            inference_outputs,
            metadata=metadata,
            duration=time.time() - start_time,
        )

        return result

    def _attempt_whisper_transcription(self, wav_path: Optional[str]) -> str:
        """Attempt transcription using local whisper if installed, else raise informative note."""
        if not wav_path or not Path(wav_path).exists():
            return ""
        try:
            import whisper
            logger.info("Transcribing audio using OpenAI Whisper (tiny)...")
            wmodel = whisper.load_model("tiny")
            res = wmodel.transcribe(wav_path)
            return res.get("text", "")
        except ImportError:
            logger.info("Whisper package not installed. Skipping automatic transcription.")
            return ""
        except Exception as e:
            logger.warning(f"Whisper transcription failed ({e}).")
            return ""

    def _run_model_inference(self, mm_segments: List[MultimodalSegment]) -> Dict[str, np.ndarray]:
        """Run batch inference through the Multi-Head Video Editorial Transformer."""
        texts = [m.segment.text for m in mm_segments]
        feat_vectors = np.stack([m.feature_vector for m in mm_segments], axis=0)

        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
        else:
            # Fallback simple token indexing
            input_ids = torch.ones((len(texts), 16), dtype=torch.long, device=self.device)
            attention_mask = torch.ones((len(texts), 16), dtype=torch.long, device=self.device)

        feat_tensor = torch.tensor(feat_vectors, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_vector=feat_tensor,
            )

        return {
            "hook_probs": outputs["hook_prob"].cpu().numpy(),
            "drop_off_risks": outputs["drop_off_risk"].cpu().numpy(),
            "quality_scores": outputs["quality_score"].cpu().numpy(),
        }

    def _synthesize_editorial_insights(
        self,
        mm_segments: List[MultimodalSegment],
        inference_outputs: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        duration: float,
    ) -> PipelineResult:
        """Combine neural scores and acoustic/lexical heuristics into actionable editorial guidance."""
        hook_probs = inference_outputs["hook_probs"]
        drop_risks = inference_outputs["drop_off_risks"]
        quality_scores = inference_outputs["quality_scores"]

        # 1. Best Hook Identification
        # High priority given to early segments (< 30s)
        weighted_hook_scores = []
        for i, (mm, hp) in enumerate(zip(mm_segments, hook_probs)):
            time_penalty = 1.0 if mm.segment.start_time <= 20.0 else max(0.2, 1.0 - (mm.segment.start_time / 100.0))
            score = float(hp) * 0.6 + (mm.hook_cue_score * 0.4)
            weighted_hook_scores.append(score * time_penalty)

        best_hook_idx = int(np.argmax(weighted_hook_scores))
        best_hook_seg = mm_segments[best_hook_idx]
        hook_text = f"{best_hook_seg.segment.start_timestamp} \"{best_hook_seg.segment.text}\""
        hook_conf = float(np.clip(weighted_hook_scores[best_hook_idx], 0.0, 1.0))

        # 2. Drop-off Risk Detection
        drop_off_risks: List[EditorialDropOffRisk] = []
        for i, (mm, dp) in enumerate(zip(mm_segments, drop_risks)):
            risk_val = float(dp)
            # Flag if predicted risk is elevated or silence/filler is high
            has_silence_issue = mm.audio_features.silence_ratio > 0.25
            has_wpm_issue = mm.audio_features.speaking_rate_wpm < 110.0 and mm.segment.duration > 4.0
            has_filler_issue = mm.filler_word_ratio > 0.15

            if risk_val >= 0.40 or has_silence_issue or has_wpm_issue or has_filler_issue:
                # Grounded diagnostic explanation
                reasons = []
                explanations = []
                evidence = {
                    "speaking_rate_wpm": mm.audio_features.speaking_rate_wpm,
                    "silence_ratio": round(mm.audio_features.silence_ratio * 100, 1),
                    "lexical_density": round(mm.lexical_density, 3),
                    "filler_word_ratio": round(mm.filler_word_ratio * 100, 1),
                    "predicted_churn_prob": round(risk_val, 3),
                }

                if has_silence_issue:
                    reasons.append("Dead air / excessive silence")
                    explanations.append(f"Acoustic analysis detected {evidence['silence_ratio']}% silence gap within this window.")
                if has_wpm_issue:
                    reasons.append("Speaking rate stagnation")
                    explanations.append(f"Pacing dropped to {evidence['speaking_rate_wpm']} WPM (optimal: 140–180 WPM).")
                if has_filler_issue:
                    reasons.append("High filler word density")
                    explanations.append(f"Filler density reached {evidence['filler_word_ratio']}%, reducing narrative clarity.")
                if not reasons:
                    reasons.append("Model-flagged drop-off risk")
                    explanations.append(
                        f"Drop-off head score {risk_val:.2f} exceeded the 0.40 threshold; "
                        "no acoustic or lexical rule fired for this segment."
                    )

                drop_off_risks.append(
                    EditorialDropOffRisk(
                        timestamp=mm.segment.start_timestamp,
                        start_time=mm.segment.start_time,
                        end_time=mm.segment.end_time,
                        risk_score=max(risk_val, 0.45 if has_silence_issue else 0.4),
                        reason=", ".join(reasons),
                        explanation=" ".join(explanations),
                        evidence=evidence,
                    )
                )

        # 3. Holistic Quality Score
        avg_quality = float(np.mean(quality_scores))
        # Factor in penalties for excessive drop-off points
        quality_score = float(np.clip(avg_quality - (len(drop_off_risks) * 0.2), 1.0, 10.0))

        # 4. Actionable Grounded Improvements
        improvements: List[str] = []
        if best_hook_idx > 0:
            improvements.append(
                f"Front-load your primary hook: Move segment at {best_hook_seg.segment.start_timestamp} to [00:00] for immediate viewer capture."
            )
        for r in drop_off_risks[:3]:
            if "silence" in r.reason.lower():
                improvements.append(
                    f"Trim dead air at {r.timestamp}: Cut pauses or overlay B-roll/music to maintain momentum."
                )
            elif "speaking rate" in r.reason.lower():
                improvements.append(
                    f"Accelerate pacing at {r.timestamp}: Condense dialogue or jump-cut pauses to raise speaking tempo closer to 150 WPM."
                )
            elif "filler" in r.reason.lower():
                improvements.append(
                    f"Eliminate filler words at {r.timestamp} to elevate perceived authority and retention."
                )

        if not improvements:
            improvements.append("Structural pacing is well-balanced. Maintain dynamic visual changes every 4-6 seconds.")

        # 5. Segment Details & Modeled Retention Curve
        detailed_segments = []
        retention_curve = []
        current_retention = 100.0  # Starts at 100%

        for i, (mm, hp, dp, qs) in enumerate(zip(mm_segments, hook_probs, drop_risks, quality_scores)):
            # Simulated retention progression based on segment risk and time
            drop_decay = float(dp) * 4.5 + (0.5 if i > 0 else 0.0)
            current_retention = max(18.0, current_retention - drop_decay)

            retention_curve.append({
                "timestamp": mm.segment.start_timestamp,
                "seconds": round(mm.segment.start_time, 1),
                "modeled_retention_index": round(current_retention, 1),
            })

            detailed_segments.append({
                "index": i,
                "start_time": round(mm.segment.start_time, 2),
                "end_time": round(mm.segment.end_time, 2),
                "start_timestamp": mm.segment.start_timestamp,
                "end_timestamp": mm.segment.end_timestamp,
                "text": mm.segment.text,
                "hook_score": round(float(hp), 3),
                "risk_score": round(float(dp), 3),
                "quality_score": round(float(qs), 2),
                "wpm": round(mm.audio_features.speaking_rate_wpm, 1),
                "silence_ratio": round(mm.audio_features.silence_ratio, 3),
            })

        metadata["retention_curve_note"] = (
            "modeled_retention_index is a heuristic index (100 = start) accumulated from per-segment "
            "drop-off scores. It is not a measured or calibrated audience-retention percentage."
        )
        metadata["total_processing_time_sec"] = round(duration, 3)
        metadata["segment_count"] = len(mm_segments)
        metadata["total_duration_sec"] = round(mm_segments[-1].segment.end_time, 2) if mm_segments else 0.0

        return PipelineResult(
            hook=hook_text,
            hook_segment_index=best_hook_idx,
            hook_confidence=hook_conf,
            drop_off_risks=drop_off_risks,
            quality_score=quality_score,
            improvements=improvements,
            segments=detailed_segments,
            retention_curve=retention_curve,
            metadata=metadata,
        )
