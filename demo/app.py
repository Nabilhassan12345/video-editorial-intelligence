"""
Video Editorial Intelligence - Interactive Demonstration Application.
Built with Streamlit.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.pipeline.report_generator import ReportGenerator
from src.pipeline.video_pipeline import VideoEditorialPipeline

st.set_page_config(
    page_title="Video Editorial Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0c10;
    }
    .metric-card {
        background: #15171e;
        border: 1px solid #232733;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-active {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-warning {
        background-color: rgba(234, 179, 8, 0.15);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline(checkpoint_dir: str):
    return VideoEditorialPipeline(checkpoint_dir=checkpoint_dir)


def main():
    st.sidebar.title("🎬 Video Editorial Intelligence")
    st.sidebar.caption("Applied ML & NLP for Video Content Optimization")

    st.sidebar.markdown("---")
    checkpoint_dir = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="checkpoints/editorial_model",
        help="Path to fine-tuned PEFT LoRA checkpoint directory.",
    )

    chk_exists = get_pipeline(checkpoint_dir).model_status == "fine_tuned"
    if chk_exists:
        st.sidebar.markdown(
            '<div class="status-badge badge-active">● Fine-Tuned Checkpoint Loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-badge badge-warning">⚠ No fine-tuned checkpoint: scores are NOT meaningful</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Project")
    st.sidebar.info(
        "Transforms spoken narrative and acoustic signals into retention forecasting, "
        "hook classification, and grounded editorial guidance."
    )

    st.title("Video Editorial Intelligence Engine")
    st.markdown(
        "Analyze video retention, detect attention drop-off moments, rank introductory hooks, "
        "and generate grounded editorial recommendations."
    )

    tab_analyze, tab_sample, tab_about = st.tabs(["🚀 Live Analysis", "📖 Sample Demonstrator", "🧠 Model Architecture"])

    with tab_analyze:
        col_input, col_config = st.columns([2, 1])

        with col_input:
            uploaded_video = st.file_uploader(
                "Upload Video File (Optional, for audio & visual extraction)",
                type=["mp4", "mov", "mkv", "webm"],
                help="Extracts acoustic features (RMS energy, silence duration, speaking rate) via FFmpeg.",
            )

            default_text = (
                "[00:00] In the next 60 seconds, I'll reveal the exact retention strategy top creators use.\n"
                "[00:12] Most beginners assume fancy camera gear is what matters most.\n"
                "[00:24] Um, so let me pause here... uh, actually let me adjust my mic for a second.\n"
                "[00:38] But here is the truth: viewer drop-off happens when narrative pacing stagnates.\n"
                "[00:52] If you apply this three-step framework, your average watch duration will double."
            )

            transcript_text = st.text_area(
                "Video Transcript (Timestamped [MM:SS] or plain text)",
                value=default_text,
                height=180,
                help="Provide spoken transcript. If video is uploaded without transcript, ASR will be attempted.",
            )

        with col_config:
            st.markdown("### Execution Settings")
            st.write(f"**FFmpeg Detected:** {'✅ Yes' if get_pipeline(checkpoint_dir).audio_extractor.is_available else '❌ No (Using Text Heuristics)'}")
            st.write(f"**Device:** `CPU / Metal Acceleration`")
            st.write(f"**Encoder Backbone:** `distilbert-base-uncased`")
            analyze_button = st.button("⚡ Run Editorial Analysis", type="primary", use_container_width=True)

        if analyze_button:
            with st.spinner("Processing media and computing multi-head editorial predictions..."):
                pipeline = get_pipeline(checkpoint_dir)
                video_temp_path = None

                if uploaded_video is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
                    tfile.write(uploaded_video.read())
                    tfile.flush()
                    video_temp_path = tfile.name

                try:
                    result = pipeline.run(video_path=video_temp_path, transcript_input=transcript_text)
                    st.session_state["analysis_result"] = result
                except Exception as e:
                    st.session_state.pop("analysis_result", None)
                    st.error(f"Analysis failed: {str(e)}")
                finally:
                    if video_temp_path and os.path.exists(video_temp_path):
                        os.remove(video_temp_path)

        result = st.session_state.get("analysis_result")
        if result is not None:
            if result.metadata.get("model_status") != "fine_tuned":
                st.error(result.metadata.get("prediction_note", "Model heads are untrained."))
            else:
                st.info(result.metadata.get("prediction_note", ""))

            # Display Top Metrics
            st.markdown("---")
            st.subheader("📊 Editorial Intelligence Overview")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Content Quality Score", f"{result.quality_score:.1f} / 10.0")
            with m2:
                st.metric("Hook Confidence", f"{result.hook_confidence * 100:.0f}%")
            with m3:
                st.metric("Identified Drop-Offs", f"{len(result.drop_off_risks)}")
            with m4:
                st.metric("Timeline Segments", f"{len(result.segments)}")

            # Hook Callout
            st.markdown(
                f"""
                <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 16px; border-radius: 6px; margin: 16px 0;">
                    <strong style="color: #38bdf8;">🎯 Optimal Introductory Hook:</strong><br>
                    <span style="font-size: 1.1em; color: inherit;">{result.hook}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Retention Curve Chart
            st.subheader("📈 Modeled Retention Index (heuristic)")
            df_retention = pd.DataFrame(result.retention_curve)
            if not df_retention.empty:
                st.line_chart(df_retention.set_index("timestamp")["modeled_retention_index"])
                st.caption(result.metadata.get("retention_curve_note", "Heuristic index, not measured retention."))

            # Recommendations & Risks
            c_rec, c_risk = st.columns([1, 1])

            with c_rec:
                st.subheader("💡 Strategic Editing Recommendations")
                for i, imp in enumerate(result.improvements, 1):
                    st.markdown(f"**{i}.** {imp}")

            with c_risk:
                st.subheader("⚠️ Retention Drop-Off Alerts")
                if result.drop_off_risks:
                    for r in result.drop_off_risks:
                        with st.expander(f"{r.timestamp} - {r.reason} (Risk: {r.risk_score * 100:.0f}%)", expanded=True):
                            st.write(r.explanation)
                            ev = r.evidence
                            st.caption(
                                f"Speaking Rate: {ev.get('speaking_rate_wpm', 0)} WPM | "
                                f"Silence: {ev.get('silence_ratio', 0)}% | "
                                f"Fillers: {ev.get('filler_word_ratio', 0)}%"
                            )
                else:
                    st.success("No high-risk drop-off points detected.")

            # Segment Table
            st.subheader("⏱️ Detailed Segment Timeline")
            df_segments = pd.DataFrame(result.segments)[
                ["start_timestamp", "end_timestamp", "text", "hook_score", "risk_score", "quality_score", "wpm"]
            ]
            st.dataframe(df_segments, use_container_width=True)

            # Report Downloads
            st.markdown("---")
            st.subheader("📥 Export Reports")
            d_json, d_md, d_html = st.columns(3)

            with d_json:
                st.download_button(
                    label="Download JSON Report",
                    data=ReportGenerator.to_json(result),
                    file_name="editorial_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with d_md:
                st.download_button(
                    label="Download Markdown Report",
                    data=ReportGenerator.to_markdown(result),
                    file_name="editorial_analysis.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with d_html:
                st.download_button(
                    label="Download HTML Report",
                    data=ReportGenerator.to_html(result),
                    file_name="editorial_analysis.html",
                    mime="text/html",
                    use_container_width=True,
                )

    with tab_sample:
        st.subheader("📖 Saved Example Output")
        st.warning(
            "Illustrative only: this is a saved report (examples/sample_report.json) produced by running "
            "the CLI on examples/sample_transcript.json with a checkpoint trained on synthetic data. "
            "It is not live inference on your input."
        )

        sample_json_path = PROJECT_ROOT / "examples" / "sample_report.json"
        if sample_json_path.exists():
            with open(sample_json_path, "r") as f:
                sample_data = json.load(f)
            st.json(sample_data)
        else:
            st.info("Run `python -m src.cli analyze` to generate sample artifacts in `examples/`.")

    with tab_about:
        st.subheader("🧠 System Architecture & Methodology")
        st.markdown(
            """
            ### Multi-Head Prediction Engine
            - **Text Backbone:** distilbert-base-uncased with PEFT LoRA adapters (q_lin / v_lin). Larger backbones are configurable but untested.
            - **Multimodal Feature Fusion:** Integrates acoustic energy, speaking rate (WPM), and silence duration ratio with token sequence embeddings.
            - **Multi-Task Heads:**
              1. **Hook Head:** Binary cross-entropy scoring for attention capture.
              2. **Retention Drop-off Head:** Sigmoid probability of viewer churn based on temporal pacing and lexical density.
              3. **Quality Regression Head:** Continuous 1.0–10.0 score reflecting holistic narrative cohesion.

            ### Microsoft Azure Cloud Integrations
            - **Azure Machine Learning SDK v2:** Job, environment, registration and deployment specs (dry-run by default; not yet run against a live workspace).
            - **Azure AI Foundry:** Optional azure-ai-inference client for LLM editorial critique (requires your own endpoint and key).
            """
        )


if __name__ == "__main__":
    main()

