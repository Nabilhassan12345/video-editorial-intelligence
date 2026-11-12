"""
Report generation module for Video Editorial Intelligence.

Exports:
- Machine-readable JSON
- Human-readable Markdown (.md)
- Standalone styled HTML (.html)
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional

from .video_pipeline import PipelineResult


class ReportGenerator:
    """Generates structured editorial intelligence reports across multiple formats."""

    @staticmethod
    def to_json(result: PipelineResult, indent: int = 2) -> str:
        """Export analysis result as JSON."""
        return json.dumps(result.to_dict(), indent=indent)

    @staticmethod
    def to_markdown(result: PipelineResult, title: str = "Video Editorial Intelligence Report") -> str:
        """Format analysis into a GitHub Flavored Markdown report."""
        meta = result.metadata
        md = [
            f"# 🎬 {title}",
            f"\n> **Generated:** {meta.get('analyzed_at', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))}  ",
            f"> **Model Backbone:** `{meta.get('model_source', 'VideoEditorialTransformer')}`  ",
            f"> **Media Analyzed:** `{meta.get('video_filename', 'Direct Transcript Ingestion')}`  ",
            f"> **Total Duration:** `{meta.get('total_duration_sec', 0):.1f}s` | **Segments:** `{meta.get('segment_count', len(result.segments))}`",
        ]
        if meta.get("model_status") and meta.get("model_status") != "fine_tuned":
            md.append("\n> **WARNING:** " + meta.get("prediction_note", "Model heads are untrained; scores are not meaningful."))
        md += [
            "\n---\n",
            "## 📊 Executive Editorial Summary",
            f"\n- **Overall Content Quality Score:** `{result.quality_score:.1f} / 10.0`",
            f"- **Optimal Hook Identified:** {result.hook} *(Confidence: `{result.hook_confidence * 100:.1f}%`)*",
            f"- **Identified Retention Risks:** `{len(result.drop_off_risks)}` high-priority drop-off moment(s)",
            "\n### 🎯 Recommended Strategic Edits\n",
        ]

        for i, imp in enumerate(result.improvements, 1):
            md.append(f"{i}. {imp}")

        md.extend([
            "\n---\n",
            "## ⚠️ Critical Drop-Off Risks & Grounded Evidence\n",
        ])

        if result.drop_off_risks:
            md.append("| Timestamp | Churn Risk | Primary Diagnostic | Grounded Evidence |")
            md.append("| :--- | :---: | :--- | :--- |")
            for r in result.drop_off_risks:
                ev = r.evidence
                ev_str = f"Silence: {ev.get('silence_ratio', 0)}% | Pace: {ev.get('speaking_rate_wpm', 0)} WPM | Fillers: {ev.get('filler_word_ratio', 0)}%"
                md.append(f"| **{r.timestamp}** | `{r.risk_score * 100:.1f}%` | **{r.reason}** | {ev_str} |")
        else:
            md.append("✅ *No severe retention risks detected across timeline.*")

        md.extend([
            "\n---\n",
            "## ⏱️ Segment-by-Segment Timeline Analysis\n",
            "| Window | Segment Text | Hook Score | Churn Risk | Pacing (WPM) | Silence |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ])

        for s in result.segments:
            text_preview = s["text"][:70] + ("..." if len(s["text"]) > 70 else "")
            md.append(
                f"| `{s['start_timestamp']} - {s['end_timestamp']}` | {text_preview} | "
                f"`{s['hook_score']:.2f}` | `{s['risk_score']:.2f}` | `{s['wpm']}` | `{s['silence_ratio']:.1%}` |"
            )

        md.extend([
            "\n---\n",
            "## 🔍 Model Transparency & Engineering Limitations\n",
            f"- **Model status:** {meta.get('model_status', 'unknown')}. {meta.get('prediction_note', '')}",
            "- **Training data:** The bundled checkpoint is trained on a small synthetic template dataset, not on real viewer analytics. Scores are editorial estimates, not retention measurements.",
            "- **Calibration:** Drop-off and hook scores are uncalibrated; the 0.40 risk threshold is a fixed heuristic.",
            "- **Retention index:** " + meta.get("retention_curve_note", "Heuristic index, not measured retention."),
            "- **Dependencies:** Acoustic features require FFmpeg; without it they are estimated from the transcript.",
        ])

        return "\n".join(md)

    @staticmethod
    def to_html(result: PipelineResult, title: str = "Video Editorial Intelligence Report") -> str:
        """Generate a modern, self-contained HTML report with CSS styling."""
        meta = result.metadata
        risks_html = ""
        for r in result.drop_off_risks:
            ev = r.evidence
            risks_html += f"""
            <div style="background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #ef4444; font-size: 1.1em;">{r.timestamp} - {r.reason}</span>
                    <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; padding: 4px 10px; border-radius: 9999px; font-size: 0.85em; font-weight: 600;">Churn Risk: {r.risk_score * 100:.1f}%</span>
                </div>
                <p style="color: #a1a1aa; margin: 8px 0; font-size: 0.95em;">{r.explanation}</p>
                <div style="font-family: monospace; color: #71717a; font-size: 0.85em;">
                    Silence: {ev.get('silence_ratio', 0)}% &bull; Speaking Rate: {ev.get('speaking_rate_wpm', 0)} WPM &bull; Lexical Density: {ev.get('lexical_density', 0)}
                </div>
            </div>
            """

        improvements_html = "".join([f"<li style='margin-bottom: 8px; color: #e4e4e7;'>{imp}</li>" for imp in result.improvements])

        segments_rows = ""
        for s in result.segments:
            segments_rows += f"""
            <tr style="border-bottom: 1px solid #27272a;">
                <td style="padding: 10px; font-family: monospace; color: #38bdf8;">{s['start_timestamp']} - {s['end_timestamp']}</td>
                <td style="padding: 10px; color: #d4d4d8;">{s['text']}</td>
                <td style="padding: 10px; text-align: center; color: #4ade80;">{s['hook_score']:.2f}</td>
                <td style="padding: 10px; text-align: center; color: #f87171;">{s['risk_score']:.2f}</td>
                <td style="padding: 10px; text-align: center; color: #a1a1aa;">{s['wpm']}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            background-color: #09090b;
            color: #f4f4f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 32px 16px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 960px;
            width: 100%;
        }}
        .header {{
            border-bottom: 1px solid #27272a;
            padding-bottom: 24px;
            margin-bottom: 32px;
        }}
        .card {{
            background: #121215;
            border: 1px solid #27272a;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-box {{
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .metric-val {{
            font-size: 2em;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        th {{
            background: #18181b;
            padding: 12px 10px;
            text-align: left;
            color: #a1a1aa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0 0 8px 0; font-size: 2em; color: #fafafa;">🎬 {title}</h1>
            <div style="color: #71717a; font-size: 0.9em;">
                Generated: {meta.get('analyzed_at')} &bull; Model: {meta.get('model_source')} &bull; Analyzed Media: {meta.get('video_filename', 'Direct Transcript')}
            </div>
        </div>

        <div class="metric-grid">
            <div class="metric-box">
                <div style="color: #a1a1aa; font-size: 0.85em; font-weight: 600;">QUALITY SCORE</div>
                <div class="metric-val" style="color: #4ade80;">{result.quality_score:.1f}<span style="font-size: 0.5em; color: #71717a;">/10</span></div>
            </div>
            <div class="metric-box">
                <div style="color: #a1a1aa; font-size: 0.85em; font-weight: 600;">HOOK CONFIDENCE</div>
                <div class="metric-val">{result.hook_confidence * 100:.0f}%</div>
            </div>
            <div class="metric-box">
                <div style="color: #a1a1aa; font-size: 0.85em; font-weight: 600;">RETENTION RISKS</div>
                <div class="metric-val" style="color: #f87171;">{len(result.drop_off_risks)}</div>
            </div>
            <div class="metric-box">
                <div style="color: #a1a1aa; font-size: 0.85em; font-weight: 600;">SEGMENTS</div>
                <div class="metric-val">{meta.get('segment_count', len(result.segments))}</div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-top: 0; color: #38bdf8;">🎯 Identified Hook</h2>
            <p style="font-size: 1.1em; color: #e4e4e7; font-style: italic; background: #18181b; padding: 16px; border-left: 4px solid #38bdf8; border-radius: 4px;">
                {result.hook}
            </p>
        </div>

        <div class="card">
            <h2 style="margin-top: 0; color: #f59e0b;">💡 Actionable Improvements</h2>
            <ul style="padding-left: 20px;">
                {improvements_html}
            </ul>
        </div>

        <div class="card">
            <h2 style="margin-top: 0; color: #ef4444;">⚠️ Retention Drop-off Risks</h2>
            {risks_html if risks_html else "<p style='color: #4ade80;'>No severe retention risks detected.</p>"}
        </div>

        <div class="card">
            <h2 style="margin-top: 0; color: #fafafa;">⏱️ Segment Breakdown</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Window</th>
                            <th>Segment Transcript</th>
                            <th style="text-align: center;">Hook</th>
                            <th style="text-align: center;">Risk</th>
                            <th style="text-align: center;">WPM</th>
                        </tr>
                    </thead>
                    <tbody>
                        {segments_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>"""
        return html

    @classmethod
    def save_reports(
        cls,
        result: PipelineResult,
        output_dir: str,
        base_name: str = "editorial_report",
    ) -> Dict[str, Path]:
        """Save JSON, Markdown, and HTML reports to disk."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        json_path = out_p / f"{base_name}.json"
        md_path = out_p / f"{base_name}.md"
        html_path = out_p / f"{base_name}.html"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(cls.to_json(result))

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(cls.to_markdown(result))

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(cls.to_html(result))

        return {
            "json": json_path,
            "markdown": md_path,
            "html": html_path,
        }

