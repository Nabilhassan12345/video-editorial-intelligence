# 🎬 Video Editorial Intelligence Report

> **Generated:** 2026-11-11 11:42:55 UTC  
> **Model Backbone:** `fine_tuned_checkpoint (checkpoints/editorial_model)`  
> **Media Analyzed:** `sample_video.mp4`  
> **Total Duration:** `75.0s` | **Segments:** `5`

---

## 📊 Executive Editorial Summary

- **Overall Content Quality Score:** `6.0 / 10.0`
- **Optimal Hook Identified:** [00:00] "In the next three minutes, I will show you the exact strategy that doubled our average watch duration." *(Confidence: `32.0%`)*
- **Identified Retention Risks:** `5` high-priority drop-off moment(s)

### 🎯 Recommended Strategic Edits

1. Accelerate pacing at [00:00]: Condense dialogue or jump-cut pauses to raise speaking tempo closer to 150 WPM.
2. Accelerate pacing at [00:12]: Condense dialogue or jump-cut pauses to raise speaking tempo closer to 150 WPM.
3. Accelerate pacing at [00:26]: Condense dialogue or jump-cut pauses to raise speaking tempo closer to 150 WPM.

---

## ⚠️ Critical Drop-Off Risks & Grounded Evidence

| Timestamp | Churn Risk | Primary Diagnostic | Grounded Evidence |
| :--- | :---: | :--- | :--- |
| **[00:00]** | `40.0%` | **Speaking rate stagnation** | Silence: 0.0% | Pace: 90.0 WPM | Fillers: 0.0% |
| **[00:12]** | `40.0%` | **Speaking rate stagnation** | Silence: 0.0% | Pace: 68.57142857142857 WPM | Fillers: 0.0% |
| **[00:26]** | `65.5%` | **Speaking rate stagnation, High filler word density** | Silence: 0.0% | Pace: 60.0 WPM | Fillers: 16.7% |
| **[00:44]** | `45.0%` | **Dead air / excessive silence, Speaking rate stagnation** | Silence: 100.0% | Pace: 0.0 WPM | Fillers: 0.0% |
| **[00:58]** | `45.0%` | **Dead air / excessive silence, Speaking rate stagnation** | Silence: 100.0% | Pace: 0.0 WPM | Fillers: 0.0% |

---

## ⏱️ Segment-by-Segment Timeline Analysis

| Window | Segment Text | Hook Score | Churn Risk | Pacing (WPM) | Silence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `[00:00] - [00:12]` | In the next three minutes, I will show you the exact strategy that dou... | `0.37` | `0.30` | `90.0` | `0.0%` |
| `[00:12] - [00:26]` | Most creators spend 80% of their time color grading and selecting musi... | `0.32` | `0.27` | `68.6` | `0.0%` |
| `[00:26] - [00:44]` | Um, so basically... let me find my notes here... uh, give me one secon... | `0.12` | `0.66` | `60.0` | `0.0%` |
| `[00:44] - [00:58]` | Here is the key insight: retention decay accelerates whenever you intr... | `0.27` | `0.29` | `0.0` | `100.0%` |
| `[00:58] - [01:15]` | Apply this three-step framework in your next cut, and watch your 30-se... | `0.29` | `0.30` | `0.0` | `100.0%` |

---

## 🔍 Model Transparency & Engineering Limitations

- **Model status:** fine_tuned. Scores come from a checkpoint fine-tuned on the bundled synthetic template dataset; they are editorial estimates, not measured audience retention.
- **Training data:** The bundled checkpoint is trained on a small synthetic template dataset, not on real viewer analytics. Scores are editorial estimates, not retention measurements.
- **Calibration:** Drop-off and hook scores are uncalibrated; the 0.40 risk threshold is a fixed heuristic.
- **Retention index:** modeled_retention_index is a heuristic index (100 = start) accumulated from per-segment drop-off scores. It is not a measured or calibrated audience-retention percentage.
- **Dependencies:** Acoustic features require FFmpeg; without it they are estimated from the transcript.