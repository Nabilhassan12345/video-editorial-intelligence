"""
Label derivation from real audience-retention curves.

A retention curve is a 1 Hz series where index 0 = 100 (% of viewers remaining
at each second of playback). From it we derive the three target labels the
multi-head model predicts:

- hook_label: 1.0 when the video keeps >= HOOK_RETENTION_THRESHOLD of viewers
  through the hook window (first HOOK_WINDOW_SECONDS), else 0.0. On short-form
  platforms (TikTok/Shorts) the median 3-second retention is roughly 10%, so
  the default threshold is 0.20 — about the top quartile of the corpus — rather
  than a long-form figure. Tunable; recorded in collection metadata.
- drop_off_risk: the largest relative per-second loss in the curve, in [0, 1].
  This is a real measurement of churn, not a heuristic.
- quality_score: 1.0-10.0 mapped from the curve's normalised area (mean
  retained fraction). Datasets like TikTok Top Ads saturate their public
  percentile at 99, so retention AUC is the discriminative signal; other
  sources may pass a real percentile via quality_from_percentile instead.
"""

HOOK_WINDOW_SECONDS = 3
HOOK_RETENTION_THRESHOLD = 0.20


def hook_label_from_curve(curve, threshold: float = HOOK_RETENTION_THRESHOLD,
                          window: int = HOOK_WINDOW_SECONDS) -> float:
    """1.0 if the clip retains >= threshold of viewers after the hook window."""
    if not curve or len(curve) < 2:
        return 0.0
    base = curve[0] or 100.0
    idx = min(window, len(curve) - 1)
    retained = (curve[idx] / base) if base else 0.0
    return 1.0 if retained >= threshold else 0.0


def drop_off_risk_from_curve(curve) -> float:
    """Largest single-second relative audience loss, in [0, 1]."""
    if not curve or len(curve) < 2:
        return 0.0
    worst = 0.0
    for i in range(1, len(curve)):
        prev = curve[i - 1]
        if prev and prev > 0:
            rel = (prev - curve[i]) / prev
            if rel > worst:
                worst = rel
    return float(max(0.0, min(1.0, worst)))


def curve_auc(curve) -> float:
    """Normalised area under the retention curve, in [0, 1] (mean retained %)."""
    if not curve:
        return 0.0
    base = curve[0] or 100.0
    vals = [max(0.0, min(1.0, v / base)) for v in curve] if base else []
    return float(sum(vals) / len(vals)) if vals else 0.0


def quality_from_curve(curve) -> float:
    """1.0-10.0 quality score from retention AUC (real engagement signal)."""
    return round(1.0 + 9.0 * curve_auc(curve), 2)


def quality_from_percentile(percentile) -> float:
    """Map a 0-99 percentile to a 1.0-10.0 quality score."""
    try:
        p = float(percentile)
    except (TypeError, ValueError):
        return 5.0
    p = max(0.0, min(99.0, p))
    return round(1.0 + 9.0 * (p / 99.0), 2)


def first_drop_off_second(curve):
    """Second index of the largest single-second loss (for evidence/reporting)."""
    if not curve or len(curve) < 2:
        return None
    worst, worst_i = 0.0, None
    for i in range(1, len(curve)):
        prev = curve[i - 1]
        if prev and prev > 0:
            rel = (prev - curve[i]) / prev
            if rel > worst:
                worst, worst_i = rel, i
    return worst_i
