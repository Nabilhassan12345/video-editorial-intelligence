"""
Adapter for the vRetention academic dataset (flowtele/vRetention on GitHub):
real audience-retention curves for YouTube (youtube.csv) and Bilibili videos.

The CSV stores each video's `retentionCurve` as an SVG path string of points.
We parse it back to a [0,100] curve, derive hook/drop-off labels, and attach
public metadata (views, duration, category). There is no audio or transcript in
this dataset, so text/acoustic features are zeros — samples are still useful
for drop-off/hook supervision but are marked `features_complete: false`.

License: academic dataset published with a research paper; research-use only.
"""

import logging
import re
from typing import Dict, Iterator, Optional

import numpy as np

from src.data.real.labels import (
    drop_off_risk_from_curve,
    first_drop_off_second,
    hook_label_from_curve,
    quality_from_curve,
)

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/flowtele/vRetention"
LICENSE_NOTE = "vRetention academic dataset — research-use only."

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_svg_retention_path(path_str: str, num_points: int = 60) -> Optional[list]:
    """Parse a `retentionCurve` SVG path into a normalized [0,100] curve.

    The path is a sequence of commands (M/L/C) with x/y coordinates; y grows
    downward in SVG space, so we invert it. Returns None when unparseable.
    """
    if not path_str or not isinstance(path_str, str):
        return None
    nums = _NUM.findall(path_str)
    vals = [float(n) for n in nums]
    pairs = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
    if len(pairs) < 2:
        return None
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    y_min, y_max = ys.min(), ys.max()
    if y_max - y_min == 0:
        return None
    ys = 100.0 * (y_max - ys) / (y_max - y_min)  # invert: top of plot = 100%
    # resample to num_points evenly spaced points
    xi = np.linspace(xs.min(), xs.max(), num_points)
    yi = np.interp(xi, xs, ys)
    yi[0] = 100.0
    return [float(max(0.0, min(100.0, v))) for v in yi]


def collect_vretention(csv_path: str, limit: int = 10000) -> Iterator[Dict]:
    """Yield samples from a local copy of the vRetention youtube.csv."""
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=limit * 2)
    emitted = 0
    for _, row in df.iterrows():
        if emitted >= limit:
            break
        curve = parse_svg_retention_path(str(row.get("retentionCurve", "")))
        if not curve:
            continue
        duration = float(row.get("duration") or len(curve) - 1 or 1)
        views = float(row.get("viewCount") or 0)
        emitted += 1
        yield {
            "id": f"vret_{row.get('id', emitted)}",
            "group_id": f"vret_{row.get('id', emitted)}",
            "video_id": str(row.get("id", "")),
            "text": str(row.get("title") or ""),
            "feature_vector": [0.0] * 8,
            "hook_label": hook_label_from_curve(curve),
            "drop_off_risk": drop_off_risk_from_curve(curve),
            "quality_score": quality_from_curve(curve),
            "duration": round(duration, 2),
            "retention_curve": curve,
            "first_drop_off_second": first_drop_off_second(curve),
            "source": "vretention",
            "transcribed": False,
            "features_complete": False,
            "engagement": {"viewCount": views, "categoryId": row.get("categoryId")},
        }
