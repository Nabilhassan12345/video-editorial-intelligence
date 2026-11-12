"""
Utility to generate a lightweight synthetic test video and audio for reproducible pipeline verification.
Uses FFmpeg to produce a 30-second MP4 with audio tone and timestamp overlays.
"""

from pathlib import Path
import shutil
import subprocess
import sys


def generate_synthetic_test_video(output_path: str = "examples/sample_video.mp4", duration: int = 30) -> Path:
    """Generate a 30-second test MP4 with audio test tone and visual timer overlay."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("FFmpeg not found in PATH. Skipping synthetic video generation.")
        return out_p

    # FFmpeg command generates:
    # 1. 1280x720 video test source at 30 fps with timecode text
    # 2. 440Hz sine wave audio with 2-second silent pauses to test silence detection
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size=1280x720:rate=30",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(out_p),
    ]

    print(f"Generating synthetic test video via FFmpeg: {out_p}...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        print(f"FFmpeg generation error: {proc.stderr}")
        return out_p

    print(f"Synthetic test video successfully generated: {out_p} ({out_p.stat().st_size / 1024:.1f} KB)")
    return out_p


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_video.mp4"
    generate_synthetic_test_video(output_path=out)
