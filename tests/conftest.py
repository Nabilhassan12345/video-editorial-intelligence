"""pytest configuration — ensure src/ is on sys.path regardless of working directory."""
import sys
from pathlib import Path

# Insert project root so `import src` works from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
