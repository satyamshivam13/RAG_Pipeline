"""Pytest configuration for tests directory."""
import sys
from pathlib import Path

# Add parent directory to path so tests can import modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
