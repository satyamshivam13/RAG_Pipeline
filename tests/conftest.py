"""Pytest configuration for tests directory."""

import os
import sys
from pathlib import Path

# Add parent directory to path so tests can import modules
parent_dir = Path(__file__).parent.parent

# TrustedHostMiddleware permits TestClient's host only in test/development mode.
os.environ.setdefault("ENVIRONMENT", "test")
sys.path.insert(0, str(parent_dir))
