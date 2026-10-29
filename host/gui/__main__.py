"""
Entry point for PyInstaller compilation
Run with: python -m gui_app
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == '__main__':
    main()
