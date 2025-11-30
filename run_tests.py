#!/usr/bin/env python
"""
Test runner script for the project.
Usage: python run_tests.py [options]
"""
import sys
import subprocess
from pathlib import Path


def run_tests(args=None):
    """
    Run pytest with optional arguments.
    
    Args:
        args: List of additional pytest arguments.
    """
    project_root = Path(__file__).parent
    
    # Default pytest arguments
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=term",
        "--cov-report=html",
    ]
    
    # Add custom arguments if provided
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    print(f"From directory: {project_root}\n")
    
    # Run pytest
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print(f"📊 Coverage report generated at: {project_root / 'htmlcov' / 'index.html'}")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(result.returncode)


if __name__ == "__main__":
    # Pass any command-line arguments to pytest
    run_tests(sys.argv[1:])
