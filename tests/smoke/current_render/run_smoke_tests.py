#!/usr/bin/env python3
"""
Current System Template Smoke Test Runner

Runs comprehensive smoke tests for all current system templates to ensure
rendering consistency during refactoring.

Usage:
    python run_smoke_tests.py
    poetry run python run_smoke_tests.py
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run current system template smoke tests."""
    
    print("🧪 Running Current System Template Smoke Tests")
    print("=" * 50)
    print()
    
    # Get the test directory
    test_dir = Path(__file__).parent
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--cov=src.pepino.analysis",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov_current",
        "--cov-report=term"
    ]
    
    try:
        print("📋 Running comprehensive current template tests...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All current template smoke tests passed!")
            print()
            print("📊 Test Summary:")
            print("   • All current system templates rendered consistently")
            print("   • Golden outputs created/verified")
            print("   • Template coverage complete")
            print()
            print("💡 Next Steps:")
            print("   1. Review the test output above")
            print("   2. Check golden outputs in tests/smoke/current_render/golden_outputs/")
            print("   3. Proceed with refactoring - tests will catch any regressions")
        else:
            print("❌ Some current template smoke tests failed!")
            print()
            print("🔍 Test Output:")
            print()
            print(result.stdout)
            print()
            print("⚠️  Errors:")
            print(result.stderr)
            print()
            print("💡 Next Steps:")
            print("   1. Review the test failures above")
            print("   2. Check if template output has changed")
            print("   3. Either fix the regression or update golden outputs")
            print("   4. Re-run tests to verify fixes")
            
            return 1
            
    except Exception as e:
        print(f"❌ Failed to run smoke tests: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 