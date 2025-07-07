#!/usr/bin/env python3
"""
Legacy Template Smoke Test Runner

This script runs all legacy template smoke tests to ensure
template rendering consistency during refactoring.
"""

import sys
import subprocess
from pathlib import Path


def run_smoke_tests():
    """Run all legacy template smoke tests."""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    print("🧪 Running Legacy Template Smoke Tests")
    print("=" * 50)
    
    # Run the comprehensive test
    print("\n📋 Running comprehensive legacy template tests...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        str(script_dir / "test_all_legacy_templates.py"),
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All legacy template smoke tests passed!")
        print("\n📊 Test Summary:")
        print("   - Template rendering consistency verified")
        print("   - Golden outputs established/validated")
        print("   - Ready for refactoring with confidence")
    else:
        print("❌ Some legacy template smoke tests failed!")
        print("\n🔍 Test Output:")
        print(result.stdout)
        if result.stderr:
            print("\n⚠️  Errors:")
            print(result.stderr)
        
        print("\n💡 Next Steps:")
        print("   1. Review the test failures above")
        print("   2. Check if template output has changed")
        print("   3. Either fix the regression or update golden outputs")
        print("   4. Re-run tests to verify fixes")
    
    return result.returncode == 0


def run_individual_tests():
    """Run individual legacy template tests."""
    
    script_dir = Path(__file__).parent
    
    tests = [
        ("User Analysis", "test_legacy_user_template.py"),
        ("Database Analysis", "test_legacy_database_template.py"),
        ("Topic Analysis", "test_legacy_topic_template.py"),
        ("Temporal Analysis", "test_legacy_temporal_template.py"),
    ]
    
    print("\n🔬 Running Individual Template Tests")
    print("=" * 40)
    
    all_passed = True
    
    for test_name, test_file in tests:
        print(f"\n📝 Testing {test_name} templates...")
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(script_dir / test_file),
            "-v",
            "--tb=short"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {test_name} templates passed")
        else:
            print(f"❌ {test_name} templates failed")
            print(result.stdout)
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run legacy template smoke tests")
    parser.add_argument(
        "--individual", 
        action="store_true",
        help="Run individual template tests instead of comprehensive test"
    )
    
    args = parser.parse_args()
    
    if args.individual:
        success = run_individual_tests()
    else:
        success = run_smoke_tests()
    
    sys.exit(0 if success else 1) 