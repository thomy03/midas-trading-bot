#!/usr/bin/env python
"""
Test Runner for TradingBot V3

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py unit         # Run unit tests only
    python run_tests.py integration  # Run integration tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py -v           # Verbose output
    python run_tests.py -k "test_rsi" # Run tests matching pattern
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent


def run_tests(
    test_type: str = 'all',
    verbose: bool = False,
    coverage: bool = False,
    pattern: str = None,
    fail_fast: bool = False
):
    """
    Run tests with specified options

    Args:
        test_type: 'all', 'unit', or 'integration'
        verbose: Enable verbose output
        coverage: Generate coverage report
        pattern: Filter tests by pattern (-k)
        fail_fast: Stop on first failure (-x)
    """
    project_root = get_project_root()
    tests_dir = project_root / 'tests'

    # Base pytest command
    cmd = [sys.executable, '-m', 'pytest']

    # Add test directory/marker based on type
    if test_type == 'unit':
        cmd.extend([str(tests_dir / 'unit'), '-m', 'unit'])
    elif test_type == 'integration':
        cmd.extend([str(tests_dir / 'integration'), '-m', 'integration'])
    else:  # all
        cmd.append(str(tests_dir))

    # Add options
    if verbose:
        cmd.append('-v')

    if coverage:
        cmd.extend([
            '--cov=src',
            '--cov=trendline_analysis',
            '--cov-report=term-missing',
            '--cov-report=html:coverage_report'
        ])

    if pattern:
        cmd.extend(['-k', pattern])

    if fail_fast:
        cmd.append('-x')

    # Add color output
    cmd.append('--color=yes')

    # Print command being run
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run tests
    result = subprocess.run(cmd, cwd=project_root)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='TradingBot V3 Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    Run all tests
  python run_tests.py unit               Run unit tests only
  python run_tests.py integration        Run integration tests only
  python run_tests.py -v integration     Run integration tests verbosely
  python run_tests.py --coverage         Run with coverage report
  python run_tests.py -k "rsi"           Run tests matching 'rsi'
  python run_tests.py -x                 Stop on first failure
        """
    )

    parser.add_argument(
        'test_type',
        nargs='?',
        default='all',
        choices=['all', 'unit', 'integration'],
        help='Type of tests to run (default: all)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )

    parser.add_argument(
        '-k', '--pattern',
        type=str,
        help='Filter tests by pattern'
    )

    parser.add_argument(
        '-x', '--fail-fast',
        action='store_true',
        help='Stop on first failure'
    )

    args = parser.parse_args()

    exit_code = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        pattern=args.pattern,
        fail_fast=args.fail_fast
    )

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
