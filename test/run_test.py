import argparse


def main(args,):
    import unittest
    import os, sys

    TEST_BASE = os.path.dirname(os.path.abspath(__file__))
    test_cases = unittest.defaultTestLoader.discover(TEST_BASE, pattern=args.file)
    runner = unittest.runner.TextTestRunner(verbosity=args.verbosity)
    ret = runner.run(test_cases)
    sys.exit(len(ret.failures) + len(ret.errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", default="test_*.py", type=str,
    )
    parser.add_argument(
        "--verbosity", default=2, type=int,
    )
    args = parser.parse_args()
    main(args)
