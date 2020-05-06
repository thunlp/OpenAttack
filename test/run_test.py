import argparse


def main(args,):
    import unittest
    import os, sys

    TEST_BASE = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(TEST_BASE, "..",))
    sys.path.append(repo_root)
    test_cases = unittest.defaultTestLoader.discover(TEST_BASE, pattern="test_*.py",)
    runner = unittest.runner.TextTestRunner(verbosity=args.verbosity)
    ret = runner.run(test_cases)
    sys.exit(len(ret.failures) + len(ret.errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbosity", default=2, type=int,
    )
    args = parser.parse_args()
    main(args)
